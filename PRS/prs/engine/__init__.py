from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from prs.checkpoint import CheckpointManager
from prs.elastic import ElasticController, NodeInfo, ReconfigurationEvent
from prs.moe import MoEExpertParallelManager
from prs.planning import LayerProfile, PipelinePlanner, PipelineTemplate, create_layer_profiles_from_model
from prs.plugin import PRSPlugin
from prs.scheduler import NodeHealth, ParallelConfig, PredictiveResourceScheduler


class EngineStatus(Enum):
    INITIALIZED = "initialized"
    RUNNING = "running"
    RECONFIGURING = "reconfiguring"
    STOPPED = "stopped"


@dataclass
class TrainingState:
    step: int = 0
    epoch: int = 0
    global_step: int = 0
    best_loss: float = float("inf")
    losses: List[float] = field(default_factory=list)
    throughput_history: List[float] = field(default_factory=list)
    status: str = EngineStatus.INITIALIZED.value

    def update(self, loss: float, throughput: float) -> None:
        self.step += 1
        self.global_step += 1
        self.losses.append(float(loss))
        self.throughput_history.append(float(throughput))
        del self.losses[:-100]
        del self.throughput_history[:-100]
        self.best_loss = min(self.best_loss, float(loss))

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "epoch": self.epoch,
            "global_step": self.global_step,
            "best_loss": self.best_loss,
            "losses": self.losses,
            "throughput_history": self.throughput_history,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TrainingState":
        return cls(
            step=int(data.get("step", 0)),
            epoch=int(data.get("epoch", 0)),
            global_step=int(data.get("global_step", 0)),
            best_loss=float(data.get("best_loss", float("inf"))),
            losses=[float(item) for item in data.get("losses", [])],
            throughput_history=[float(item) for item in data.get("throughput_history", [])],
            status=str(data.get("status", EngineStatus.INITIALIZED.value)),
        )


class PRSEngine:
    """统一编排模板、检查点、弹性恢复、MoE 和预测调度。"""

    def __init__(
        self,
        plugin: Optional[PRSPlugin] = None,
        checkpoint_dir: Path | str = ".prs_runtime/checkpoints",
        nodes: int = 4,
        gpu_memory: int = 16 * 1024 * 1024 * 1024,
    ):
        self.plugin = plugin or PRSPlugin()
        self.checkpoint_dir = Path(checkpoint_dir)
        self.nodes = int(nodes)
        self.gpu_memory = int(gpu_memory)
        self.state = TrainingState()
        self.layer_profiles: List[LayerProfile] = []
        self.planner: Optional[PipelinePlanner] = None
        self.pipeline_templates: Dict[int, PipelineTemplate] = {}
        self.current_template: Optional[PipelineTemplate] = None
        self.checkpoint_manager: Optional[CheckpointManager] = None
        self.elastic_controller = ElasticController(self.plugin.fault_tolerance_threshold)
        self.scheduler: Optional[PredictiveResourceScheduler] = None
        self.moe_manager: Optional[MoEExpertParallelManager] = None
        self.logs: List[dict] = []
        self.reconfiguration_events: List[ReconfigurationEvent] = []
        self._rng = random.Random(2026)

    @property
    def status(self) -> EngineStatus:
        return EngineStatus(self.state.status)

    def prepare(
        self,
        model: Any = None,
        optimizer: Any = None,
        dataloader: Any = None,
        lr_scheduler: Any = None,
        layer_profiles: Optional[Iterable[LayerProfile]] = None,
        nodes: Optional[int] = None,
    ) -> tuple[Any, Any, Any, Any, Any]:
        if nodes is not None:
            self.nodes = int(nodes)
        self.layer_profiles = list(layer_profiles) if layer_profiles else self._profile_model(model)
        # 论文中的离线模板库在这里生成，后续故障恢复直接按节点数选择模板。
        self.planner = PipelinePlanner(self.layer_profiles, self.gpu_memory)
        self.pipeline_templates = self.planner.create_templates(1, max(1, self.nodes))
        if not self.pipeline_templates:
            raise RuntimeError("无法生成满足显存约束的流水线模板")
        self.current_template = self.pipeline_templates[max(self.pipeline_templates)]

        self.checkpoint_manager = CheckpointManager(
            self.checkpoint_dir,
            interval=self.plugin.checkpoint_interval,
            max_to_keep=self.plugin.max_checkpoints_to_keep,
            enable_incremental=self.plugin.enable_incremental_checkpoint,
            enable_async=self.plugin.enable_async_checkpoint,
            full_interval=max(self.plugin.checkpoint_interval * 4, self.plugin.checkpoint_interval),
        )
        self.elastic_controller.initialize(self._create_nodes(self.nodes))
        self.scheduler = PredictiveResourceScheduler(
            initial_nodes=self.nodes,
            initial_config=ParallelConfig(1, self.current_template.num_stages, 1, self.plugin.num_microbatches_global),
            scheduling_interval=self.plugin.scheduling_interval,
            forecast_horizon=self.plugin.forecast_horizon,
        )
        if self.plugin.enable_moe:
            # MoE 专家适配和资源调度共用节点视图，但专家副本由独立管理器维护。
            self.moe_manager = MoEExpertParallelManager(
                num_experts=self.plugin.num_experts,
                num_layers=len(self.layer_profiles),
                num_nodes=self.nodes,
                node_capacity=self.plugin.moe_node_capacity,
                fault_threshold=1,
            )
            self._seed_moe_loads()

        restored = self.checkpoint_manager.load(model=model, optimizer=optimizer)
        if restored and "training_state" in restored:
            self.state = TrainingState.from_dict(restored["training_state"])
            self._log("INFO", f"从检查点恢复到 step={self.state.step}")

        self.state.status = EngineStatus.RUNNING.value
        self._log("INFO", f"PRS 已准备完成，模板数={len(self.pipeline_templates)}，当前阶段数={self.current_template.num_stages}")
        return model, optimizer, None, dataloader, lr_scheduler

    def execute_step(
        self,
        dataloader_iter: Any = None,
        model: Any = None,
        criterion: Any = None,
        optimizer: Any = None,
    ) -> Optional[Dict[str, Any]]:
        if self.state.status == EngineStatus.RECONFIGURING.value:
            return None

        loss = self._simulate_loss()
        throughput = self._simulate_throughput()
        self.state.update(loss, throughput)
        self.state.status = EngineStatus.RUNNING.value

        if self.scheduler:
            node_healths = self._node_healths()
            self.scheduler.update_state(self.elastic_controller.get_num_nodes(), node_healths, throughput)

        if self.moe_manager:
            self._seed_moe_loads()
            self.moe_manager.allocate_replicas(layer_id=0)
            self.moe_manager.place_replicas(layer_id=0)

        if self.checkpoint_manager and self.checkpoint_manager.should_checkpoint(self.state.step):
            metadata = self.checkpoint_manager.save_state(self.state.step, self._checkpoint_state())
            self._log("INFO", f"保存检查点 step={metadata.step} incremental={metadata.is_incremental}")

        return {
            "step": self.state.step,
            "loss": loss,
            "throughput": throughput,
            "template_id": self.current_template.template_id if self.current_template else None,
        }

    def record_training_observation(
        self,
        loss: float,
        throughput: float,
        model: Any = None,
        optimizer: Any = None,
        extra_state: Optional[Dict[str, Any]] = None,
        moe_routes: Optional[Dict[int, Dict[int, int]]] = None,
        save_checkpoint: bool = True,
    ) -> Dict[str, Any]:
        """记录真实训练循环产生的损失、吞吐和运行状态。"""
        if self.state.status == EngineStatus.RECONFIGURING.value:
            return self.get_training_stats()

        self.state.update(float(loss), float(throughput))
        self.state.status = EngineStatus.RUNNING.value

        if self.scheduler:
            self.scheduler.update_state(
                node_count=self.elastic_controller.get_num_nodes(),
                node_healths=self._node_healths(),
                throughput=float(throughput),
            )

        if self.moe_manager:
            if moe_routes:
                self.moe_manager.load_monitor.reset()
                for layer_id, routes in moe_routes.items():
                    self.moe_manager.record_batch_routing(layer_id, routes)
            else:
                self._seed_moe_loads()
            self.moe_manager.allocate_replicas(layer_id=0)
            self.moe_manager.place_replicas(layer_id=0)

        if (
            save_checkpoint
            and self.checkpoint_manager
            and self.checkpoint_manager.should_checkpoint(self.state.step)
        ):
            state = self._checkpoint_state()
            if extra_state:
                state.update(extra_state)
            self.checkpoint_manager.save(
                step=self.state.step,
                model=_unwrap_model(model),
                optimizer=optimizer,
                extra_state=state,
            )
            self._log("INFO", f"真实训练检查点已保存 step={self.state.step}")

        return self.get_training_stats()

    def run_steps(self, steps: int, fail_at: Optional[int] = None) -> List[Dict[str, Any]]:
        outputs: List[Dict[str, Any]] = []
        for index in range(int(steps)):
            result = self.execute_step()
            if result:
                outputs.append(result)
            if fail_at is not None and index + 1 == fail_at:
                failed = max(self.elastic_controller.get_current_nodes() or [0])
                self.simulate_failure(failed)
        return outputs

    def simulate_failure(self, node_id: Optional[int] = None) -> ReconfigurationEvent:
        active = self.elastic_controller.get_current_nodes()
        if not active:
            raise RuntimeError("没有可故障注入的活跃节点")
        target = int(node_id if node_id is not None else active[-1])
        self._log("WARNING", f"检测到节点故障 node={target}")
        event = self.elastic_controller.mark_failed(target)
        self.reconfigure(event=event)
        return event

    def scheduler_step(self) -> Optional[dict]:
        if not self.scheduler:
            return None
        decision = self.scheduler.step()
        # 调度器输出并行配置，engine 将 pipeline_depth 映射到已生成的流水线模板。
        target_stages = min(
            decision.target_config.pipeline_depth,
            max(self.pipeline_templates) if self.pipeline_templates else 1,
        )
        if target_stages in self.pipeline_templates and (
            not self.current_template or target_stages != self.current_template.num_stages
        ):
            self.current_template = self.pipeline_templates[target_stages]
            self._log("INFO", f"主动调度切换模板 {self.current_template.template_id}")
        else:
            self._log("INFO", "主动调度保持当前模板")
        return decision.to_dict()

    def reconfigure(
        self,
        new_num_nodes: Optional[int] = None,
        event: Optional[ReconfigurationEvent] = None,
    ) -> None:
        self.state.status = EngineStatus.RECONFIGURING.value
        remaining_nodes = event.remaining_nodes if event else self.elastic_controller.get_current_nodes()
        available_nodes = int(new_num_nodes or max(len(remaining_nodes), 1))
        # 对应论文的快速恢复路径：故障后按剩余节点数查模板，再从最新 checkpoint 恢复状态。
        template_key = self._select_template_key(available_nodes)
        self.current_template = self.pipeline_templates[template_key]
        if event:
            event.selected_template_id = self.current_template.template_id
            self.reconfiguration_events.append(event)

        if self.checkpoint_manager:
            if self.checkpoint_manager.get_latest_step() != self.state.step:
                self.checkpoint_manager.save_state(self.state.step, self._checkpoint_state())
            restored = self.checkpoint_manager.load()
            if restored and "training_state" in restored:
                self.state = TrainingState.from_dict(restored["training_state"])

        self.state.status = EngineStatus.RUNNING.value
        self.elastic_controller.finish_reconfiguration()
        self._log("INFO", f"重配置完成，切换到 {self.current_template.template_id}")

    def get_template_info(self) -> dict:
        return self.current_template.to_dict() if self.current_template else {}

    def get_training_stats(self) -> dict:
        current_loss = self.state.losses[-1] if self.state.losses else None
        throughput = self.state.throughput_history[-1] if self.state.throughput_history else 0.0
        return {
            "step": self.state.step,
            "best_loss": self.state.best_loss,
            "current_loss": current_loss,
            "avg_loss": sum(self.state.losses[-10:]) / min(10, len(self.state.losses)) if self.state.losses else None,
            "throughput": throughput,
            "status": self.state.status,
        }

    def snapshot(self) -> dict:
        checkpoints = self.checkpoint_manager.list_checkpoints() if self.checkpoint_manager else []
        nodes = self.elastic_controller.heartbeat_monitor.nodes()
        return {
            "training": self.get_training_stats(),
            "templates": [template.to_dict() for template in self.pipeline_templates.values()],
            "current_template": self.get_template_info(),
            "nodes": [node.to_dict() for node in nodes.values()],
            "checkpoints": [checkpoint.to_dict() for checkpoint in checkpoints],
            "scheduler": self.scheduler.get_statistics() if self.scheduler else {},
            "moe": self.moe_manager.get_statistics() if self.moe_manager else {},
            "events": [event.to_dict() for event in self.reconfiguration_events[-20:]],
            "logs": self.logs[-100:],
        }

    def shutdown(self) -> None:
        if self.checkpoint_manager:
            self.checkpoint_manager.close()
        self.elastic_controller.shutdown()
        self.state.status = EngineStatus.STOPPED.value
        self._log("INFO", "PRS 已关闭")

    def _profile_model(self, model: Any) -> List[LayerProfile]:
        if model is not None:
            return create_layer_profiles_from_model(model)
        return create_demo_profiles()

    def _create_nodes(self, count: int) -> List[NodeInfo]:
        nodes = []
        for node_id in range(int(count)):
            nodes.append(
                NodeInfo(
                    node_id=node_id,
                    ip=f"127.0.0.{node_id + 1}",
                    devices="0,1,2,3",
                    gpu_utilization=55.0 + node_id * 4,
                    memory_usage=40.0 + node_id * 3,
                )
            )
        return nodes

    def _select_template_key(self, available_nodes: int) -> int:
        candidates = sorted(self.pipeline_templates)
        feasible = [key for key in candidates if key <= available_nodes]
        return feasible[-1] if feasible else candidates[0]

    def _checkpoint_state(self) -> dict:
        return {
            "training_state": self.state.to_dict(),
            "template_id": self.current_template.template_id if self.current_template else None,
            "templates": {key: value.to_dict() for key, value in self.pipeline_templates.items()},
        }

    def _simulate_loss(self) -> float:
        base = 4.0 * math_exp_decay(self.state.step, 0.035) + 0.18
        return max(0.05, base + self._rng.uniform(-0.04, 0.04))

    def _simulate_throughput(self) -> float:
        stages = self.current_template.num_stages if self.current_template else 1
        balance = 1.0 - min(self.current_template.balance_score() if self.current_template else 0.0, 0.5)
        return round(1.6 * stages * balance + self._rng.uniform(-0.08, 0.08), 3)

    def _node_healths(self) -> Dict[int, NodeHealth]:
        healths: Dict[int, NodeHealth] = {}
        for node_id, node in self.elastic_controller.heartbeat_monitor.nodes().items():
            healths[node_id] = NodeHealth(
                node_id=node_id,
                is_alive=node.status.value == "up",
                bandwidth=max(10.0, 100.0 - node_id * 5),
                gpu_utilization=node.gpu_utilization,
                memory_usage=node.memory_usage,
                failure_probability=0.08 + node.failure_count * 0.2,
            )
        return healths

    def _seed_moe_loads(self) -> None:
        if not self.moe_manager:
            return
        self.moe_manager.load_monitor.reset()
        for layer_id in range(min(3, len(self.layer_profiles))):
            for expert_id in range(self.plugin.num_experts):
                hot = 3 if expert_id == (self.state.step + layer_id) % self.plugin.num_experts else 1
                self.moe_manager.record_routing(layer_id, expert_id, hot * (20 + expert_id))

    def _log(self, level: str, message: str) -> None:
        self.logs.append(
            {
                "timestamp": time.strftime("%H:%M:%S"),
                "level": level,
                "message": message,
            }
        )
        del self.logs[:-200]


def math_exp_decay(step: int, rate: float) -> float:
    import math

    return math.exp(-rate * max(step, 0))


def create_demo_profiles(num_layers: int = 12) -> List[LayerProfile]:
    profiles: List[LayerProfile] = []
    for idx in range(num_layers):
        width_factor = 1.0 + (idx % 4) * 0.15
        profiles.append(
            LayerProfile(
                layer_index=idx,
                layer_name=f"transformer.layer.{idx}",
                forward_time=8.0 * width_factor,
                backward_time=16.0 * width_factor,
                memory_required=int((260 + idx * 12) * 1024 * 1024),
                parameter_count=8_000_000 + idx * 250_000,
            )
        )
    return profiles


def create_demo_engine(
    checkpoint_dir: Path | str = ".prs_runtime/checkpoints",
    nodes: int = 4,
) -> PRSEngine:
    engine = PRSEngine(get_default_demo_config(), checkpoint_dir=checkpoint_dir, nodes=nodes)
    engine.prepare(layer_profiles=create_demo_profiles(), nodes=nodes)
    return engine


def get_default_demo_config() -> PRSPlugin:
    return PRSPlugin(
        global_batch_size=32,
        microbatch_size=4,
        checkpoint_interval=4,
        max_checkpoints_to_keep=12,
        enable_incremental_checkpoint=True,
        enable_async_checkpoint=False,
        scheduling_interval=120.0,
        forecast_horizon=4,
        enable_moe=True,
        num_experts=8,
        moe_node_capacity=4,
    )


def _unwrap_model(model: Any) -> Any:
    return getattr(model, "module", model)
