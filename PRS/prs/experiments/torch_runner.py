from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from prs.engine import PRSEngine
from prs.experiments.metrics import MetricLogger
from prs.planning import create_layer_profiles_from_model
from prs.plugin import PRSPlugin


class InjectedFault(RuntimeError):
    pass


@dataclass
class FaultInjectionConfig:
    enabled: bool = False
    step: int = -1
    rank: int = 0
    mode: str = "exception"

    def should_fire(self, step: int, rank: int) -> bool:
        return self.enabled and step == self.step and rank == self.rank


@dataclass
class TorchTrainingConfig:
    output_dir: str = ".prs_runtime/torch_experiment"
    model_source: str = "synthetic"
    hf_model_name: str = ""
    trust_remote_code: bool = True
    total_steps: int = 20
    batch_size: int = 2
    sequence_length: int = 64
    vocab_size: int = 4096
    hidden_size: int = 128
    num_layers: int = 2
    num_heads: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    checkpoint_interval: int = 5
    scheduler_interval_steps: int = 10
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = False
    seed: int = 2026
    prs_nodes: Optional[int] = None
    enable_moe_adaptation: bool = True
    num_experts: int = 8
    fault: FaultInjectionConfig = field(default_factory=FaultInjectionConfig)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["fault"] = asdict(self.fault)
        return data


class TorchExperimentRunner:
    """真实 PyTorch 训练入口，可由 torchrun 单机或多机启动。"""

    def __init__(self, config: TorchTrainingConfig):
        self.config = config
        self.torch = _import_torch()
        self.rank = int(os.environ.get("RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.device = self._select_device()
        self.distributed = False
        self.metric_logger = MetricLogger(Path(config.output_dir), rank=self.rank)
        self.engine: Optional[PRSEngine] = None

    def run(self) -> Dict[str, Any]:
        start_time = time.perf_counter()
        self._setup_distributed()
        self._set_seed()
        model = self._build_model().to(self.device)
        self._cached_model_ref = model
        if self.config.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()

        optimizer = self.torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        profiles = create_layer_profiles_from_model(model)
        plugin = PRSPlugin(
            global_batch_size=max(self.config.batch_size * self.world_size, 1),
            microbatch_size=max(self.config.batch_size, 1),
            checkpoint_interval=self.config.checkpoint_interval,
            max_checkpoints_to_keep=16,
            enable_incremental_checkpoint=True,
            enable_moe=self.config.enable_moe_adaptation,
            num_experts=self.config.num_experts,
        )
        self.engine = PRSEngine(
            plugin=plugin,
            checkpoint_dir=Path(self.config.output_dir) / "checkpoints",
            nodes=self.config.prs_nodes or max(self.world_size, 1),
        )
        # 真实训练层把 PyTorch 模型画像交给 PRSEngine，后者负责模板、checkpoint 和调度。
        self.engine.prepare(model=model, optimizer=optimizer, layer_profiles=profiles)
        resumed_step = self.engine.state.step
        recovery_time = time.perf_counter() - start_time if resumed_step else 0.0

        train_model = self._wrap_ddp(model)
        completed_steps = self._train_loop(train_model, optimizer, resumed_step)
        summary = self.metric_logger.summarize(
            {
                "world_size": self.world_size,
                "device": str(self.device),
                "model_source": self.config.model_source,
                "hf_model_name": self.config.hf_model_name,
                "completed_steps": completed_steps,
                "resumed_step": resumed_step,
                "recovery_time": recovery_time,
                "prs_snapshot": self.engine.snapshot() if self.rank == 0 else None,
            }
        )
        self.barrier()
        self.metric_logger.close()
        self._cleanup_distributed()
        return summary

    def _train_loop(self, model: Any, optimizer: Any, start_step: int) -> int:
        model.train()
        step = int(start_step)
        while step < self.config.total_steps:
            if self._should_inject_fault(step + 1):
                # 故障注入前强制保存 checkpoint，对应论文中的故障前状态保护。
                self._save_pre_fault_checkpoint(model, optimizer)
                if self.config.fault.mode == "exit":
                    os._exit(42)
                raise InjectedFault(f"注入故障: rank={self.rank}, step={step + 1}")

            step_start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            total_loss = None
            for _ in range(self.config.gradient_accumulation_steps):
                batch = self._random_batch()
                output = model(**batch)
                loss = output["loss"] if isinstance(output, dict) else output.loss
                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()
                total_loss = loss if total_loss is None else total_loss + loss.detach()
            optimizer.step()

            step_time = time.perf_counter() - step_start
            step += 1
            loss_value = self._mean_scalar(float(total_loss.detach().item()))
            samples = self.config.batch_size * self.world_size * self.config.gradient_accumulation_steps
            throughput = samples / max(step_time, 1e-9)
            # 多卡时取各 rank 的均值，保证 metrics 与 PRS 调度输入口径一致。
            throughput = self._mean_scalar(throughput)

            if self.engine:
                self.engine.record_training_observation(
                    loss=loss_value,
                    throughput=throughput,
                    model=model,
                    optimizer=optimizer,
                    moe_routes=self._synthetic_moe_routes(step),
                    extra_state={"experiment_config": self.config.to_dict()},
                    # 多 rank 共享输出目录时只让 rank 0 写检查点，避免 metadata 竞争写入。
                    save_checkpoint=self.rank == 0,
                )
                if step % max(self.config.scheduler_interval_steps, 1) == 0:
                    self.engine.scheduler_step()

            if self.rank == 0:
                template_id = (
                    self.engine.current_template.template_id
                    if self.engine and self.engine.current_template else None
                )
                self.metric_logger.log(
                    step=step,
                    loss=loss_value,
                    throughput=throughput,
                    step_time=step_time,
                    template_id=template_id,
                )

        return step

    def barrier(self) -> None:
        if self.distributed and self.torch.distributed.is_initialized():
            if self.device.type == "cuda":
                self.torch.distributed.barrier(device_ids=[self.local_rank])
            else:
                self.torch.distributed.barrier()

    def _setup_distributed(self) -> None:
        if self.world_size <= 1:
            return
        dist = self.torch.distributed
        if dist.is_initialized():
            self.distributed = True
            return
        backend = "nccl" if self.device.type == "cuda" else "gloo"
        dist.init_process_group(backend=backend)
        self.distributed = True

    def _cleanup_distributed(self) -> None:
        if self.distributed and self.torch.distributed.is_initialized():
            self.torch.distributed.destroy_process_group()

    def _wrap_ddp(self, model: Any) -> Any:
        if not self.distributed:
            return model
        ddp = self.torch.nn.parallel.DistributedDataParallel
        if self.device.type == "cuda":
            return ddp(model, device_ids=[self.local_rank], output_device=self.local_rank)
        return ddp(model)

    def _select_device(self) -> Any:
        if self.torch.cuda.is_available():
            self.torch.cuda.set_device(self.local_rank)
            return self.torch.device("cuda", self.local_rank)
        return self.torch.device("cpu")

    def _set_seed(self) -> None:
        seed = self.config.seed + self.rank
        self.torch.manual_seed(seed)
        if self.torch.cuda.is_available():
            self.torch.cuda.manual_seed_all(seed)

    def _build_model(self) -> Any:
        if self.config.model_source == "hf":
            try:
                from transformers import AutoModelForCausalLM
            except ImportError as exc:
                raise RuntimeError("使用 HuggingFace 模型需要安装 transformers") from exc
            if not self.config.hf_model_name:
                raise ValueError("model_source=hf 时必须提供 hf_model_name")
            return AutoModelForCausalLM.from_pretrained(
                self.config.hf_model_name,
                trust_remote_code=self.config.trust_remote_code,
            )
        return TinyCausalLM(
            vocab_size=self.config.vocab_size,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            num_heads=self.config.num_heads,
            max_position=self.config.sequence_length,
        )

    def _random_batch(self) -> Dict[str, Any]:
        model = getattr(self, "_cached_model_ref", None)
        vocab_size = self.config.vocab_size
        if self.config.model_source == "hf" and model is not None:
            vocab_size = int(getattr(model.config, "vocab_size", vocab_size))
        tokens = self.torch.randint(
            low=0,
            high=vocab_size,
            size=(self.config.batch_size, self.config.sequence_length + 1),
            device=self.device,
        )
        return {"input_ids": tokens[:, :-1], "labels": tokens[:, 1:].contiguous()}

    def _mean_scalar(self, value: float) -> float:
        if not self.distributed:
            return float(value)
        tensor = self.torch.tensor(float(value), device=self.device)
        self.torch.distributed.all_reduce(tensor, op=self.torch.distributed.ReduceOp.SUM)
        tensor /= self.world_size
        return float(tensor.item())

    def _synthetic_moe_routes(self, step: int) -> Dict[int, Dict[int, int]]:
        if not self.config.enable_moe_adaptation:
            return {}
        routes: Dict[int, Dict[int, int]] = {}
        for layer_id in range(min(self.config.num_layers, 3)):
            routes[layer_id] = {}
            for expert_id in range(self.config.num_experts):
                hot = 3 if expert_id == (step + layer_id) % self.config.num_experts else 1
                routes[layer_id][expert_id] = hot * self.config.batch_size * self.config.sequence_length
        return routes

    def _save_pre_fault_checkpoint(self, model: Any, optimizer: Any) -> None:
        self._fault_marker_path().write_text("done", encoding="utf-8")
        if self.engine and self.engine.checkpoint_manager:
            self.engine.checkpoint_manager.save(
                step=max(self.engine.state.step, 0),
                model=getattr(model, "module", model),
                optimizer=optimizer,
                extra_state={"training_state": self.engine.state.to_dict()},
            )
            self.metric_logger.log(event="fault_injected", step=self.engine.state.step, rank=self.rank)

    def _should_inject_fault(self, step: int) -> bool:
        # fault marker 让同一输出目录恢复运行时不会重复注入同一个故障。
        return self.config.fault.should_fire(step, self.rank) and not self._fault_marker_path().exists()

    def _fault_marker_path(self) -> Path:
        return (
            Path(self.config.output_dir)
            / f"fault_step{self.config.fault.step}_rank{self.config.fault.rank}.done"
        )


class TinyCausalLM:
    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        torch = _import_torch()

        class _TinyCausalLM(torch.nn.Module):
            def __init__(self, vocab_size: int, hidden_size: int, num_layers: int, num_heads: int, max_position: int):
                super().__init__()
                self.config = type("TinyConfig", (), {"vocab_size": vocab_size})()
                self.token_embed = torch.nn.Embedding(vocab_size, hidden_size)
                self.position_embed = torch.nn.Embedding(max_position, hidden_size)
                layer = torch.nn.TransformerEncoderLayer(
                    d_model=hidden_size,
                    nhead=num_heads,
                    dim_feedforward=hidden_size * 4,
                    dropout=0.0,
                    batch_first=True,
                    activation="gelu",
                )
                self.layers = torch.nn.TransformerEncoder(layer, num_layers=num_layers)
                self.norm = torch.nn.LayerNorm(hidden_size)
                self.lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False)

            def forward(self, input_ids: Any, labels: Optional[Any] = None) -> Dict[str, Any]:
                seq_len = input_ids.shape[1]
                positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
                hidden = self.token_embed(input_ids) + self.position_embed(positions)
                hidden = self.layers(hidden)
                logits = self.lm_head(self.norm(hidden))
                loss = None
                if labels is not None:
                    loss = torch.nn.functional.cross_entropy(
                        logits.reshape(-1, logits.shape[-1]),
                        labels.reshape(-1),
                    )
                return {"loss": loss, "logits": logits}

        return _TinyCausalLM(*args, **kwargs)


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("真实训练需要安装 PyTorch：pip install -e .[torch]") from exc
    return torch
