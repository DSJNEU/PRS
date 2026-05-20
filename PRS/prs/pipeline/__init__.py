from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class StageConfig:
    stage_id: int
    num_stages: int
    layer_indices: List[int]
    device: str = "cpu"
    prev_stage_rank: Optional[int] = None
    next_stage_rank: Optional[int] = None

    @property
    def is_first(self) -> bool:
        return self.stage_id == 0

    @property
    def is_last(self) -> bool:
        return self.stage_id == self.num_stages - 1

    def to_dict(self) -> dict:
        return {
            "stage_id": self.stage_id,
            "num_stages": self.num_stages,
            "layer_indices": self.layer_indices,
            "device": self.device,
            "is_first": self.is_first,
            "is_last": self.is_last,
            "prev_stage_rank": self.prev_stage_rank,
            "next_stage_rank": self.next_stage_rank,
        }


@dataclass
class Microbatch:
    microbatch_id: int
    data: Any = None
    labels: Any = None
    output: Any = None
    loss: Optional[float] = None
    activations: Dict[int, Any] = field(default_factory=dict)


class PipelineStage:
    def __init__(self, config: StageConfig, model_layers: Any = None):
        self.config = config
        self.model_layers = model_layers

    def forward(self, value: Any) -> Any:
        if callable(self.model_layers):
            return self.model_layers(value)
        return value


class PipelineExecutor:
    """生成 1F1B 调度序列，真实训练由 PRSEngine 编排。"""

    def __init__(self, stage: PipelineStage, num_microbatches: int):
        self.stage = stage
        self.num_microbatches = int(num_microbatches)

    def schedule(self) -> List[tuple[str, int]]:
        stage_id = self.stage.config.stage_id
        num_stages = self.stage.config.num_stages
        # 1F1B 分为 warmup、稳定 forward/backward 交替、drain 三段。
        warmup = min(num_stages - stage_id - 1, self.num_microbatches)
        schedule: List[tuple[str, int]] = []
        for microbatch in range(warmup):
            schedule.append(("forward", microbatch))
        for microbatch in range(warmup, self.num_microbatches):
            schedule.append(("forward", microbatch))
            schedule.append(("backward", microbatch - warmup))
        for microbatch in range(self.num_microbatches - warmup, self.num_microbatches):
            if microbatch >= 0:
                schedule.append(("backward", microbatch))
        return schedule


class MultiNodePipelineTrainer:
    def __init__(self, engine: Any):
        self.engine = engine

    def setup(self) -> None:
        if getattr(self.engine, "current_template", None) is None:
            self.engine.prepare()

    def train_step(self) -> Optional[Dict[str, Any]]:
        return self.engine.execute_step()

    def reconfigure(self, new_num_nodes: Optional[int] = None) -> None:
        self.engine.reconfigure(new_num_nodes=new_num_nodes)

    def shutdown(self) -> None:
        self.engine.shutdown()
