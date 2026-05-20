from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class Precision(Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"


@dataclass
class PipelineConfig:
    """论文中流水线 stage 的落地配置：阶段编号、层范围和设备位置。"""

    pipeline_id: int
    stage_id: int
    num_stages: int
    layer_indices: List[int]
    device: str = "cpu"

    def to_dict(self) -> dict:
        return {
            "pipeline_id": self.pipeline_id,
            "stage_id": self.stage_id,
            "num_stages": self.num_stages,
            "layer_indices": self.layer_indices,
            "device": self.device,
        }


@dataclass
class StageManager:
    stage_id: int
    num_stages: int
    is_first_stage: bool = field(init=False)
    is_last_stage: bool = field(init=False)
    prev_stage: Optional[int] = field(init=False)
    next_stage: Optional[int] = field(init=False)

    def __post_init__(self) -> None:
        # 根据 stage_id 推导相邻阶段，用于 1F1B 调度和故障后模板重连。
        if self.num_stages < 1:
            raise ValueError("num_stages 必须大于 0")
        if self.stage_id < 0 or self.stage_id >= self.num_stages:
            raise ValueError("stage_id 超出阶段范围")
        self.is_first_stage = self.stage_id == 0
        self.is_last_stage = self.stage_id == self.num_stages - 1
        self.prev_stage = None if self.is_first_stage else self.stage_id - 1
        self.next_stage = None if self.is_last_stage else self.stage_id + 1

    def to_dict(self) -> dict:
        return {
            "stage_id": self.stage_id,
            "num_stages": self.num_stages,
            "is_first_stage": self.is_first_stage,
            "is_last_stage": self.is_last_stage,
            "prev_stage": self.prev_stage,
            "next_stage": self.next_stage,
        }


@dataclass
class PRSPlugin:
    """PRS 全局配置，集中承接论文中的批大小、检查点、调度和 MoE 参数。"""

    global_batch_size: int = 32
    microbatch_size: int = 4
    precision: str = "fp32"
    tensor_parallel: int = 1
    fault_tolerance_threshold: int = 1
    checkpoint_interval: int = 10
    max_checkpoints_to_keep: int = 8
    enable_incremental_checkpoint: bool = True
    enable_async_checkpoint: bool = False
    scheduling_interval: float = 300.0
    forecast_horizon: int = 6
    enable_moe: bool = True
    num_experts: int = 8
    moe_node_capacity: int = 4

    pipelines: List[PipelineConfig] = field(default_factory=list, repr=False)
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # microbatch 必须整除 global batch，否则流水线气泡和梯度累积口径会不一致。
        if self.global_batch_size <= 0:
            raise ValueError("global_batch_size 必须大于 0")
        if self.microbatch_size <= 0:
            raise ValueError("microbatch_size 必须大于 0")
        if self.global_batch_size < self.microbatch_size:
            raise ValueError("global_batch_size 必须不小于 microbatch_size")
        if self.global_batch_size % self.microbatch_size != 0:
            raise ValueError("global_batch_size 必须能被 microbatch_size 整除")
        if self.precision not in {item.value for item in Precision}:
            raise ValueError("precision 必须是 fp32、fp16 或 bf16")
        if self.fault_tolerance_threshold < 1:
            raise ValueError("fault_tolerance_threshold 必须大于 0")

    @property
    def num_microbatches_global(self) -> int:
        return self.global_batch_size // self.microbatch_size

    def set_pipelines(self, pipelines: List[PipelineConfig]) -> None:
        self.pipelines = list(pipelines)

    def get_precision_dtype(self) -> object:
        try:
            import torch

            return {
                Precision.FP32.value: torch.float32,
                Precision.FP16.value: torch.float16,
                Precision.BF16.value: torch.bfloat16,
            }[self.precision]
        except Exception:
            return self.precision

    def to_dict(self) -> dict:
        return {
            "global_batch_size": self.global_batch_size,
            "microbatch_size": self.microbatch_size,
            "precision": self.precision,
            "tensor_parallel": self.tensor_parallel,
            "fault_tolerance_threshold": self.fault_tolerance_threshold,
            "checkpoint_interval": self.checkpoint_interval,
            "max_checkpoints_to_keep": self.max_checkpoints_to_keep,
            "enable_incremental_checkpoint": self.enable_incremental_checkpoint,
            "enable_async_checkpoint": self.enable_async_checkpoint,
            "scheduling_interval": self.scheduling_interval,
            "forecast_horizon": self.forecast_horizon,
            "enable_moe": self.enable_moe,
            "num_experts": self.num_experts,
            "moe_node_capacity": self.moe_node_capacity,
            "num_microbatches_global": self.num_microbatches_global,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PRSPlugin":
        fields = cls.__dataclass_fields__.keys()
        return cls(**{key: value for key, value in data.items() if key in fields})


def get_default_config() -> PRSPlugin:
    return PRSPlugin()


def get_large_model_config() -> PRSPlugin:
    return PRSPlugin(
        global_batch_size=128,
        microbatch_size=8,
        precision="bf16",
        fault_tolerance_threshold=2,
        checkpoint_interval=20,
        forecast_horizon=12,
        enable_moe=True,
        num_experts=16,
        moe_node_capacity=4,
    )


def get_debug_config() -> PRSPlugin:
    return PRSPlugin(
        global_batch_size=8,
        microbatch_size=2,
        precision="fp32",
        checkpoint_interval=2,
        enable_incremental_checkpoint=True,
        enable_async_checkpoint=False,
        scheduling_interval=30.0,
        forecast_horizon=4,
        num_experts=4,
    )
