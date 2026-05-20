from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional


@dataclass(frozen=True)
class LayerProfile:
    """模型层级性能画像。"""

    layer_index: int
    layer_name: str
    forward_time: float
    backward_time: float
    memory_required: int
    parameter_count: int = 0

    @property
    def total_time(self) -> float:
        return self.forward_time + self.backward_time

    @property
    def compute_intensity(self) -> float:
        memory_mb = self.memory_required / (1024 * 1024)
        return self.total_time / memory_mb if memory_mb else 0.0

    def to_dict(self) -> dict:
        return {
            "layer_index": self.layer_index,
            "layer_name": self.layer_name,
            "forward_time": self.forward_time,
            "backward_time": self.backward_time,
            "memory_required": self.memory_required,
            "parameter_count": self.parameter_count,
            "total_time": self.total_time,
            "compute_intensity": self.compute_intensity,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LayerProfile":
        return cls(
            layer_index=int(data["layer_index"]),
            layer_name=str(data["layer_name"]),
            forward_time=float(data["forward_time"]),
            backward_time=float(data["backward_time"]),
            memory_required=int(data["memory_required"]),
            parameter_count=int(data.get("parameter_count", 0)),
        )


@dataclass
class PipelineTemplate:
    """可在故障后直接实例化的流水线模板。"""

    template_id: str
    num_stages: int
    stage_layers: List[List[int]]
    estimated_latency: float
    stage_latencies: List[float] = field(default_factory=list)
    stage_memories: List[int] = field(default_factory=list)
    node_mapping: Dict[int, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.stage_latencies:
            self.stage_latencies = [0.0] * self.num_stages
        if not self.stage_memories:
            self.stage_memories = [0] * self.num_stages
        if not self.node_mapping:
            self.node_mapping = {stage_id: stage_id for stage_id in range(self.num_stages)}

    def latency(self, num_microbatches: int) -> float:
        return (num_microbatches + self.num_stages - 1) * self.estimated_latency

    def balance_score(self) -> float:
        if not self.stage_latencies:
            return 0.0
        mean = sum(self.stage_latencies) / len(self.stage_latencies)
        if mean == 0:
            return 0.0
        variance = sum((value - mean) ** 2 for value in self.stage_latencies)
        return (variance / len(self.stage_latencies)) ** 0.5 / mean

    def get_stage_for_layer(self, layer_index: int) -> Optional[int]:
        for stage_id, layers in enumerate(self.stage_layers):
            if layer_index in layers:
                return stage_id
        return None

    def to_dict(self) -> dict:
        return {
            "template_id": self.template_id,
            "num_stages": self.num_stages,
            "stage_layers": self.stage_layers,
            "estimated_latency": self.estimated_latency,
            "stage_latencies": self.stage_latencies,
            "stage_memories": self.stage_memories,
            "node_mapping": self.node_mapping,
            "balance_score": self.balance_score(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PipelineTemplate":
        mapping = data.get("node_mapping", {})
        return cls(
            template_id=str(data.get("template_id", f"tpl-{data['num_stages']}")),
            num_stages=int(data["num_stages"]),
            stage_layers=[list(stage) for stage in data["stage_layers"]],
            estimated_latency=float(data["estimated_latency"]),
            stage_latencies=[float(v) for v in data.get("stage_latencies", [])],
            stage_memories=[int(v) for v in data.get("stage_memories", [])],
            node_mapping={int(k): int(v) for k, v in mapping.items()},
        )


class PipelinePlanner:
    """ O(L^2*N) 动态规划生成流水线模板。"""

    def __init__(self, layer_profiles: Iterable[LayerProfile], gpu_memory: int):
        self.layer_profiles = list(layer_profiles)
        if not self.layer_profiles:
            raise ValueError("layer_profiles 不能为空")
        self.gpu_memory = int(gpu_memory)
        self._build_prefix_arrays()

    def _build_prefix_arrays(self) -> None:
        # 前缀和让论文中的 segment_time / segment_memory 查询变成 O(1)。
        self._time_prefix = [0.0]
        self._memory_prefix = [0]
        for profile in self.layer_profiles:
            self._time_prefix.append(self._time_prefix[-1] + profile.total_time)
            self._memory_prefix.append(self._memory_prefix[-1] + profile.memory_required)

    def segment_time(self, start: int, end: int) -> float:
        return self._time_prefix[end] - self._time_prefix[start]

    def segment_memory(self, start: int, end: int) -> int:
        return self._memory_prefix[end] - self._memory_prefix[start]

    def create_templates(self, min_nodes: int, max_nodes: int) -> Dict[int, PipelineTemplate]:
        # 模板库提前覆盖不同节点数，故障后只查表，不在线穷举切分。
        templates: Dict[int, PipelineTemplate] = {}
        for stages in range(int(min_nodes), int(max_nodes) + 1):
            template = self.create_template(stages)
            if template:
                templates[stages] = template
        return templates

    def create_template(self, num_stages: int) -> Optional[PipelineTemplate]:
        layer_count = len(self.layer_profiles)
        if num_stages < 1 or num_stages > layer_count:
            return None

        # dp[end][stage] 保存“前 end 层切成 stage 段”时的最慢阶段耗时和切分点。
        inf = float("inf")
        dp = [[(inf, -1) for _ in range(num_stages + 1)] for _ in range(layer_count + 1)]
        dp[0][0] = (0.0, -1)

        for end in range(1, layer_count + 1):
            for stage in range(1, min(end, num_stages) + 1):
                for split in range(stage - 1, end):
                    memory = self.segment_memory(split, end)
                    if memory > self.gpu_memory:
                        continue
                    stage_time = self.segment_time(split, end)
                    worst_time = max(dp[split][stage - 1][0], stage_time)
                    if worst_time < dp[end][stage][0]:
                        dp[end][stage] = (worst_time, split)

        if dp[layer_count][num_stages][0] == inf:
            return None

        # 从最后一层向前回溯切分点，得到连续且不重叠的阶段层号。
        stages: List[List[int]] = []
        end = layer_count
        stage = num_stages
        while stage:
            split = dp[end][stage][1]
            stages.append(list(range(split, end)))
            end = split
            stage -= 1
        stages.reverse()

        stage_latencies = [self.segment_time(s[0], s[-1] + 1) for s in stages]
        stage_memories = [self.segment_memory(s[0], s[-1] + 1) for s in stages]
        return PipelineTemplate(
            template_id=f"tpl-{num_stages}stage",
            num_stages=num_stages,
            stage_layers=stages,
            estimated_latency=max(stage_latencies),
            stage_latencies=stage_latencies,
            stage_memories=stage_memories,
        )

    def validate_template(self, template: PipelineTemplate, balance_limit: float = 0.35) -> bool:
        if template.num_stages != len(template.stage_layers):
            return False
        if any(memory > self.gpu_memory for memory in template.stage_memories):
            return False
        all_layers = [layer for stage in template.stage_layers for layer in stage]
        if all_layers != list(range(len(self.layer_profiles))):
            return False
        return template.balance_score() <= balance_limit or template.num_stages == 1

    def save_templates(self, templates: Dict[int, PipelineTemplate], path: Path) -> None:
        payload = {
            "gpu_memory": self.gpu_memory,
            "layers": [profile.to_dict() for profile in self.layer_profiles],
            "templates": {str(key): value.to_dict() for key, value in templates.items()},
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def load_templates(path: Path) -> Dict[int, PipelineTemplate]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return {
            int(key): PipelineTemplate.from_dict(value)
            for key, value in payload.get("templates", {}).items()
        }


def create_layer_profiles_from_model(model: object, default_layer_time: float = 12.0) -> List[LayerProfile]:
    """从模型结构提取层级画像，对应论文中离线 profile 的输入。"""

    profiles: List[LayerProfile] = []
    children = _profile_children(model)

    if not children:
        parameter_count = _count_parameters(model)
        return [
            LayerProfile(
                layer_index=0,
                layer_name="model",
                forward_time=default_layer_time,
                backward_time=default_layer_time * 2,
                memory_required=max(parameter_count * 4, 1),
                parameter_count=parameter_count,
            )
        ]

    for idx, (name, module) in enumerate(children):
        parameter_count = _count_parameters(module)
        scale = max(parameter_count / 1_000_000, 0.1)
        profiles.append(
            LayerProfile(
                layer_index=idx,
                layer_name=name,
                forward_time=default_layer_time * scale,
                backward_time=default_layer_time * 2 * scale,
                memory_required=max(parameter_count * 4, 64 * 1024 * 1024),
                parameter_count=parameter_count,
            )
        )
    return profiles


def _profile_children(model: object) -> List[tuple[str, object]]:
    named_children = getattr(model, "named_children", None)
    children = list(named_children()) if callable(named_children) else []

    if not children:
        layer_stack = getattr(model, "layers", None)
        return [(f"layer.{idx}", layer) for idx, layer in _enumerate_layers(layer_stack)]

    expanded: List[tuple[str, object]] = []
    for name, module in children:
        repeated_layers = _expand_repeated_layers(name, module)
        expanded.extend(repeated_layers or [(name, module)])
    return expanded


def _expand_repeated_layers(name: str, module: object) -> List[tuple[str, object]]:
    layer_stack = getattr(module, "layers", None)
    if layer_stack is not None:
        return [
            (f"{name}.{idx}", layer)
            for idx, layer in _enumerate_layers(layer_stack)
        ]
    if name in {"layers", "layer", "blocks", "h"}:
        named_children = getattr(module, "named_children", None)
        children = list(named_children()) if callable(named_children) else []
        if children:
            return [(f"{name}.{child_name}", child) for child_name, child in children]
    return []


def _enumerate_layers(layer_stack: object) -> List[tuple[int, object]]:
    if layer_stack is None:
        return []
    try:
        return [(idx, layer) for idx, layer in enumerate(layer_stack)]
    except TypeError:
        return []


def _count_parameters(module: object) -> int:
    parameters = getattr(module, "parameters", None)
    if not callable(parameters):
        return 0
    total = 0
    for param in parameters():
        numel = getattr(param, "numel", None)
        total += int(numel()) if callable(numel) else 0
    return total
