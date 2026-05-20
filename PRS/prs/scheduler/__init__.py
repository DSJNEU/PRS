from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass
class NodeHealth:
    node_id: int
    is_alive: bool = True
    bandwidth: float = 100.0
    gpu_utilization: float = 0.0
    memory_usage: float = 0.0
    failure_probability: float = 0.0
    last_update: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "is_alive": self.is_alive,
            "bandwidth": self.bandwidth,
            "gpu_utilization": self.gpu_utilization,
            "memory_usage": self.memory_usage,
            "failure_probability": self.failure_probability,
            "last_update": self.last_update,
        }


@dataclass(frozen=True)
class ParallelConfig:
    data_parallel: int = 1
    pipeline_depth: int = 1
    tensor_parallel: int = 1
    num_microbatches: int = 8

    @property
    def required_nodes(self) -> int:
        return self.data_parallel * self.pipeline_depth * self.tensor_parallel

    def is_feasible(self, num_nodes: int) -> bool:
        return self.required_nodes <= num_nodes

    def key(self) -> str:
        return f"D{self.data_parallel}-P{self.pipeline_depth}-T{self.tensor_parallel}"

    def to_dict(self) -> dict:
        return {
            "data_parallel": self.data_parallel,
            "pipeline_depth": self.pipeline_depth,
            "tensor_parallel": self.tensor_parallel,
            "num_microbatches": self.num_microbatches,
            "required_nodes": self.required_nodes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ParallelConfig":
        return cls(
            data_parallel=int(data.get("data_parallel", 1)),
            pipeline_depth=int(data.get("pipeline_depth", 1)),
            tensor_parallel=int(data.get("tensor_parallel", 1)),
            num_microbatches=int(data.get("num_microbatches", 8)),
        )


@dataclass
class PredictionResult:
    timestamp: float
    predicted_nodes: List[int]
    confidence: List[float]

    @property
    def horizon(self) -> int:
        return len(self.predicted_nodes)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "predicted_nodes": self.predicted_nodes,
            "confidence": self.confidence,
            "horizon": self.horizon,
        }


@dataclass
class SchedulingDecision:
    timestamp: float
    current_config: ParallelConfig
    target_config: ParallelConfig
    expected_liveput: float
    migration_cost: float
    config_path: List[ParallelConfig] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "current_config": self.current_config.to_dict(),
            "target_config": self.target_config.to_dict(),
            "expected_liveput": self.expected_liveput,
            "migration_cost": self.migration_cost,
            "config_path": [config.to_dict() for config in self.config_path],
            "actions": self.actions,
        }


class ARIMANodePredictor:
    """轻量 ARIMA(2,1,1) 风格预测器，用于本地演示和调度决策。"""

    def __init__(self, forecast_horizon: int = 6, history_size: int = 64):
        self.forecast_horizon = forecast_horizon
        self.history_size = history_size
        self.node_history: List[int] = []
        self.bandwidth_history: List[float] = []
        self.throughput_history: List[float] = []

    def update(self, node_count: int, avg_bandwidth: float = 100.0, throughput: float = 0.0) -> None:
        self.node_history.append(max(1, int(node_count)))
        self.bandwidth_history.append(float(avg_bandwidth))
        self.throughput_history.append(float(throughput))
        del self.node_history[:-self.history_size]
        del self.bandwidth_history[:-self.history_size]
        del self.throughput_history[:-self.history_size]

    def predict(self, horizon: Optional[int] = None) -> PredictionResult:
        horizon = int(horizon or self.forecast_horizon)
        if not self.node_history:
            return PredictionResult(time.time(), [1] * horizon, [0.4] * horizon)
        if len(self.node_history) == 1:
            current = self.node_history[-1]
            return PredictionResult(time.time(), [current] * horizon, [0.55] * horizon)

        diffs = [self.node_history[i] - self.node_history[i - 1] for i in range(1, len(self.node_history))]
        # 对应论文中的资源预测项：用节点数差分近似 ARIMA 的自回归部分。
        ar1 = diffs[-1]
        ar2 = diffs[-2] if len(diffs) > 1 else 0
        residual = ar1 - (0.65 * ar2)
        current = self.node_history[-1]
        predictions: List[int] = []
        confidence: List[float] = []
        for step in range(horizon):
            delta = 0.65 * ar1 + 0.2 * ar2 + 0.15 * residual
            bandwidth_correction = self._bandwidth_correction()
            current = max(1, current + int(round(delta + bandwidth_correction)))
            predictions.append(current)
            confidence.append(max(0.35, 0.88 * math.exp(-0.08 * step)))
            ar2, ar1 = ar1, delta
        return PredictionResult(time.time(), predictions, confidence)

    def _bandwidth_correction(self) -> float:
        if not self.bandwidth_history:
            return 0.0
        avg = sum(self.bandwidth_history[-5:]) / min(5, len(self.bandwidth_history))
        if avg > 90:
            return -0.25
        if avg < 45:
            return 0.1
        return 0.0


class LiveputCalculator:
    def __init__(self, scheduling_interval: float = 300.0):
        self.scheduling_interval = float(scheduling_interval)

    def throughput(self, config: ParallelConfig) -> float:
        pipeline_efficiency = config.num_microbatches / (
            config.num_microbatches + config.pipeline_depth - 1
        )
        tensor_penalty = 1.0 / (1 + 0.08 * max(config.tensor_parallel - 1, 0))
        return 2.0 * config.data_parallel * config.pipeline_depth * pipeline_efficiency * tensor_penalty

    def migration_cost(self, current: ParallelConfig, target: ParallelConfig) -> float:
        cost = 0.0
        if current.pipeline_depth != target.pipeline_depth:
            cost += 8.0 + 4.0 * abs(current.pipeline_depth - target.pipeline_depth)
        if current.data_parallel != target.data_parallel:
            cost += 5.0
        if current.tensor_parallel != target.tensor_parallel:
            cost += 6.0
        if cost:
            cost += 10.0
        return cost

    def compute_liveput(
        self,
        config: ParallelConfig,
        predicted_nodes: int,
        migration_cost: float = 0.0,
    ) -> float:
        # Liveput = 吞吐收益 × 可用训练时间 × 容错余量，迁移成本从窗口时间中扣除。
        if not config.is_feasible(predicted_nodes):
            return float("-inf")
        spare_nodes = max(predicted_nodes - config.required_nodes, 0)
        resilience = min(1.0, 0.86 + 0.03 * spare_nodes)
        effective_time = max(0.0, self.scheduling_interval - migration_cost)
        return self.throughput(config) * effective_time * resilience


class DPConfigSearcher:
    def __init__(
        self,
        liveput_calculator: LiveputCalculator,
        max_data_parallel: int = 8,
        max_pipeline_depth: int = 16,
        tensor_parallel_choices: Iterable[int] = (1,),
    ):
        self.liveput_calculator = liveput_calculator
        self.max_data_parallel = max_data_parallel
        self.max_pipeline_depth = max_pipeline_depth
        self.tensor_parallel_choices = tuple(tensor_parallel_choices)

    def candidate_configs(self, nodes: int, num_microbatches: int = 8) -> List[ParallelConfig]:
        # 论文 DP 搜索的候选配置集合：只保留节点数能够承载的并行组合。
        candidates: List[ParallelConfig] = []
        for tensor in self.tensor_parallel_choices:
            for data in range(1, min(self.max_data_parallel, nodes) + 1):
                for pipe in range(1, min(self.max_pipeline_depth, nodes) + 1):
                    config = ParallelConfig(data, pipe, tensor, num_microbatches)
                    if config.is_feasible(nodes):
                        candidates.append(config)
        return candidates

    def search(
        self,
        current_config: ParallelConfig,
        predicted_nodes: List[int],
    ) -> Tuple[ParallelConfig, float, List[ParallelConfig]]:
        if not predicted_nodes:
            return current_config, 0.0, [current_config]

        # dp[t][config] 记录预测窗口第 t 步到达 config 时的最大累计 Liveput 和前驱配置。
        dp: List[Dict[ParallelConfig, Tuple[float, Optional[ParallelConfig]]]] = []
        first_nodes = predicted_nodes[0]
        first: Dict[ParallelConfig, Tuple[float, Optional[ParallelConfig]]] = {}
        for config in self.candidate_configs(first_nodes, current_config.num_microbatches):
            cost = self.liveput_calculator.migration_cost(current_config, config)
            first[config] = (
                self.liveput_calculator.compute_liveput(config, first_nodes, cost),
                None,
            )
        dp.append(first)

        for idx, nodes in enumerate(predicted_nodes[1:], start=1):
            table: Dict[ParallelConfig, Tuple[float, Optional[ParallelConfig]]] = {}
            for config in self.candidate_configs(nodes, current_config.num_microbatches):
                best_value = float("-inf")
                best_prev: Optional[ParallelConfig] = None
                for prev_config, (prev_value, _) in dp[idx - 1].items():
                    cost = self.liveput_calculator.migration_cost(prev_config, config)
                    value = prev_value + self.liveput_calculator.compute_liveput(config, nodes, cost)
                    if value > best_value:
                        best_value = value
                        best_prev = prev_config
                table[config] = (best_value, best_prev)
            dp.append(table)

        final_config, (best_value, _) = max(dp[-1].items(), key=lambda item: item[1][0])
        path = [final_config]
        # 回溯得到论文算法中的最优配置路径，path[0] 是当前窗口应切换的目标配置。
        for idx in range(len(dp) - 1, 0, -1):
            prev = dp[idx][path[-1]][1]
            if prev is None:
                break
            path.append(prev)
        path.reverse()
        return path[0], best_value, path


class ProactiveResourceAdjuster:
    def __init__(self, failure_threshold: float = 0.4):
        self.failure_threshold = failure_threshold
        self.actions: List[str] = []

    def high_risk_nodes(self, node_healths: Dict[int, NodeHealth]) -> List[int]:
        return [
            node_id for node_id, health in node_healths.items()
            if health.failure_probability >= self.failure_threshold
        ]

    def plan_actions(self, current: ParallelConfig, target: ParallelConfig, risks: List[int]) -> List[str]:
        actions: List[str] = []
        if current.pipeline_depth != target.pipeline_depth:
            actions.append(f"切换流水线深度 {current.pipeline_depth}->{target.pipeline_depth}")
        if current.data_parallel != target.data_parallel:
            actions.append(f"调整数据并行 {current.data_parallel}->{target.data_parallel}")
        if risks:
            actions.append(f"迁移高风险节点上的专家副本: {risks}")
        if not actions:
            actions.append("保持当前配置")
        self.actions.extend(actions)
        return actions


class PredictiveResourceScheduler:
    def __init__(
        self,
        initial_nodes: int = 4,
        initial_config: Optional[ParallelConfig] = None,
        scheduling_interval: float = 300.0,
        forecast_horizon: int = 6,
    ):
        self.current_nodes = int(initial_nodes)
        self.current_config = initial_config or ParallelConfig(1, max(1, min(initial_nodes, 4)))
        self.predictor = ARIMANodePredictor(forecast_horizon=forecast_horizon)
        self.liveput_calculator = LiveputCalculator(scheduling_interval)
        self.config_searcher = DPConfigSearcher(self.liveput_calculator)
        self.adjuster = ProactiveResourceAdjuster()
        self.node_healths: Dict[int, NodeHealth] = {}
        self.decisions: List[SchedulingDecision] = []
        self.last_prediction: Optional[PredictionResult] = None

    def update_state(
        self,
        node_count: Optional[int] = None,
        node_healths: Optional[Dict[int, NodeHealth]] = None,
        throughput: float = 0.0,
    ) -> None:
        if node_count is not None:
            self.current_nodes = max(1, int(node_count))
        if node_healths:
            self.node_healths.update(node_healths)
        avg_bandwidth = (
            sum(item.bandwidth for item in self.node_healths.values()) / len(self.node_healths)
            if self.node_healths else 100.0
        )
        self.predictor.update(self.current_nodes, avg_bandwidth, throughput)

    def step(self) -> SchedulingDecision:
        self.last_prediction = self.predictor.predict()
        target, liveput, path = self.config_searcher.search(
            self.current_config,
            self.last_prediction.predicted_nodes,
        )
        cost = self.liveput_calculator.migration_cost(self.current_config, target)
        risks = self.adjuster.high_risk_nodes(self.node_healths)
        actions = self.adjuster.plan_actions(self.current_config, target, risks)
        decision = SchedulingDecision(
            timestamp=time.time(),
            current_config=self.current_config,
            target_config=target,
            expected_liveput=liveput,
            migration_cost=cost,
            config_path=path,
            actions=actions,
        )
        self.current_config = target
        self.decisions.append(decision)
        return decision

    def get_statistics(self) -> dict:
        return {
            "current_nodes": self.current_nodes,
            "current_config": self.current_config.to_dict(),
            "decision_count": len(self.decisions),
            "last_prediction": self.last_prediction.to_dict() if self.last_prediction else None,
            "last_decision": self.decisions[-1].to_dict() if self.decisions else None,
        }
