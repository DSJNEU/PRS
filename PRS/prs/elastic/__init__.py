from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock
from typing import Callable, Dict, Iterable, List, Optional


class NodeStatus(Enum):
    UP = "up"
    SUSPECT = "suspect"
    DOWN = "down"
    RECOVERING = "recovering"


@dataclass
class NodeInfo:
    node_id: int
    ip: str
    port: int = 29500
    devices: str = "0"
    status: NodeStatus = NodeStatus.UP
    last_heartbeat: float = field(default_factory=time.time)
    gpu_utilization: float = 0.0
    memory_usage: float = 0.0
    failure_count: int = 0

    @property
    def gpu_count(self) -> int:
        return len([item for item in self.devices.split(",") if item.strip()])

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "ip": self.ip,
            "port": self.port,
            "devices": self.devices,
            "gpu_count": self.gpu_count,
            "status": self.status.value,
            "last_heartbeat": self.last_heartbeat,
            "gpu_utilization": self.gpu_utilization,
            "memory_usage": self.memory_usage,
            "failure_count": self.failure_count,
        }


@dataclass
class ReconfigurationEvent:
    timestamp: float
    failed_nodes: List[int]
    remaining_nodes: List[int]
    selected_template_id: Optional[str] = None
    reason: str = "node_failure"

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "failed_nodes": self.failed_nodes,
            "remaining_nodes": self.remaining_nodes,
            "selected_template_id": self.selected_template_id,
            "reason": self.reason,
        }


class HeartbeatMonitor:
    """带自适应超时阈值的心跳监控器。"""

    def __init__(
        self,
        min_timeout: float = 2.0,
        max_timeout: float = 30.0,
        probe: Optional[Callable[[NodeInfo], bool]] = None,
    ):
        self.min_timeout = float(min_timeout)
        self.max_timeout = float(max_timeout)
        self.probe = probe
        self._nodes: Dict[int, NodeInfo] = {}
        self._latencies: Dict[int, List[float]] = {}
        self._lock = Lock()

    def register_node(self, node: NodeInfo) -> None:
        with self._lock:
            node.last_heartbeat = time.time()
            self._nodes[node.node_id] = node
            self._latencies[node.node_id] = []

    def update_heartbeat(
        self,
        node_id: int,
        latency: float = 0.0,
        gpu_utilization: Optional[float] = None,
        memory_usage: Optional[float] = None,
    ) -> None:
        with self._lock:
            node = self._nodes[node_id]
            node.last_heartbeat = time.time()
            node.status = NodeStatus.UP
            if gpu_utilization is not None:
                node.gpu_utilization = gpu_utilization
            if memory_usage is not None:
                node.memory_usage = memory_usage
            values = self._latencies.setdefault(node_id, [])
            values.append(max(float(latency), 0.0))
            del values[:-20]

    def mark_failed(self, node_id: int) -> None:
        with self._lock:
            node = self._nodes[node_id]
            node.status = NodeStatus.DOWN
            node.failure_count += 1

    def check(self, now: Optional[float] = None) -> List[int]:
        # 论文中的“心跳超时 + 主动确认”：先标记 suspect，再通过 probe 确认故障。
        now = now or time.time()
        failed: List[int] = []
        with self._lock:
            for node in self._nodes.values():
                if node.status == NodeStatus.DOWN:
                    failed.append(node.node_id)
                    continue
                elapsed = now - node.last_heartbeat
                if elapsed <= self.adaptive_timeout(node.node_id):
                    continue
                node.status = NodeStatus.SUSPECT
                if self._confirm_failure(node):
                    node.status = NodeStatus.DOWN
                    node.failure_count += 1
                    failed.append(node.node_id)
        return sorted(set(failed))

    def adaptive_timeout(self, node_id: int) -> float:
        # 用近期心跳延迟的 mean + 3σ 做自适应阈值，减少网络抖动导致的误判。
        samples = self._latencies.get(node_id, [])
        if len(samples) < 2:
            return self.min_timeout
        mean = statistics.fmean(samples)
        std = statistics.pstdev(samples) if len(samples) > 1 else 0.0
        return min(max(mean + 3 * std, self.min_timeout), self.max_timeout)

    def nodes(self) -> Dict[int, NodeInfo]:
        with self._lock:
            return dict(self._nodes)

    def _confirm_failure(self, node: NodeInfo) -> bool:
        if self.probe is None:
            return True
        try:
            return not self.probe(node)
        except Exception:
            return True


class FailureDetector:
    def __init__(self, heartbeat_monitor: HeartbeatMonitor):
        self.heartbeat_monitor = heartbeat_monitor

    def detect(self) -> Optional[ReconfigurationEvent]:
        failed = self.heartbeat_monitor.check()
        if not failed:
            return None
        # ReconfigurationEvent 是故障检测和 PRSEngine 模板切换之间的衔接对象。
        nodes = self.heartbeat_monitor.nodes()
        remaining = [
            node_id for node_id, node in nodes.items()
            if node_id not in failed and node.status == NodeStatus.UP
        ]
        return ReconfigurationEvent(time.time(), failed, sorted(remaining))


class ElasticController:
    """协调故障隔离、模板切换和恢复流程。"""

    def __init__(self, min_nodes: int = 1, heartbeat_monitor: Optional[HeartbeatMonitor] = None):
        self.min_nodes = int(min_nodes)
        self.heartbeat_monitor = heartbeat_monitor or HeartbeatMonitor()
        self.failure_detector = FailureDetector(self.heartbeat_monitor)
        self.events: List[ReconfigurationEvent] = []
        self._callbacks: List[Callable[[ReconfigurationEvent], None]] = []
        self._reconfiguring = False

    def initialize(self, nodes: Iterable[NodeInfo]) -> None:
        for node in nodes:
            self.heartbeat_monitor.register_node(node)

    def register_node(self, node: NodeInfo) -> None:
        self.heartbeat_monitor.register_node(node)

    def send_heartbeat(self, node_id: int, **metrics: float) -> None:
        self.heartbeat_monitor.update_heartbeat(node_id, **metrics)

    def mark_failed(self, node_id: int) -> ReconfigurationEvent:
        self.heartbeat_monitor.mark_failed(node_id)
        event = self.failure_detector.detect()
        if event is None:
            event = ReconfigurationEvent(time.time(), [node_id], self.get_current_nodes())
        self._record_event(event)
        return event

    def check_failure(self) -> Optional[ReconfigurationEvent]:
        event = self.failure_detector.detect()
        if event:
            self._record_event(event)
        return event

    def add_reconfiguration_callback(self, callback: Callable[[ReconfigurationEvent], None]) -> None:
        self._callbacks.append(callback)

    def start_reconfiguration(self) -> None:
        self._reconfiguring = True

    def finish_reconfiguration(self) -> None:
        self._reconfiguring = False

    def need_reconfiguration(self) -> bool:
        return self._reconfiguring

    def get_current_nodes(self) -> List[int]:
        return [
            node_id for node_id, node in self.heartbeat_monitor.nodes().items()
            if node.status == NodeStatus.UP
        ]

    def get_num_nodes(self) -> int:
        return len(self.get_current_nodes())

    def can_continue_training(self) -> bool:
        return self.get_num_nodes() >= self.min_nodes

    def shutdown(self) -> None:
        self._callbacks.clear()

    def _record_event(self, event: ReconfigurationEvent) -> None:
        # 控制层只记录事件并通知回调，具体模板选择和状态恢复由 PRSEngine 完成。
        self.events.append(event)
        self._reconfiguring = True
        for callback in self._callbacks:
            callback(event)


class DistributedEnvironmentManager:
    """本地演示版通信环境管理器。"""

    def __init__(self) -> None:
        self.rank = 0
        self.world_size = 1
        self.initialized = False

    def initialize(self, rank: int = 0, world_size: int = 1, **_: object) -> None:
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.initialized = True

    def destroy(self) -> None:
        self.initialized = False

    def is_initialized(self) -> bool:
        return self.initialized

    def get_rank(self) -> int:
        return self.rank

    def get_world_size(self) -> int:
        return self.world_size

    def is_master(self) -> bool:
        return self.rank == 0
