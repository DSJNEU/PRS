from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


@dataclass
class ExpertLoad:
    expert_id: int
    layer_id: int
    token_count: int = 0
    token_ratio: float = 0.0
    compute_time: float = 0.0
    memory_usage: int = 0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "expert_id": self.expert_id,
            "layer_id": self.layer_id,
            "token_count": self.token_count,
            "token_ratio": self.token_ratio,
            "compute_time": self.compute_time,
            "memory_usage": self.memory_usage,
            "timestamp": self.timestamp,
        }


@dataclass
class ExpertReplica:
    expert_id: int
    replica_id: int
    node_id: int
    device_id: int = 0
    group_id: int = 0
    is_primary: bool = False
    status: str = "active"

    def to_dict(self) -> dict:
        return {
            "expert_id": self.expert_id,
            "replica_id": self.replica_id,
            "node_id": self.node_id,
            "device_id": self.device_id,
            "group_id": self.group_id,
            "is_primary": self.is_primary,
            "status": self.status,
        }


@dataclass
class ExpertGroup:
    group_id: int
    experts: List[int]
    nodes: Set[int]
    total_load: float = 0.0
    is_high_load: bool = False

    def to_dict(self) -> dict:
        return {
            "group_id": self.group_id,
            "experts": self.experts,
            "nodes": sorted(self.nodes),
            "total_load": self.total_load,
            "is_high_load": self.is_high_load,
        }


@dataclass
class TokenBatch:
    tokens: Any
    target_experts: List[int]
    target_nodes: List[int]
    local_count: int = 0
    remote_count: int = 0

    def to_dict(self) -> dict:
        return {
            "target_experts": self.target_experts,
            "target_nodes": self.target_nodes,
            "local_count": self.local_count,
            "remote_count": self.remote_count,
        }


class LoadHeatmapMonitor:
    """统计每层专家 token 分布，对应论文中的专家负载热力图。"""

    def __init__(self, num_experts: int, num_layers: int):
        self.num_experts = int(num_experts)
        self.num_layers = int(num_layers)
        self._counters: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))

    def record_token_routing(self, layer_id: int, expert_id: int, token_count: int = 1) -> None:
        self._counters[int(layer_id)][int(expert_id)] += int(token_count)

    def record_batch_routing(self, layer_id: int, routing_decisions: Dict[int, int]) -> None:
        for expert_id, count in routing_decisions.items():
            self.record_token_routing(layer_id, expert_id, count)

    def get_layer_heatmap(self, layer_id: int) -> Dict[int, ExpertLoad]:
        counts = self._counters.get(int(layer_id), {})
        total = sum(counts.values())
        heatmap: Dict[int, ExpertLoad] = {}
        for expert_id in range(self.num_experts):
            count = int(counts.get(expert_id, 0))
            heatmap[expert_id] = ExpertLoad(
                expert_id=expert_id,
                layer_id=int(layer_id),
                token_count=count,
                token_ratio=count / total if total else 0.0,
                compute_time=count * 0.08,
                memory_usage=count * 1024,
            )
        return heatmap

    def get_all_heatmaps(self) -> Dict[int, Dict[int, ExpertLoad]]:
        return {layer_id: self.get_layer_heatmap(layer_id) for layer_id in range(self.num_layers)}

    def reset(self) -> None:
        self._counters.clear()


class FaultAwareReplicaAllocator:
    def __init__(
        self,
        num_experts: int,
        num_nodes: int,
        node_capacity: int = 4,
        fault_threshold: int = 1,
    ):
        self.num_experts = int(num_experts)
        self.num_nodes = int(num_nodes)
        self.node_capacity = int(node_capacity)
        self.fault_threshold = int(fault_threshold)
        self.replica_allocation: Dict[int, Dict[int, int]] = {}

    @property
    def total_capacity(self) -> int:
        return max(self.num_nodes * self.node_capacity, self.num_experts)

    def compute_replica_allocation(
        self,
        layer_id: int,
        expert_loads: Dict[int, ExpertLoad],
    ) -> Dict[int, int]:
        # 按 token 占比分配副本，同时保留 fault_threshold 个基础副本用于容错。
        total_tokens = sum(load.token_count for load in expert_loads.values())
        remaining = self.total_capacity
        allocation: Dict[int, int] = {}
        sorted_loads = sorted(expert_loads.values(), key=lambda item: item.token_count, reverse=True)

        for index, load in enumerate(sorted_loads):
            left = len(sorted_loads) - index - 1
            if total_tokens <= 0:
                replicas = max(self.fault_threshold, self.total_capacity // self.num_experts)
            else:
                replicas = max(self.fault_threshold, math.ceil(load.token_count / total_tokens * self.total_capacity))
            replicas = min(replicas, max(self.fault_threshold, remaining - left * self.fault_threshold))
            allocation[load.expert_id] = replicas
            remaining -= replicas
        self.replica_allocation[int(layer_id)] = allocation
        return allocation

    def get_replica_allocation(self, layer_id: int) -> Dict[int, int]:
        return dict(self.replica_allocation.get(int(layer_id), {}))


class MROReplicaPlacer:
    def __init__(self, num_experts: int, num_nodes: int, node_capacity: int = 4):
        self.num_experts = int(num_experts)
        self.num_nodes = int(num_nodes)
        self.node_capacity = int(node_capacity)
        self.current_placement: Dict[int, List[ExpertReplica]] = {}
        self.groups: List[ExpertGroup] = []

    def place_replicas(
        self,
        expert_loads: Dict[int, ExpertLoad],
        allocation: Dict[int, int],
    ) -> List[ExpertReplica]:
        # MRO 放置：先把负载相近的专家分组，再在组内节点池中做均衡放置。
        self.groups = self._build_groups(expert_loads)
        node_counts: Dict[int, int] = defaultdict(int)
        replicas: List[ExpertReplica] = []
        replica_id = 0

        for group in self.groups:
            for expert_id in group.experts:
                for idx in range(allocation.get(expert_id, 1)):
                    node_id = self._select_node(group.nodes, node_counts)
                    replicas.append(
                        ExpertReplica(
                            expert_id=expert_id,
                            replica_id=replica_id,
                            node_id=node_id,
                            group_id=group.group_id,
                            is_primary=idx == 0,
                        )
                    )
                    node_counts[node_id] += 1
                    replica_id += 1

        self.current_placement = defaultdict(list)
        for replica in replicas:
            self.current_placement[replica.expert_id].append(replica)
        return replicas

    def get_replicas_on_node(self, node_id: int) -> List[ExpertReplica]:
        return [
            replica for replicas in self.current_placement.values()
            for replica in replicas
            if replica.node_id == int(node_id)
        ]

    def get_failover_nodes(self, failed_node: int) -> Dict[int, List[int]]:
        result: Dict[int, List[int]] = {}
        for expert_id, replicas in self.current_placement.items():
            if any(replica.node_id == failed_node for replica in replicas):
                result[expert_id] = sorted(
                    {replica.node_id for replica in replicas if replica.node_id != failed_node}
                )
        return result

    def _build_groups(self, expert_loads: Dict[int, ExpertLoad]) -> List[ExpertGroup]:
        ordered = sorted(expert_loads.values(), key=lambda item: item.token_ratio, reverse=True)
        group_size = max(1, self.node_capacity)
        groups: List[ExpertGroup] = []
        for group_id, start in enumerate(range(0, len(ordered), group_size)):
            chunk = ordered[start:start + group_size]
            nodes = {
                node_id for node_id in range(group_id, self.num_nodes, max(1, math.ceil(len(ordered) / group_size)))
            } or {group_id % max(1, self.num_nodes)}
            total_load = sum(item.token_ratio for item in chunk)
            groups.append(
                ExpertGroup(
                    group_id=group_id,
                    experts=[item.expert_id for item in chunk],
                    nodes=nodes,
                    total_load=total_load,
                    is_high_load=total_load > 0.5,
                )
            )
        return groups

    def _select_node(self, nodes: Iterable[int], node_counts: Dict[int, int]) -> int:
        candidates = list(nodes) or list(range(max(1, self.num_nodes)))
        return min(candidates, key=lambda node_id: node_counts[node_id])


class ZeroPaddingCommunicator:
    def __init__(self, num_experts: int, num_nodes: int):
        self.num_experts = int(num_experts)
        self.num_nodes = int(num_nodes)
        self.stats = {"total_tokens": 0, "local_tokens": 0, "remote_tokens": 0, "padding_saved": 0}

    def schedule_tokens(
        self,
        tokens: Any,
        routing_decisions: Dict[int, int],
        local_experts: Set[int],
        node_capacities: Optional[Dict[int, Dict[int, int]]] = None,
    ) -> TokenBatch:
        # Zero-Padding 通信统计：只统计需要跨节点发送的 token，并估算少填充的数量。
        target_experts: List[int] = []
        target_nodes: List[int] = []
        local_count = 0
        remote_count = 0

        for expert_id, count in routing_decisions.items():
            target_experts.extend([expert_id] * count)
            if expert_id in local_experts:
                local_count += count
                target_nodes.extend([0] * count)
            else:
                remote_count += count
                node_id = self._node_for_expert(expert_id, node_capacities)
                target_nodes.extend([node_id] * count)

        self.stats["total_tokens"] += sum(routing_decisions.values())
        self.stats["local_tokens"] += local_count
        self.stats["remote_tokens"] += remote_count
        self.stats["padding_saved"] += self._padding_saved(target_nodes)
        return TokenBatch(tokens, target_experts, target_nodes, local_count, remote_count)

    def get_communication_stats(self) -> dict:
        stats = dict(self.stats)
        total = stats["total_tokens"] or 1
        stats["local_ratio"] = stats["local_tokens"] / total
        stats["remote_ratio"] = stats["remote_tokens"] / total
        return stats

    def _node_for_expert(
        self,
        expert_id: int,
        node_capacities: Optional[Dict[int, Dict[int, int]]],
    ) -> int:
        if not node_capacities:
            return expert_id % max(1, self.num_nodes)
        candidates = [
            (node_id, capacity.get(expert_id, 0))
            for node_id, capacity in node_capacities.items()
            if capacity.get(expert_id, 0) > 0
        ]
        return max(candidates, key=lambda item: item[1])[0] if candidates else 0

    def _padding_saved(self, target_nodes: List[int]) -> int:
        if not target_nodes:
            return 0
        counts: Dict[int, int] = defaultdict(int)
        for node_id in target_nodes:
            counts[node_id] += 1
        return max(counts.values()) * len(counts) - sum(counts.values())


class MoEExpertParallelManager:
    def __init__(
        self,
        num_experts: int,
        num_layers: int,
        num_nodes: int,
        node_capacity: int = 4,
        fault_threshold: int = 1,
    ):
        self.num_experts = int(num_experts)
        self.num_layers = int(num_layers)
        self.num_nodes = int(num_nodes)
        self.load_monitor = LoadHeatmapMonitor(num_experts, num_layers)
        self.replica_allocator = FaultAwareReplicaAllocator(
            num_experts, num_nodes, node_capacity, fault_threshold
        )
        self.replica_placer = MROReplicaPlacer(num_experts, num_nodes, node_capacity)
        self.communicator = ZeroPaddingCommunicator(num_experts, num_nodes)
        self.latest_replicas: Dict[int, List[ExpertReplica]] = {}

    def record_routing(self, layer_id: int, expert_id: int, token_count: int = 1) -> None:
        self.load_monitor.record_token_routing(layer_id, expert_id, token_count)

    def record_batch_routing(self, layer_id: int, routing_decisions: Dict[int, int]) -> None:
        self.load_monitor.record_batch_routing(layer_id, routing_decisions)

    def allocate_replicas(self, layer_id: Optional[int] = None) -> Dict[int, Dict[int, int]]:
        # 先由热力图得到每层专家负载，再调用容错感知副本分配。
        layers = [layer_id] if layer_id is not None else list(range(self.num_layers))
        result: Dict[int, Dict[int, int]] = {}
        for current_layer in layers:
            heatmap = self.load_monitor.get_layer_heatmap(int(current_layer))
            result[int(current_layer)] = self.replica_allocator.compute_replica_allocation(
                int(current_layer),
                heatmap,
            )
        return result

    def place_replicas(self, layer_id: Optional[int] = None) -> Dict[int, List[ExpertReplica]]:
        layers = [layer_id] if layer_id is not None else list(range(self.num_layers))
        placements: Dict[int, List[ExpertReplica]] = {}
        for current_layer in layers:
            layer_id_int = int(current_layer)
            heatmap = self.load_monitor.get_layer_heatmap(layer_id_int)
            allocation = self.replica_allocator.get_replica_allocation(layer_id_int)
            if not allocation:
                allocation = self.replica_allocator.compute_replica_allocation(layer_id_int, heatmap)
            placements[layer_id_int] = self.replica_placer.place_replicas(heatmap, allocation)
        self.latest_replicas = placements
        return placements

    def schedule_tokens(
        self,
        tokens: Any,
        routing_decisions: Dict[int, int],
        local_experts: Set[int],
        node_id: int = 0,
    ) -> TokenBatch:
        capacities: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        for replicas in self.latest_replicas.values():
            for replica in replicas:
                capacities[replica.node_id][replica.expert_id] += 256
        return self.communicator.schedule_tokens(tokens, routing_decisions, local_experts, capacities)

    def handle_node_failure(self, failed_node: int) -> Dict[int, List[int]]:
        return self.replica_placer.get_failover_nodes(failed_node)

    def get_load_heatmap(self, layer_id: Optional[int] = None) -> dict:
        if layer_id is not None:
            return {
                expert_id: load.to_dict()
                for expert_id, load in self.load_monitor.get_layer_heatmap(layer_id).items()
            }
        return {
            current_layer: {
                expert_id: load.to_dict()
                for expert_id, load in self.load_monitor.get_layer_heatmap(current_layer).items()
            }
            for current_layer in range(self.num_layers)
        }

    def get_statistics(self) -> dict:
        total_replicas = sum(len(items) for items in self.latest_replicas.values())
        return {
            "num_experts": self.num_experts,
            "num_layers": self.num_layers,
            "num_nodes": self.num_nodes,
            "total_replicas": total_replicas,
            "communication": self.communicator.get_communication_stats(),
            "groups": [group.to_dict() for group in self.replica_placer.groups],
        }
