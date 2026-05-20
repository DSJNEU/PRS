from __future__ import annotations

import json
import os
import socket
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class MessageType(Enum):
    # 分布式控制面消息类型，对应论文中的心跳、故障通知、checkpoint 和重配置事件。
    HEARTBEAT = "heartbeat"
    FAILURE_NOTIFY = "failure_notify"
    CHECKPOINT = "checkpoint"
    RECONFIGURE = "reconfigure"
    BROADCAST = "broadcast"


@dataclass
class Message:
    msg_type: MessageType
    sender_id: int
    receiver_id: int
    timestamp: float
    data: Any = None
    seq_num: int = 0

    def serialize(self) -> bytes:
        return json.dumps(
            {
                "msg_type": self.msg_type.value,
                "sender_id": self.sender_id,
                "receiver_id": self.receiver_id,
                "timestamp": self.timestamp,
                "data": self.data,
                "seq_num": self.seq_num,
            },
            ensure_ascii=False,
        ).encode("utf-8")

    @classmethod
    def deserialize(cls, payload: bytes) -> "Message":
        data = json.loads(payload.decode("utf-8"))
        return cls(
            msg_type=MessageType(data["msg_type"]),
            sender_id=int(data["sender_id"]),
            receiver_id=int(data["receiver_id"]),
            timestamp=float(data["timestamp"]),
            data=data.get("data"),
            seq_num=int(data.get("seq_num", 0)),
        )


class TCPConnection:
    """演示用连接对象，保留接口但不启动真实后台网络服务。"""

    def __init__(self, host: str, port: int, is_server: bool = False):
        self.host = host
        self.port = int(port)
        self.is_server = is_server
        self.messages: List[Message] = []

    def send_message(self, message: Message, *_: object) -> None:
        self.messages.append(message)

    def receive_message(self, *_: object) -> Optional[Message]:
        return self.messages.pop(0) if self.messages else None

    def close(self) -> None:
        self.messages.clear()


class DistributedCoordinator:
    def __init__(
        self,
        is_master: bool = True,
        node_id: int = 0,
        master_addr: str = "127.0.0.1",
        master_port: int = 29500,
        num_nodes: int = 1,
        heartbeat_interval: float = 5.0,
        heartbeat_timeout: float = 30.0,
    ):
        self.is_master = is_master
        self.node_id = node_id
        self.master_addr = master_addr
        self.master_port = master_port
        self.num_nodes = num_nodes
        self.heartbeat_interval = heartbeat_interval
        self.heartbeat_timeout = heartbeat_timeout
        self.connected_nodes = {node_id}
        self.failure_callbacks: List[Callable[[List[int]], None]] = []
        self.running = False

    def start(self, *_: object, **__: object) -> None:
        self.running = True

    def add_failure_callback(self, callback: Callable[[List[int]], None]) -> None:
        self.failure_callbacks.append(callback)

    def notify_failure(self, failed_nodes: List[int]) -> None:
        # 控制面收到故障后先更新连接视图，再把故障节点列表交给上层恢复逻辑。
        for node_id in failed_nodes:
            self.connected_nodes.discard(node_id)
        for callback in self.failure_callbacks:
            callback(failed_nodes)

    def broadcast(self, data: Any) -> Message:
        return Message(MessageType.BROADCAST, self.node_id, -1, time.time(), data)

    def get_connected_nodes(self) -> List[int]:
        return sorted(self.connected_nodes)

    def get_num_connected_nodes(self) -> int:
        return len(self.connected_nodes)

    def is_connected(self) -> bool:
        return self.running

    def shutdown(self) -> None:
        self.running = False


@dataclass
class NodeConfig:
    node_id: int
    master_addr: str
    master_port: int
    num_nodes: int
    num_gpus: int = 1
    world_size: int = 1

    def to_env_vars(self) -> Dict[str, str]:
        return {
            "PRS_NODE_ID": str(self.node_id),
            "PRS_MASTER_ADDR": self.master_addr,
            "PRS_MASTER_PORT": str(self.master_port),
            "PRS_NUM_NODES": str(self.num_nodes),
            "PRS_NUM_GPUS": str(self.num_gpus),
            "PRS_WORLD_SIZE": str(self.world_size),
        }


def parse_node_config_from_env() -> NodeConfig:
    return NodeConfig(
        node_id=int(os.environ.get("PRS_NODE_ID", 0)),
        master_addr=os.environ.get("PRS_MASTER_ADDR", "127.0.0.1"),
        master_port=int(os.environ.get("PRS_MASTER_PORT", 29500)),
        num_nodes=int(os.environ.get("PRS_NUM_NODES", 1)),
        num_gpus=int(os.environ.get("PRS_NUM_GPUS", 1)),
        world_size=int(os.environ.get("PRS_WORLD_SIZE", 1)),
    )


def get_local_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"


def find_free_port(start_port: int = 29500, max_attempts: int = 100) -> int:
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("", port))
                return port
            except OSError:
                continue
    raise RuntimeError("找不到可用端口")
