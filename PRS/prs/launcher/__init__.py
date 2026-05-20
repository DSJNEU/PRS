from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class NodeSpec:
    node_id: int
    hostname: str = "127.0.0.1"
    port: int = 29500
    num_gpus: int = 1
    devices: str = "0"
    user: str = ""

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "hostname": self.hostname,
            "port": self.port,
            "num_gpus": self.num_gpus,
            "devices": self.devices,
            "user": self.user,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "NodeSpec":
        return cls(
            node_id=int(data["node_id"]),
            hostname=str(data.get("hostname", "127.0.0.1")),
            port=int(data.get("port", 29500)),
            num_gpus=int(data.get("num_gpus", 1)),
            devices=str(data.get("devices", "0")),
            user=str(data.get("user", "")),
        )


@dataclass
class ClusterConfig:
    nodes: List[NodeSpec]
    master_port: int = 29500
    job_name: str = "prs_demo"
    script_path: str = "examples/run_demo.py"
    script_args: List[str] = field(default_factory=list)

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def world_size(self) -> int:
        return sum(node.num_gpus for node in self.nodes)

    @property
    def master_addr(self) -> str:
        return self.nodes[0].hostname if self.nodes else "127.0.0.1"

    def to_dict(self) -> dict:
        return {
            "nodes": [node.to_dict() for node in self.nodes],
            "master_port": self.master_port,
            "job_name": self.job_name,
            "script_path": self.script_path,
            "script_args": self.script_args,
        }

    def to_json(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    def torchrun_command(
        self,
        node_rank: int,
        module: str = "prs.experiments.cli",
        extra_args: Optional[List[str]] = None,
    ) -> List[str]:
        node = self.nodes[node_rank]
        # 多机实验入口：把集群配置转换成 torchrun 所需的 nnodes/node_rank/master 参数。
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nproc_per_node={node.num_gpus}",
            f"--nnodes={self.num_nodes}",
            f"--node_rank={node_rank}",
            f"--master_addr={self.master_addr}",
            f"--master_port={self.master_port}",
            "-m",
            module,
        ]
        command.extend(extra_args or self.script_args)
        return command

    @classmethod
    def from_json(cls, path: Path) -> "ClusterConfig":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            nodes=[NodeSpec.from_dict(item) for item in data.get("nodes", [])],
            master_port=int(data.get("master_port", 29500)),
            job_name=str(data.get("job_name", "prs_demo")),
            script_path=str(data.get("script_path", "examples/run_demo.py")),
            script_args=list(data.get("script_args", [])),
        )


class NodeLauncher:
    def __init__(self, node_spec: NodeSpec, cluster_config: ClusterConfig):
        self.node = node_spec
        self.cluster = cluster_config
        self.running = False
        self.process: Optional[subprocess.Popen] = None

    def launch(self) -> bool:
        command = self.cluster.torchrun_command(self.node.node_id)
        self.process = subprocess.Popen(command)
        self.running = True
        return True

    def stop(self) -> None:
        if self.process and self.process.poll() is None:
            self.process.terminate()
        self.running = False

    def is_running(self) -> bool:
        if self.process is not None:
            return self.process.poll() is None
        return self.running


class ClusterLauncher:
    def __init__(self, config: ClusterConfig):
        self.config = config
        self.launchers = [NodeLauncher(node, config) for node in config.nodes]

    def launch_all(self) -> bool:
        return all(launcher.launch() for launcher in self.launchers)

    def stop_all(self) -> None:
        for launcher in self.launchers:
            launcher.stop()

    def get_status(self) -> Dict[str, int | bool]:
        return {
            "num_nodes": self.config.num_nodes,
            "world_size": self.config.world_size,
            "running_nodes": sum(1 for launcher in self.launchers if launcher.is_running()),
            "is_running": any(launcher.is_running() for launcher in self.launchers),
        }
