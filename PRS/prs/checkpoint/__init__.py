from __future__ import annotations

import hashlib
import json
import pickle
import shutil
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class CheckpointMetadata:
    step: int
    timestamp: float
    size_bytes: int
    is_incremental: bool
    checksum: str
    base_step: Optional[int] = None
    hot_keys: List[str] = field(default_factory=list)
    cold_keys: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "timestamp": self.timestamp,
            "size_bytes": self.size_bytes,
            "is_incremental": self.is_incremental,
            "base_step": self.base_step,
            "checksum": self.checksum,
            "hot_keys": self.hot_keys,
            "cold_keys": self.cold_keys,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CheckpointMetadata":
        return cls(
            step=int(data["step"]),
            timestamp=float(data["timestamp"]),
            size_bytes=int(data["size_bytes"]),
            is_incremental=bool(data["is_incremental"]),
            base_step=data.get("base_step"),
            checksum=str(data.get("checksum", "")),
            hot_keys=list(data.get("hot_keys", [])),
            cold_keys=list(data.get("cold_keys", [])),
        )


class IncrementalCheckpoint:
    """递归比较训练状态，只保存变化项。"""

    def diff(self, old_state: Dict[str, Any], new_state: Dict[str, Any]) -> Dict[str, Any]:
        # 对嵌套 dict 递归求差异，对应论文里的增量 checkpoint delta。
        result: Dict[str, Any] = {}
        for key, new_value in new_state.items():
            if key not in old_state:
                result[key] = new_value
                continue
            old_value = old_state[key]
            if isinstance(old_value, dict) and isinstance(new_value, dict):
                nested = self.diff(old_value, new_value)
                if nested:
                    result[key] = nested
            elif not self._equal(old_value, new_value):
                result[key] = new_value
        return result

    def merge(self, base_state: Dict[str, Any], diff_state: Dict[str, Any]) -> Dict[str, Any]:
        # 恢复时按 base + delta 重建完整状态，不能直接覆盖整棵状态树。
        merged = self._copy_state(base_state)
        for key, value in diff_state.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = self.merge(merged[key], value)
            else:
                merged[key] = self._copy_value(value)
        return merged

    def checksum(self, state: Dict[str, Any]) -> str:
        payload = pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
        return hashlib.sha256(payload).hexdigest()[:16]

    def _equal(self, left: Any, right: Any) -> bool:
        try:
            import torch

            if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
                return bool(torch.equal(left, right))
        except Exception:
            pass
        try:
            return left == right
        except Exception:
            return pickle.dumps(left, protocol=pickle.HIGHEST_PROTOCOL) == pickle.dumps(
                right, protocol=pickle.HIGHEST_PROTOCOL
            )

    def _copy_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return {key: self._copy_value(value) for key, value in state.items()}

    def _copy_value(self, value: Any) -> Any:
        clone = getattr(value, "clone", None)
        if callable(clone):
            return clone()
        return pickle.loads(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))


class HierarchicalCheckpointManager:
    def __init__(self, hot_interval: int = 10, cold_interval: int = 100):
        self.hot_interval = hot_interval
        self.cold_interval = cold_interval
        self.hot_patterns = ("optimizer", "grad", "lr", "scheduler", "step", "loss")
        self.cold_patterns = ("model", "weight", "param", "template", "config")

    def classify_state(self, state: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
        # 论文中的热/冷状态分层：优化器、step 等高频变化归热状态，模型和配置归冷状态。
        hot: Dict[str, Any] = {}
        cold: Dict[str, Any] = {}
        for key, value in state.items():
            lowered = key.lower()
            if any(pattern in lowered for pattern in self.hot_patterns):
                hot[key] = value
            elif any(pattern in lowered for pattern in self.cold_patterns):
                cold[key] = value
            else:
                hot[key] = value
        return hot, cold

    def should_save_hot(self, step: int) -> bool:
        return step > 0 and step % self.hot_interval == 0

    def should_save_cold(self, step: int) -> bool:
        return step == 0 or (step > 0 and step % self.cold_interval == 0)


class CheckpointManager:
    """支持完整、增量和异步保存的检查点管理器。"""

    def __init__(
        self,
        checkpoint_dir: Path | str,
        interval: int = 10,
        max_to_keep: int = 5,
        enable_incremental: bool = True,
        enable_async: bool = False,
        full_interval: int = 50,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.interval = int(interval)
        self.max_to_keep = int(max_to_keep)
        self.enable_incremental = bool(enable_incremental)
        self.enable_async = bool(enable_async)
        self.full_interval = max(int(full_interval), self.interval)
        self.incremental = IncrementalCheckpoint()
        self.hierarchy = HierarchicalCheckpointManager(interval, self.full_interval)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints: List[CheckpointMetadata] = []
        self._last_full_state: Optional[Dict[str, Any]] = None
        self._last_state: Optional[Dict[str, Any]] = None
        self._executor: Optional[ThreadPoolExecutor] = None
        self._pending: List[Future] = []
        self._load_metadata()

    def should_checkpoint(self, step: int) -> bool:
        return step > 0 and step % self.interval == 0

    def save_state(self, step: int, state: Dict[str, Any]) -> CheckpointMetadata:
        # 周期性落完整状态，其余步数只保存相对上一状态的差异。
        full_required = (
            not self.enable_incremental
            or self._last_full_state is None
            or step % self.full_interval == 0
        )
        if full_required:
            payload = self.incremental._copy_state(state)
            is_incremental = False
            base_step = None
            self._last_full_state = self.incremental._copy_state(state)
        else:
            payload = self.incremental.diff(self._last_state or {}, state)
            is_incremental = bool(payload)
            base_step = self._latest_full_step()
            if not is_incremental:
                payload = {"step": step}

        hot, cold = self.hierarchy.classify_state(state)
        path = self._checkpoint_path(step)
        checksum = self.incremental.checksum(payload)
        metadata = CheckpointMetadata(
            step=step,
            timestamp=time.time(),
            size_bytes=0,
            is_incremental=is_incremental,
            base_step=base_step,
            checksum=checksum,
            hot_keys=sorted(hot),
            cold_keys=sorted(cold),
        )

        if self.enable_async:
            self._submit_save(path, payload, metadata)
        else:
            self._write_checkpoint(path, payload, metadata)

        self._last_state = self.incremental._copy_state(state)
        self.checkpoints.append(metadata)
        self._cleanup()
        self._save_metadata()
        return metadata

    def save(
        self,
        step: int,
        model: Any = None,
        optimizer: Any = None,
        extra_state: Optional[Dict[str, Any]] = None,
    ) -> CheckpointMetadata:
        state: Dict[str, Any] = {"step": step}
        if model is not None:
            state["model"] = _move_tensors_to_cpu(
                model.state_dict() if hasattr(model, "state_dict") else model
            )
        if optimizer is not None:
            state["optimizer"] = _move_tensors_to_cpu(
                optimizer.state_dict() if hasattr(optimizer, "state_dict") else optimizer
            )
        if extra_state:
            state.update(extra_state)
        return self.save_state(step, state)

    def load(self, step: Optional[int] = None, model: Any = None, optimizer: Any = None) -> Optional[Dict[str, Any]]:
        self.wait()
        metadata = self._metadata_for_step(step)
        if metadata is None:
            return None
        state = self._reconstruct(metadata.step)
        if state is None:
            return None
        if model is not None and "model" in state and hasattr(model, "load_state_dict"):
            model.load_state_dict(state["model"])
        if optimizer is not None and "optimizer" in state and hasattr(optimizer, "load_state_dict"):
            optimizer.load_state_dict(state["optimizer"])
        return state

    def list_checkpoints(self) -> List[CheckpointMetadata]:
        return list(self.checkpoints)

    def get_latest_step(self) -> Optional[int]:
        return self.checkpoints[-1].step if self.checkpoints else None

    def delete_checkpoint(self, step: int) -> bool:
        path = self._checkpoint_path(step)
        if not path.exists():
            return False
        shutil.rmtree(path)
        self.checkpoints = [item for item in self.checkpoints if item.step != step]
        self._save_metadata()
        return True

    def wait(self) -> None:
        while self._pending:
            future = self._pending.pop(0)
            future.result()

    def close(self) -> None:
        self.wait()
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None

    def _submit_save(self, path: Path, payload: Dict[str, Any], metadata: CheckpointMetadata) -> None:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1)
        self._pending.append(self._executor.submit(self._write_checkpoint, path, payload, metadata))

    def _write_checkpoint(self, path: Path, payload: Dict[str, Any], metadata: CheckpointMetadata) -> None:
        path.mkdir(parents=True, exist_ok=True)
        data_path = path / "state.pkl"
        data_path.write_bytes(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
        metadata.size_bytes = data_path.stat().st_size
        (path / "metadata.json").write_text(
            json.dumps(metadata.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _reconstruct(self, step: int) -> Optional[Dict[str, Any]]:
        metadata = self._metadata_for_step(step)
        if metadata is None:
            return None
        payload = self._read_payload(metadata.step)
        if payload is None:
            return None
        if not metadata.is_incremental:
            return payload
        if metadata.base_step is None:
            return payload
        # 从最近完整检查点开始，顺序合并中间增量，保证恢复到指定 step。
        base_state = self._reconstruct(metadata.base_step)
        if base_state is None:
            return None
        for item in sorted(self.checkpoints, key=lambda item: item.step):
            if item.step <= metadata.base_step or item.step > metadata.step:
                continue
            diff = self._read_payload(item.step)
            if diff is not None:
                base_state = self.incremental.merge(base_state, diff)
        return base_state

    def _read_payload(self, step: int) -> Optional[Dict[str, Any]]:
        path = self._checkpoint_path(step) / "state.pkl"
        if not path.exists():
            return None
        return pickle.loads(path.read_bytes())

    def _metadata_for_step(self, step: Optional[int]) -> Optional[CheckpointMetadata]:
        if not self.checkpoints:
            return None
        if step is None:
            return self.checkpoints[-1]
        for item in self.checkpoints:
            if item.step == step:
                return item
        return None

    def _latest_full_step(self) -> Optional[int]:
        for item in reversed(self.checkpoints):
            if not item.is_incremental:
                return item.step
        return None

    def _checkpoint_path(self, step: int) -> Path:
        return self.checkpoint_dir / f"checkpoint_{step}"

    def _load_metadata(self) -> None:
        metadata_path = self.checkpoint_dir / "metadata.json"
        if not metadata_path.exists():
            return
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.checkpoints = [
            CheckpointMetadata.from_dict(item) for item in payload.get("checkpoints", [])
        ]

    def _save_metadata(self) -> None:
        payload = {"checkpoints": [item.to_dict() for item in self.checkpoints]}
        (self.checkpoint_dir / "metadata.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _cleanup(self) -> None:
        # 增量检查点依赖 base_step，对应的完整检查点不能被保留策略提前删除。
        protected = {item.base_step for item in self.checkpoints if item.base_step is not None}
        while len(self.checkpoints) > self.max_to_keep:
            old_index = next(
                (index for index, item in enumerate(self.checkpoints) if item.step not in protected),
                0,
            )
            old = self.checkpoints.pop(old_index)
            path = self._checkpoint_path(old.step)
            if path.exists():
                shutil.rmtree(path)


def _move_tensors_to_cpu(value: Any) -> Any:
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().clone()
    except Exception:
        pass
    if isinstance(value, dict):
        return {key: _move_tensors_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_tensors_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_tensors_to_cpu(item) for item in value)
    return value
