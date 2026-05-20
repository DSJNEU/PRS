from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List


class MetricLogger:
    """训练实验指标落盘器。"""

    def __init__(self, output_dir: Path | str, rank: int = 0):
        self.output_dir = Path(output_dir)
        self.rank = int(rank)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.output_dir / f"metrics_rank{self.rank}.jsonl"
        self.records: List[Dict[str, Any]] = []
        self._file = self.path.open("a", encoding="utf-8")

    def log(self, **record: Any) -> None:
        record.setdefault("time", time.time())
        record.setdefault("rank", self.rank)
        self.records.append(record)
        self._file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._file.flush()

    def summarize(self, extra: Dict[str, Any] | None = None) -> Dict[str, Any]:
        # 这里汇总论文实验常用指标：最终损失、最优损失、平均吞吐和平均 step time。
        losses = [float(item["loss"]) for item in self.records if "loss" in item]
        throughputs = [float(item["throughput"]) for item in self.records if "throughput" in item]
        step_times = [float(item["step_time"]) for item in self.records if "step_time" in item]
        summary: Dict[str, Any] = {
            "rank": self.rank,
            "steps": len(losses),
            "final_loss": losses[-1] if losses else None,
            "best_loss": min(losses) if losses else None,
            "avg_throughput": statistics.fmean(throughputs) if throughputs else 0.0,
            "avg_step_time": statistics.fmean(step_times) if step_times else 0.0,
            "metrics_file": str(self.path),
        }
        if extra:
            summary.update(extra)
        summary_path = self.output_dir / f"summary_rank{self.rank}.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        if self.rank == 0:
            (self.output_dir / "summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        return summary

    def close(self) -> None:
        self._file.close()


def load_metrics(paths: Iterable[Path | str]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for path in paths:
        metric_path = Path(path)
        if not metric_path.exists():
            continue
        for line in metric_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    return records
