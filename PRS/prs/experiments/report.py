from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from prs.experiments.metrics import load_metrics


def build_experiment_report(output_dir: Path | str) -> Dict[str, Any]:
    root = Path(output_dir)
    records = load_metrics(root.glob("metrics_rank*.jsonl"))
    summaries: List[Dict[str, Any]] = []
    for path in root.glob("summary_rank*.json"):
        summaries.append(json.loads(path.read_text(encoding="utf-8")))

    losses = [float(item["loss"]) for item in records if "loss" in item]
    throughputs = [float(item["throughput"]) for item in records if "throughput" in item]
    # 报告层只做实验数据归并，不重新计算训练或调度逻辑。
    report = {
        "output_dir": str(root),
        "record_count": len(records),
        "summary_count": len(summaries),
        "final_loss": losses[-1] if losses else None,
        "best_loss": min(losses) if losses else None,
        "avg_throughput": sum(throughputs) / len(throughputs) if throughputs else 0.0,
        "max_throughput": max(throughputs) if throughputs else 0.0,
        "summaries": summaries,
    }
    (root / "experiment_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (root / "experiment_report.md").write_text(_to_markdown(report), encoding="utf-8")
    return report


def _to_markdown(report: Dict[str, Any]) -> str:
    return "\n".join(
        [
            "# PRS 实验记录",
            "",
            f"- 输出目录：`{report['output_dir']}`",
            f"- 指标记录数：{report['record_count']}",
            f"- 最终损失：{report['final_loss']}",
            f"- 最优损失：{report['best_loss']}",
            f"- 平均吞吐：{report['avg_throughput']:.4f} samples/s",
            f"- 峰值吞吐：{report['max_throughput']:.4f} samples/s",
            "",
            "记录口径：以上指标来自本次训练写入的 metrics 和 summary 文件，适合继续整理吞吐、恢复时间和有效样本数。",
        ]
    )
