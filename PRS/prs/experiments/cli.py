from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from prs.experiments import (
    FaultInjectionConfig,
    TorchExperimentRunner,
    TorchTrainingConfig,
    build_experiment_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PRS 真实训练实验入口")
    parser.add_argument("--output-dir", default=".prs_runtime/torch_experiment")
    parser.add_argument("--model-source", choices=["synthetic", "hf"], default="synthetic")
    parser.add_argument("--hf-model-name", default="")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--checkpoint-interval", type=int, default=5)
    parser.add_argument("--scheduler-interval-steps", type=int, default=10)
    parser.add_argument("--disable-moe-adaptation", action="store_true")
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--fault-step", type=int, default=-1)
    parser.add_argument("--fault-rank", type=int, default=0)
    parser.add_argument("--fault-mode", choices=["exception", "exit"], default="exception")
    return parser


def config_from_args(args: argparse.Namespace) -> TorchTrainingConfig:
    # CLI 参数直接映射到真实训练配置，便于论文实验固定同一组运行口径。
    return TorchTrainingConfig(
        output_dir=args.output_dir,
        model_source=args.model_source,
        hf_model_name=args.hf_model_name,
        total_steps=args.steps,
        batch_size=args.batch_size,
        sequence_length=args.seq_len,
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.layers,
        num_heads=args.heads,
        learning_rate=args.lr,
        checkpoint_interval=args.checkpoint_interval,
        scheduler_interval_steps=args.scheduler_interval_steps,
        enable_moe_adaptation=not args.disable_moe_adaptation,
        num_experts=args.num_experts,
        fault=FaultInjectionConfig(
            enabled=args.fault_step > 0,
            step=args.fault_step,
            rank=args.fault_rank,
            mode=args.fault_mode,
        ),
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = config_from_args(args)
    rank = int(os.environ.get("RANK", "0"))
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    if rank == 0:
        # 多 rank 共享输出目录时只由 rank 0 写 config，避免并发覆盖。
        (Path(config.output_dir) / "config.json").write_text(
            json.dumps(config.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    summary = TorchExperimentRunner(config).run()
    if summary.get("rank", 0) == 0:
        build_experiment_report(config.output_dir)
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
