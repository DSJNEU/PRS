from __future__ import annotations

import sys
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prs import create_demo_engine


def main() -> None:
    checkpoint_dir = Path(".prs_runtime/example_checkpoints")
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    engine = create_demo_engine(checkpoint_dir, nodes=4)

    print("PRS 示例启动")
    print(f"当前模板: {engine.get_template_info()['template_id']}")

    for item in engine.run_steps(8):
        print(f"step={item['step']:02d} loss={item['loss']:.4f} throughput={item['throughput']:.3f}")

    event = engine.simulate_failure()
    print(f"故障恢复: failed={event.failed_nodes} template={event.selected_template_id}")

    decision = engine.scheduler_step()
    print(f"主动调度: target={decision['target_config']} actions={decision['actions']}")

    for item in engine.run_steps(4):
        print(f"step={item['step']:02d} loss={item['loss']:.4f} throughput={item['throughput']:.3f}")

    snapshot = engine.snapshot()
    print(f"检查点数量: {len(snapshot['checkpoints'])}")
    print("PRS 示例完成")
    engine.shutdown()


if __name__ == "__main__":
    main()
