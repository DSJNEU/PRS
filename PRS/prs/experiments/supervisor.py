from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field
from typing import List


@dataclass
class SupervisedRunResult:
    attempts: int
    return_codes: List[int] = field(default_factory=list)

    @property
    def succeeded(self) -> bool:
        return bool(self.return_codes) and self.return_codes[-1] == 0


def run_with_restarts(command: List[str], max_restarts: int = 1, restart_delay: float = 2.0) -> SupervisedRunResult:
    """本机实验监督器，进程故障后重新拉起同一 torchrun 命令。"""
    result = SupervisedRunResult(attempts=0)
    for attempt in range(max_restarts + 1):
        # 重启后仍使用同一命令和输出目录，由 checkpoint 逻辑完成状态恢复。
        result.attempts = attempt + 1
        process = subprocess.Popen(command)
        code = process.wait()
        result.return_codes.append(code)
        if code == 0:
            break
        if attempt < max_restarts:
            time.sleep(restart_delay)
    return result
