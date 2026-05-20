# PRS

PRS 是面向分布式训练的弹性容错与资源调度实验系统。本项目把流水线模板、检查点、故障恢复、MoE 专家副本和预测调度放在同一套编排入口里，既可以在本机跑演示，也可以接入 PyTorch 和 `torchrun` 做多机多卡实验。

## 模块概览

| 模块              | 说明                                                    |
| ----------------- | ------------------------------------------------------- |
| `prs.planning`    | 根据层级性能画像，用动态规划切分流水线阶段              |
| `prs.checkpoint`  | 管理完整检查点、增量检查点、热/冷状态分类和异步保存     |
| `prs.elastic`     | 维护节点心跳，检测故障并记录重配置事件                  |
| `prs.moe`         | 统计专家负载，分配专家副本，给出 MRO 放置和通信统计     |
| `prs.scheduler`   | 根据节点历史和 Liveput 估计选择并行配置                 |
| `prs.engine`      | 统一编排训练状态、模板、检查点、故障恢复和调度          |
| `prs.web`         | 提供 Flask 监控台，查看训练、节点、模板和调度状态       |
| `prs.experiments` | 提供真实 PyTorch 训练、DDP 指标采集、故障注入和报告整理 |

## 安装

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

```bash
python -m pip install -e ".[torch]"
```

如果要加载 HuggingFace 模型，再安装实验依赖：

```bash
python -m pip install -e ".[experiments]"
```

```bash
python examples/run_demo.py
```

这个示例会完成一次 PRS 控制闭环：初始化模板、推进训练步、保存检查点、模拟节点故障、切换模板，并执行一次主动调度。

Web 监控台

```bash
python -m prs.web.monitor --host 127.0.0.1 --port 5000
```

浏览器打开：

```text
http://127.0.0.1:5000
```

页面可以查看训练状态、节点状态、当前流水线模板、检查点、MoE 专家分组和调度决策，也可以手动触发训练步、故障注入和主动调度。

单进程烟测：

```bash
python examples/run_real_training.py --steps 20 --batch-size 2 --seq-len 64
```

单机多卡 DDP：

```bash
torchrun --nproc_per_node=4 -m prs.experiments.cli \
  --model-source synthetic \
  --steps 200 \
  --batch-size 4 \
  --checkpoint-interval 20 \
  --scheduler-interval-steps 20
```

故障注入与恢复验证：

```bash
torchrun --nproc_per_node=4 -m prs.experiments.cli \
  --steps 100 \
  --fault-step 30 \
  --fault-rank 1 \
  --fault-mode exit
```

默认输出目录为 `.prs_runtime/torch_experiment/`，主要文件包括：

- `metrics_rank*.jsonl`：逐步记录 loss、吞吐、耗时和模板信息
- `summary.json`：rank 0 汇总结果
- `experiment_report.md`：可直接阅读的实验记录
- `experiment_report.json`：结构化实验结果
- `checkpoints/`：PRS 检查点目录

## 测试

```bash
python -m unittest discover -s tests
```

## 目录结构

```text
PRS/
├── configs/
│   └── default.json
├── docs/
│   ├── code_methodology.md
│   ├── usage.md
│   ├── v100_experiment.md
│   ├── server_8gpu_experiment.md
│   └── five_min_fault_tolerance_demo.md
├── examples/
│   ├── run_demo.py
│   └── run_real_training.py
├── prs/
│   ├── checkpoint/
│   ├── distributed/
│   ├── elastic/
│   ├── engine/
│   ├── experiments/
│   ├── launcher/
│   ├── moe/
│   ├── pipeline/
│   ├── planning/
│   ├── scheduler/
│   └── web/
├── scripts/
│   └── run_5min_fault_demo.sh
├── tests/
│   └── test_prs.py
├── pyproject.toml
└── requirements.txt
```
