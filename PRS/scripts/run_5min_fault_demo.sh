#!/usr/bin/env bash
set -euo pipefail

cd /home/ubuntu/dsj/PRS

OUT_DIR=".prs_runtime/five_min_fault_demo_stable"
mkdir -p "${OUT_DIR}"

echo "========================================"
echo "PRS 5 分钟 8×V100 容错演示"
echo "项目目录：$(pwd)"
echo "输出目录：${OUT_DIR}"
echo "开始时间：$(date)"
echo "========================================"

timeout --signal=SIGINT --kill-after=20s 300s bash -lc '
env \
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  NCCL_DEBUG=WARN \
  NCCL_SOCKET_IFNAME=lo \
  TORCH_NCCL_BLOCKING_WAIT=1 \
  .venv-linux/bin/torchrun \
    --nproc_per_node=8 \
    --max_restarts=1 \
    --master_addr=127.0.0.1 \
    --master_port=29617 \
    -m prs.experiments.cli \
    --output-dir .prs_runtime/five_min_fault_demo_stable \
    --model-source synthetic \
    --steps 100000 \
    --batch-size 1 \
    --seq-len 256 \
    --hidden-size 512 \
    --layers 8 \
    --heads 8 \
    --checkpoint-interval 5 \
    --scheduler-interval-steps 25 \
    --fault-step 50 \
    --fault-rank 0 \
    --fault-mode exit
'

EXIT_CODE=$?

echo "========================================"
echo "结束时间：$(date)"
echo "退出码：${EXIT_CODE}"
echo "========================================"

if [ "${EXIT_CODE}" -eq 124 ]; then
  echo "结果：运行到 300 秒超时停止，符合 5 分钟演示预期。"
else
  echo "结果：进程在超时前结束，请查看日志确认训练状态。"
fi

echo "输出文件："
find "${OUT_DIR}" -maxdepth 3 -type f | sort || true
