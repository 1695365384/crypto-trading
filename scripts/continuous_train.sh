#!/bin/bash
# 持续训练脚本 - 自动从最新模型继续训练
# 用法: ./scripts/continuous_train.sh [训练轮数]

ROUNDS=${1:-10}  # 默认训练10轮

echo "=========================================="
echo "Continuous Training - $ROUNDS rounds"
echo "=========================================="

for i in $(seq 1 $ROUNDS); do
    echo ""
    echo "========== Round $i / $ROUNDS =========="
    python scripts/train.py

    if [ $? -ne 0 ]; then
        echo "Error in round $i, stopping..."
        exit 1
    fi
    echo "Round $i completed!"
done

echo ""
echo "=========================================="
echo "All $ROUNDS rounds completed!"
echo "=========================================="
