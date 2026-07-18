#!/bin/bash
# BES Sweep Script
# 运行 7 个维度 × 3 个数据集 = 21 组实验
#
# 用法:
#   chmod +x run_bes_sweep.sh
#   ./run_bes_sweep.sh                # 运行所有实验
#   ./run_bes_sweep.sh arxiv          # 只运行 arxiv
#   ./run_bes_sweep.sh arxiv 64       # 只运行 arxiv dim=64
#
# 维度: 32, 64, 128, 256, 384, 512, 768
# 数据集: arxiv (full), products (neighbor), mag (neighbor)

set -e

# 配置
PYTHON="${PYTHON:-python}"
GPU="${GPU:-0}"
DIMS=(32 64 128 256 384 512 768)
DATASETS=(arxiv products mag)

# 可选的局部过滤
if [ $# -ge 1 ]; then
    DATASETS=("$1")
fi
if [ $# -ge 2 ]; then
    DIMS=("$2")
fi

# 日志目录
LOGDIR="logs/bes_sweep"
mkdir -p "$LOGDIR"

echo "============================================"
echo "BES Sweep: ${#DIMS[@]} dims × ${#DATASETS[@]} datasets"
echo "DIMS:   ${DIMS[*]}"
echo "DATASETS: ${DATASETS[*]}"
echo "GPU:    $GPU"
echo "Logs:   $LOGDIR"
echo "============================================"

for dataset in "${DATASETS[@]}"; do
    # 数据集特定参数
    case "$dataset" in
        arxiv)
            MODE="full"
            BATCH_SIZE=1024
            EVAL_BATCH_SIZE=4096
            NUM_NEIGHBORS="15 10"
            EVAL_NUM_NEIGHBORS="-1 -1"
            ;;
        products)
            MODE="neighbor"
            BATCH_SIZE=4096
            EVAL_BATCH_SIZE=2048
            NUM_NEIGHBORS="10 5"
            EVAL_NUM_NEIGHBORS="-1 -1"
            ;;
        mag)
            MODE="neighbor"
            BATCH_SIZE=4096
            EVAL_BATCH_SIZE=2048
            NUM_NEIGHBORS="10 5"
            EVAL_NUM_NEIGHBORS="-1 -1"
            ;;
    esac

    for dim in "${DIMS[@]}"; do
        echo ""
        echo ">>> [$dataset] hidden_dim=$dim mode=$MODE <<<"
        LOGFILE="$LOGDIR/${dataset}_dim${dim}.log"

        CUDA_VISIBLE_DEVICES="$GPU" $PYTHON train_unified_methods.py \
            --method bes \
            --dataset "$dataset" \
            --hidden-dim "$dim" \
            --num-layers 2 \
            --dropout 0.5 \
            --mode "$MODE" \
            --batch-size "$BATCH_SIZE" \
            --eval-batch-size "$EVAL_BATCH_SIZE" \
            --num-neighbors $NUM_NEIGHBORS \
            --eval-num-neighbors $EVAL_NUM_NEIGHBORS \
            --pretrain-epochs 200 \
            --pretrain-lr 0.0001 \
            --pretrain-weight-decay 0.0005 \
            --pretrain-input train \
            --pretrain-early-stop \
            --pretrain-patience 20 \
            --bes-tau 1.0 \
            --bes-delta 5.0 \
            --bes-alpha 1.0 \
            --cls-epochs 200 \
            --cls-lr 0.01 \
            --cls-early-stop \
            --gpu-id "$GPU" \
            2>&1 | tee "$LOGFILE"

        echo ">>> [$dataset dim=$dim] DONE (log: $LOGFILE)"
    done
done

echo ""
echo "============================================"
echo "BES Sweep Complete!"
echo "Results summary:"
for dataset in "${DATASETS[@]}"; do
    SUMMARY="results/${dataset}/bes/accuracy_summary.txt"
    if [ -f "$SUMMARY" ]; then
        echo "--- $dataset ---"
        cat "$SUMMARY"
    fi
done
echo "============================================"
