#!/usr/bin/env python3
"""GRACE PCA基线实验：只训练最高维度(768)，其他维度通过PCA降维产生，每维度测评5次。

用法示例:
    python grace_PCA.py --dataset products --mode neighbor --gpu-id auto \\
        --num-layers 2 --proj-dim 768 --dropout 0.0 --tau 0.1 \\
        --pretrain-epochs 20 --pretrain-lr 0.001 --pretrain-weight-decay 0.00001 \\
        --batch-size 4096 --num-neighbors 10 10 --eval-batch-size 4096 \\
        --eval-num-neighbors -1 -1 --p-feat-mask-1 0.0 --p-edge-drop-1 0.5 \\
        --p-feat-mask-2 0.0 --p-edge-drop-2 0.5 --output-dir grace_pca_results

等效于: train once @ hidden_dim=768, PCA → [32,64,128,256,384,512], 各eval 5次
"""

import subprocess
import sys
from pathlib import Path


def main() -> None:
    script = Path(__file__).resolve().parent / "train_unified_methods.py"
    cmd = [
        sys.executable, str(script),
        "--method", "grace",
        "--hidden-dim", "768",
        "--pca-dims", "32,64,128,256,384,512",
    ] + sys.argv[1:]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
