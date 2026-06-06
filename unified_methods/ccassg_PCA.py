#!/usr/bin/env python3
"""CCA-SSG PCA基线实验：只训练最高维度(768)，其他维度通过PCA降维产生，每维度测评5次。

用法示例:
    python ccassg_PCA.py --dataset arxiv --mode neighbor --gpu-id auto \\
        --num-layers 2 --dropout 0.0 --pretrain-epochs 100 --pretrain-lr 0.001 \\
        --pretrain-weight-decay 0.0 --lam 0.001 --p-feat-mask-1 0.0 \\
        --p-edge-drop-1 0.5 --p-feat-mask-2 0.0 --p-edge-drop-2 0.5 \\
        --batch-size 1024 --num-neighbors 15 10 --eval-batch-size 4096 \\
        --eval-num-neighbors -1 -1 --output-dir ccassg_pca_results

等效于: train once @ hidden_dim=768, PCA → [32,64,128,256,384,512], 各eval 5次
"""

import subprocess
import sys
from pathlib import Path


def main() -> None:
    script = Path(__file__).resolve().parent.parent / "train_unified_methods.py"
    cmd = [
        sys.executable, str(script),
        "--method", "ccassg",
        "--hidden-dim", "768",
        "--pca-dims", "32,64,128,256,384,512",
    ] + sys.argv[1:]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
