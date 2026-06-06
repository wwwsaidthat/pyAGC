#!/usr/bin/env python3
"""DGI PCA基线实验：只训练最高维度(768)，其他维度通过PCA降维产生，每维度测评5次。

用法示例:
    python dgi_PCA.py --dataset mag --mode neighbor --gpu-id 1 --num-layers 3 \\
        --dropout 0.0 --pretrain-epochs 300 --pretrain-lr 0.001 \\
        --pretrain-weight-decay 0.0 --cls-epochs 200 --cls-lr 0.01 \\
        --cls-weight-decay 0.0 --batch-size 1024 --num-neighbors 15 10 5 \\
        --eval-batch-size 4096 --eval-num-neighbors -1 -1 -1 \\
        --output-dir dgi_pca_results

等效于: train once @ hidden_dim=768, PCA → [32,64,128,256,384,512], 各eval 5次
"""

import subprocess
import sys
from pathlib import Path


def main() -> None:
    script = Path(__file__).resolve().parent.parent / "train_unified_methods.py"
    cmd = [
        sys.executable, str(script),
        "--method", "dgi",
        "--hidden-dim", "768",
        "--pca-dims", "32,64,128,256,384,512",
    ] + sys.argv[1:]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
