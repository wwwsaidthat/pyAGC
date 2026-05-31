# PyAGC

PyAGC 是一个基于 PyTorch 与 PyTorch Geometric 的属性图聚类与图表示学习项目，既提供可复用的 Python 包，也包含一套面向实验的统一训练脚本。当前仓库已经上传了代码与配置文件，体积较大的数据文件默认不纳入 Git 管理，需要在本地自行准备。

## 📌 Overview

- 提供 `pyagc` 包，覆盖图聚类、图表示学习、数据加载、增强、指标与工具模块。
- 提供统一训练入口 `train_unified_methods.py`，支持监督式 GCN、DGI、CCA-SSG、GRACE 及其 MRL 版本。
- 内置多个实验配置文件，如 `gcn.yaml`、`dgi.yaml`、`ccassg.yaml`、`grace.yaml`。
- 支持 `Cora`、`ogbn-arxiv`、`ogbn-products`、`reddit2` 等图数据集的本地加载。

## ✨ Features

### 1. 📦 Core Package `pyagc`

- `pyagc.models`
  - 包含 `DGI`、`DMoN`、`MinCut`、`Neuromap`、`GBT`、`CCASSG`、`NS4GC`、`SAGSC`、`SGC`、`SSGC`
  - 包含 `GAE`、`VGAE`、`ARGA`、`ARGVA`
  - 包含 `Node2Vec`、`S3GC`、`MAGI`、`S2CAG`、`MS2CAG`、`GCSBM`、`DinkNet`、`DAEGC`、`NAFS`
- `pyagc.clusters`
  - 聚类头与 K-Means 相关实现
- `pyagc.encoders`
  - 图编码器实现，如 `H2GCN`、`SGFormer`、`PolyNormer`、`TabEncoder`
- `pyagc.data`
  - 数据集加载接口 `get_dataset`
  - `GraphLandDataset` 与 Tabular GraphLand 相关封装
- `pyagc.transforms`
  - 图增强与特征增强，如随机删边、随机特征掩码
- `pyagc.metrics`
  - 标签指标与结构指标
- `pyagc.utils`
  - checkpoint 与通用辅助函数

### 2. 🚀 Unified Training Script

`train_unified_methods.py` 当前支持以下方法：

- `gcn`
- `dgi`
- `ccassg`
- `grace`
- `dgi_mrl`
- `ccassg_mrl`
- `grace_mrl`

脚本特性包括：

- 自动选择 `full` 或 `neighbor` 推理/训练模式
- 统一记录 `checkpoints/` 与 `results/`
- 支持预训练、线性评估、监督训练与多次评估统计
- 支持早停、日志输出、结果摘要导出
- 为 PyTorch 2.6+ 的 PyG 序列化限制做了兼容处理

## 🗂️ Structure

```text
pyAGC/
├── pyagc/
│   ├── clusters/
│   ├── data/
│   ├── encoders/
│   ├── metrics/
│   ├── models/
│   ├── transforms/
│   └── utils/
├── unified_methods/
├── train_unified_methods.py
├── pyproject.toml
├── gcn.yaml
├── dgi.yaml
├── ccassg.yaml
└── grace.yaml
```

## 🧰 Environment

- Python `>= 3.10`
- `torch`
- `torch-geometric >= 2.7.0`
- `pytorch-frame`
- `scikit-learn`
- `numpy`
- `scipy`
- `matplotlib`
- `pyyaml`
- `ogb`

## 🔧 Installation

建议先按你的 CUDA/CPU 环境安装匹配版本的 `torch` 与 `torch-geometric`，再安装项目本身。

### 1. 📥 Clone Repository

```bash
git clone https://github.com/wwwsaidthat/pyAGC.git
cd pyAGC
```

### 2. 🐍 Create Environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 3. 📚 Install Dependencies

```bash
pip install torch
pip install torch-geometric
pip install -e .
```

如果你需要开发依赖：

```bash
pip install -e ".[dev]"
```

## 🗃️ Dataset

当前仓库默认忽略根目录下的 `data/`，因此运行前需要手动准备本地数据集目录。统一训练脚本不会主动为你下载数据，而是直接检查本地目录是否存在。

默认数据根目录：

```text
./data
```

期望的数据目录结构如下：

```text
data/
├── Cora/
├── ogbn_arxiv/
├── ogbn_products/
└── reddit2/   # 或 reddit/
```

脚本会按如下名称检查目录：

- `cora` -> `data/Cora`
- `arxiv` -> `data/ogbn_arxiv`
- `products` -> `data/ogbn_products`
- `reddit2` -> `data/reddit2` 或 `data/reddit`

## ⚡ Quick Start

### 1. 🧩 Import As A Python Package

```python
import pyagc
from pyagc.data import get_dataset
from pyagc.models import DGI, CCASSG, DMoN
from unified_methods import GRACEMethod
```

### 2. ▶️ Run Unified Training

监督式 GCN：

```bash
python train_unified_methods.py \
  --method gcn \
  --dataset cora \
  --root ./data \
  --mode full \
  --gpu-id auto
```

DGI 自监督训练：

```bash
python train_unified_methods.py \
  --method dgi \
  --dataset arxiv \
  --root ./data \
  --mode neighbor \
  --gpu-id auto
```

GRACE + MRL：

```bash
python train_unified_methods.py \
  --method grace_mrl \
  --dataset products \
  --root ./data \
  --mode neighbor \
  --mrl-dims 64,128,256,512 \
  --gpu-id auto
```

### 3. 🛠️ Common Arguments

- `--method`: `gcn`、`dgi`、`ccassg`、`grace`、`dgi_mrl`、`ccassg_mrl`、`grace_mrl`
- `--dataset`: `cora`、`arxiv`、`reddit2`、`products`
- `--root`: 数据根目录，默认 `./data`
- `--mode`: `auto`、`full`、`neighbor`
- `--gpu-id`: `auto`、`cpu` 或具体 GPU 编号
- `--hidden-dim`: 隐层维度
- `--num-layers`: 层数
- `--batch-size`: 邻居采样训练批大小
- `--eval-batch-size`: 评估批大小
- `--mrl-dims`: MRL 输出维度列表

## 📤 Outputs

运行统一训练脚本后，结果默认保存在以下目录：

- `checkpoints/<method>/<dataset>/...`
- `results/<dataset>/<method>/...`

每次运行通常会生成：

- `config.json`
- `model/model.pt`
- `eval/eval_seed*.metrics.json`
- `summary.json`
- `accuracy_summary.txt`
- `run.log`

## ⚙️ Config

仓库内提供的 YAML 配置文件可作为不同方法的实验参考：

- `gcn.yaml`
- `dgi.yaml`
- `ccassg.yaml`
- `grace.yaml`

这些文件主要描述：

- 训练轮数、学习率、权重衰减
- GNN 结构配置
- mini-batch 与推理参数
- 数据集特定超参数
- 对比学习增强参数

## 📝 Notes

- 根目录 `data/` 当前未上传到仓库，需要你在本地自行准备。
- `.DS_Store`、`__pycache__/`、`*.pyc` 已加入忽略规则，不会提交到 Git。
- `pyproject.toml` 中保留了项目元信息与依赖定义，可直接用于 `pip install -e .`。
- 如果使用 `ogbn-products` 等大规模数据集，建议优先使用 `neighbor` 模式并合理设置 `batch-size`。

## 📄 License

本项目在 `pyproject.toml` 中声明为 MIT License。
