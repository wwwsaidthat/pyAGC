# PyAGC

**⚠️ 重要提示：数据文件缺失说明**

> 本仓库由于 **GitHub 单文件 100MB 限制** 及 **Git LFS 配额限制**，**无法直接上传核心数据文件**。这意味着您克隆仓库后**无法直接运行项目**，必须先完成下方【数据补全部署】章节的数据准备步骤。
> 
> 缺失的数据文件总体积约 **2.9 GB**，包含图神经网络训练所需的预处理数据、特征文件和划分索引。

---

## 📋 目录

- [⚠️ 数据缺失说明](#️-重要提示数据文件缺失说明)
- [🚀 项目简介](#-项目简介)
- [📦 功能特性](#-功能特性)
- [🗂️ 项目结构](#️-项目结构)
- [💻 环境要求](#-环境要求)
- [🔧 安装部署](#-安装部署)
- [📥 数据补全部署（必做）](#-数据补全部署必做)
  - [缺失文件清单](#缺失文件清单)
  - [数据获取方式](#数据获取方式)
  - [数据部署步骤](#数据部署步骤)
- [▶️ 运行验证](#️-运行验证)
- [🌐 替代方案](#-替代方案)
- [📝 注意事项](#-注意事项)
- [📄 许可证](#-许可证)

---

## 🚀 项目简介

PyAGC 是一个基于 **PyTorch** 与 **PyTorch Geometric** 的属性图聚类与图表示学习项目。本项目既提供可复用的 Python 包，也包含一套面向实验的统一训练脚本。

### 核心能力

- 支持多种图自监督学习方法（DGI、CCA-SSG、GRACE 等）
- 支持 MRL（Matryoshka Representation Learning）融合版本
- 提供统一的训练和评估流水线
- 支持多种图数据集（Cora、ogbn-arxiv、ogbn-products、reddit2）

---

## 📦 功能特性

### 1. 核心包 `pyagc`

| 模块 | 功能描述 |
|------|----------|
| `pyagc.models` | 图神经网络模型（DGI、CCASSG、GRACE、S3GC、MAGI 等 30+ 模型） |
| `pyagc.clusters` | 聚类头与 K-Means 实现（支持 PyTorch、Triton、sklearn 后端） |
| `pyagc.encoders` | 图编码器（H2GCN、SGFormer、PolyNormer、TabEncoder） |
| `pyagc.data` | 数据集加载接口（支持 GraphLand、OGB、PyG 格式） |
| `pyagc.transforms` | 图增强（随机删边、特征掩码、子图采样） |
| `pyagc.metrics` | 评估指标（NMI、ARI、ACC、Modularity、Conductance） |
| `pyagc.utils` | 工具函数（checkpoint、日志、随机种子） |

### 2. 统一训练脚本

`train_unified_methods.py` 支持的方法：

- `gcn` - 监督式 GCN
- `dgi` - Deep Graph Infomax
- `ccassg` - CCA-SSG
- `grace` - GRACE
- `dgi_mrl` - DGI + MRL
- `ccassg_mrl` - CCA-SSG + MRL
- `grace_mrl` - GRACE + MRL

脚本特性：
- 自动选择 `full` 或 `neighbor` 训练/推理模式
- 支持预训练、线性评估、监督训练
- 支持早停、多轮评估统计
- PyTorch 2.6+ PyG 序列化兼容

---

## 🗂️ 项目结构

```text
pyAGC/
├── pyagc/                      # 核心 Python 包
│   ├── clusters/               # 聚类实现
│   ├── data/                   # 数据加载
│   ├── encoders/               # 图编码器
│   ├── metrics/                # 评估指标
│   ├── models/                 # GNN 模型
│   ├── transforms/             # 数据增强
│   └── utils/                  # 工具函数
├── unified_methods/            # 统一训练方法实现
├── train_unified_methods.py    # 统一训练入口脚本
├── pyproject.toml              # 项目配置与依赖
├── gcn.yaml                    # GCN 实验配置
├── dgi.yaml                    # DGI 实验配置
├── ccassg.yaml                 # CCA-SSG 实验配置
├── grace.yaml                  # GRACE 实验配置
└── .gitignore                  # Git 忽略规则
```

---

## 💻 环境要求

- **Python**: >= 3.10
- **PyTorch**: >= 2.0
- **PyTorch Geometric**: >= 2.7.0

核心依赖：
```
torch
torch-geometric >= 2.7.0
pytorch-frame
scikit-learn
numpy
scipy
matplotlib
pyyaml
ogb
```

---

## 🔧 安装部署

### 步骤 1：克隆仓库

```bash
git clone https://github.com/wwwsaidthat/pyAGC.git
cd pyAGC
```

### 步骤 2：创建虚拟环境

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate  # Windows
python -m pip install --upgrade pip
```

### 步骤 3：安装依赖

```bash
# 安装 PyTorch（根据你的 CUDA 版本选择）
pip install torch

# 安装 PyTorch Geometric
pip install torch-geometric

# 安装项目
pip install -e .

# 可选：安装开发依赖
pip install -e ".[dev]"
```

---

## � 数据补全部署（必做）

**⚠️ 这是运行项目前的必需步骤！**

由于 GitHub 仓库限制，以下数据文件需要手动下载并放置到指定目录。

### 缺失文件清单

| 数据集 | 文件路径 | 大小 | 说明 |
|--------|----------|------|------|
| ogbn-products | `data/ogbn_products/processed/geometric_data_processed.pt` | ~2.8 GB | 预处理后的图数据（**最大文件**） |
| ogbn-products | `data/ogbn_products/raw/node-feat.csv.gz` | ~1.1 GB | 原始节点特征 |
| ogbn-products | `data/ogbn_products/raw/edge.csv.gz` | ~287 MB | 原始边列表 |
| ogbn-arxiv | `data/ogbn_arxiv/processed/geometric_data_processed.pt` | ~103 MB | 预处理后的图数据 |
| ogbn-arxiv | `data/ogbn_arxiv/raw/node-feat.csv.gz` | ~72 MB | 原始节点特征 |
| ogbn-arxiv | `data/ogbn_arxiv/raw/edge.csv.gz` | ~48 MB | 原始边列表 |
| Cora | `data/Cora/processed/data.pt` | ~15 MB | 预处理数据 |
| **总计** | - | **~4.5 GB** | - |

### 数据获取方式

#### 方式一：官方 OGB 下载（推荐）

使用 OGB 官方工具下载：

```bash
# 安装 ogb 包
pip install ogb

# 使用 Python 下载
python -c "
from ogb.nodeproppred import NodePropPredDataset
import os

# 设置数据保存路径
os.environ['OGB_DIR'] = './data'

# 下载 arxiv
dataset = NodePropPredDataset(name='ogbn-arxiv', root='./data')

# 下载 products  
dataset = NodePropPredDataset(name='ogbn-products', root='./data')

print('Download complete!')
"
```

#### 方式二：百度网盘下载

如果官方下载较慢，可以使用百度网盘镜像：

- **链接**: https://pan.baidu.com/s/1pyAGC_data_placeholder 
- **提取码**: `pyag`

下载后解压到 `data/` 目录即可。

#### 方式三：Google Drive（国际用户）

- **链接**: https://drive.google.com/drive/folders/pyAGC_data_placeholder

### 数据部署步骤

**步骤 1**: 创建数据目录

```bash
mkdir -p data
```

**步骤 2**: 下载数据并放置到对应目录

期望的目录结构：

```text
data/
├── Cora/
│   ├── processed/
│   │   ├── data.pt
│   │   ├── pre_filter.pt
│   │   └── pre_transform.pt
│   └── raw/
│       ├── ind.cora.x
│       ├── ind.cora.tx
│       ├── ind.cora.allx
│       ├── ind.cora.y
│       ├── ind.cora.ty
│       ├── ind.cora.ally
│       ├── ind.cora.graph
│       └── ind.cora.test.index
├── ogbn_arxiv/
│   ├── processed/
│   │   ├── geometric_data_processed.pt
│   │   ├── pre_filter.pt
│   │   └── pre_transform.pt
│   ├── raw/
│   │   ├── edge.csv.gz
│   │   ├── node-feat.csv.gz
│   │   ├── node-label.csv.gz
│   │   ├── node_year.csv.gz
│   │   ├── num-edge-list.csv.gz
│   │   └── num-node-list.csv.gz
│   └── split/
│       └── time/
│           ├── test.csv.gz
│           ├── train.csv.gz
│           └── valid.csv.gz
└── ogbn_products/
    ├── processed/
    │   ├── geometric_data_processed.pt (~2.8GB)
    │   ├── pre_filter.pt
    │   └── pre_transform.pt
    ├── raw/
    │   ├── edge.csv.gz (~287MB)
    │   ├── node-feat.csv.gz (~1.1GB)
    │   ├── node-label.csv.gz
    │   ├── num-edge-list.csv.gz
    │   └── num-node-list.csv.gz
    └── split/
        └── sales_ranking/
            ├── test.csv.gz
            ├── train.csv.gz
            └── valid.csv.gz
```

**步骤 3**: 验证数据完整性

```bash
# 检查文件是否存在
ls -lh data/ogbn_products/processed/geometric_data_processed.pt
ls -lh data/ogbn_arxiv/processed/geometric_data_processed.pt
ls -lh data/Cora/processed/data.pt
```

---

## ▶️ 运行验证

完成数据部署后，可以通过以下命令验证项目是否正常运行：

### 验证 1：Python 包导入

```bash
python -c "
import pyagc
from pyagc.data import get_dataset
from pyagc.models import DGI, CCASSG
print('✅ 包导入成功！')
"
```

### 验证 2：小规模数据集测试（Cora）

```bash
# 测试监督式 GCN
python train_unified_methods.py \
  --method gcn \
  --dataset cora \
  --root ./data \
  --mode full \
  --gpu-id auto \
  --supervised-epochs 10
```

如果成功运行并输出训练日志，说明部署成功！

### 验证 3：检查输出文件

```bash
# 检查是否生成了结果目录
ls -la checkpoints/
ls -la results/
```

---

## 🌐 替代方案

如果数据下载和本地部署存在困难，我们还提供以下替代方案：

### 方案 1：云存储托管（推荐用于团队协作）

将数据托管到云存储服务：

- **阿里云 OSS**：上传数据后生成临时访问链接
- **AWS S3**：使用 presigned URL 共享数据
- **Google Cloud Storage**：国际用户推荐

操作步骤：

1. 管理员将 `data/` 目录打包上传到云存储
2. 生成带时效的下载链接
3. 团队成员通过 `wget` 或 `curl` 下载：
   ```bash
   wget "https://your-cloud-storage.com/pyagc-data.tar.gz?token=xxx" -O data.tar.gz
   tar -xzf data.tar.gz
   ```

### 方案 2：数据分卷上传（GitHub 辅助方案）

对于小于 100MB 的数据文件，可以考虑：

1. 将大文件分割成多个小文件：
   ```bash
   # 将大文件分割为 90MB 的块
   split -b 90M large_file.pt large_file.pt.part.
   ```

2. 上传分卷文件到 GitHub Release 附件

3. 用户下载后合并：
   ```bash
   cat large_file.pt.part.* > large_file.pt
   ```

### 方案 3：Docker 镜像（预装数据）

构建包含数据的 Docker 镜像：

```dockerfile
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

WORKDIR /workspace

# 安装依赖
RUN pip install torch-geometric pytorch-frame ogb

# 复制代码
COPY . .

# 安装项目
RUN pip install -e .

# 下载数据（通过 OGB）
RUN python -c "from ogb.nodeproppred import NodePropPredDataset; NodePropPredDataset(name='ogbn-arxiv', root='./data'); NodePropPredDataset(name='ogbn-products', root='./data')"

CMD ["/bin/bash"]
```

构建和运行：

```bash
docker build -t pyagc:latest .
docker run --gpus all -it pyagc:latest
```

---

## 📝 注意事项

### 数据相关

1. **数据文件不在 Git 管理中**：根目录 `data/` 已通过 `.gitignore` 排除，无法通过 `git clone` 获取
2. **首次运行前必须准备数据**：否则会报错 `FileNotFoundError`
3. **数据版本兼容性**：使用 OGB 官方工具下载的数据与本项目兼容
4. **磁盘空间**：确保至少有 **5GB** 可用空间（数据 2.9GB + 运行缓存）

### 运行相关

1. **GPU 内存**：
   - `ogbn-products` 数据集需要至少 **8GB** GPU 显存
   - 显存不足时可使用 `--mode neighbor` 进行邻居采样训练

2. **Python 版本**：必须使用 **Python >= 3.10**

3. **PyTorch 版本**：建议 **>= 2.0**，以支持 PyG 2.7+

### 常见问题

**Q: 克隆仓库后为什么不能直接运行？**  
A: 因为数据文件（约 2.9GB）超过了 GitHub 的文件大小限制，无法上传。请参考【数据补全部署】章节获取数据。

**Q: 可以使用 CPU 运行吗？**  
A: 可以，但速度较慢。使用 `--gpu-id cpu` 参数指定 CPU 运行。

**Q: 数据从哪里下载？**  
A: 推荐通过 OGB 官方工具自动下载（见【数据获取方式】），或从百度网盘/云存储获取。

---

## � 许可证

本项目在 `pyproject.toml` 中声明为 **MIT License**。

```
MIT License

Copyright (c) 2024 Yunhui Liu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 📮 联系我们

如有问题或建议，欢迎通过以下方式联系：

- **GitHub Issues**: [https://github.com/wwwsaidthat/pyAGC/issues](https://github.com/wwwsaidthat/pyAGC/issues)
- **Email**: lyhcloudy1225@gmail.com

---

**⭐ 如果本项目对您有帮助，欢迎 Star 支持！**