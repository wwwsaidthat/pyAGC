#!/usr/bin/env python3
"""统一训练脚本：监督 GCN 与多种图自监督表示学习方法。"""

import argparse
import hashlib
import inspect
import json
import logging
import random
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# 忽略 OGB / PyG 数据加载的已知无害警告
warnings.filterwarnings("ignore", category=FutureWarning, module="ogb")
warnings.filterwarnings("ignore", message=".*non-writable.*")
warnings.filterwarnings("ignore", message=".*NeighborSampler.*without.*pyg-lib.*")

import numpy as np
import platform
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch import Tensor
from torch_geometric.data import Data

from pyagc.data import get_dataset
from unified_methods import (
    BaseMethod,
    CCASSGMethod,
    MCNEMethod,
    DGIMethod,
    GRACEMethod,
    GRACEWithMRLMethod,
    GRACEWithMRLMutualLearningMethod,
    SSGEMethod,
    SupervisedGCNMethod,
    GraphCLMethod,
)


MCNE_METHODS = {"mcne", "mcne_no_cdmd", "mcne_no_hpem", "mcne_no_dals"}
GRAPHCL_METHODS = {"graphcl"}


def is_nested_method(method_name: str) -> bool:
    """Return whether a method evaluates one embedding at several prefix dimensions."""
    return "mrl" in method_name or "mcne" in method_name


# ============================================================================
# 1. 基础工具
# ============================================================================


def setup_logger(level: str, log_file: Optional[Path] = None) -> logging.Logger:
    """配置日志输出。"""
    logger = logging.getLogger("unified_ssl")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False
    logger.handlers.clear()
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger


def set_seed(seed: int) -> None:
    """设置随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def enable_torch26_compat() -> None:
    """兼容 PyTorch 2.6+ 读取 PyG 对象时的安全限制。"""
    if not hasattr(torch, "serialization") or not hasattr(torch.serialization, "add_safe_globals"):
        return
    from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
    from torch_geometric.data.storage import (
        BaseStorage,
        EdgeStorage,
        GlobalStorage,
        NodeStorage,
    )

    torch.serialization.add_safe_globals(
        [
            Data,
            DataEdgeAttr,
            DataTensorAttr,
            BaseStorage,
            NodeStorage,
            EdgeStorage,
            GlobalStorage,
        ]
    )


def resolve_device(gpu_id: str) -> torch.device:
    """解析设备参数：auto/cpu/数字。"""
    if gpu_id == "cpu":
        return torch.device("cpu")
    if gpu_id == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not gpu_id.isdigit():
        raise ValueError("--gpu-id 仅支持 auto / cpu / 非负整数")
    idx = int(gpu_id)
    if not torch.cuda.is_available():
        raise RuntimeError("当前环境未检测到 CUDA，无法指定 GPU。")
    if idx < 0 or idx >= torch.cuda.device_count():
        raise ValueError(f"无效 GPU 编号: {idx}, 可用范围 0~{torch.cuda.device_count()-1}")
    return torch.device(f"cuda:{idx}")


def resolve_infer_device(infer_device: str) -> torch.device:
    """解析推理设备参数：默认强制 CPU。"""
    if infer_device == "cpu":
        return torch.device("cpu")
    if infer_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not infer_device.isdigit():
        raise ValueError("--infer-device 仅支持 cpu / auto / 非负整数")
    idx = int(infer_device)
    if not torch.cuda.is_available():
        raise RuntimeError("当前环境未检测到 CUDA，无法指定 GPU。")
    if idx < 0 or idx >= torch.cuda.device_count():
        raise ValueError(f"无效 GPU 编号: {idx}, 可用范围 0~{torch.cuda.device_count()-1}")
    return torch.device(f"cuda:{idx}")


def run_infer_with_fallback(
    method,
    data: Data,
    mode: str,
    prefer_device: torch.device,
    eval_num_neighbors,
    eval_batch_size: int,
    logger: logging.Logger,
) -> tuple[Tensor, torch.device]:
    """优先在 GPU 上执行推理；OOM 时自动清理显存并回退到 CPU。

    返回 (result, actual_device)，actual_device 供下游线性评估沿用。
    """
    try:
        if method.is_supervised:
            result = method.supervised_predict(
                data=data, mode=mode, device=prefer_device,
                eval_num_neighbors=eval_num_neighbors,
                eval_batch_size=eval_batch_size,
            )
        else:
            result = method.infer_embeddings(
                data=data, mode=mode, device=prefer_device,
                eval_num_neighbors=eval_num_neighbors,
                eval_batch_size=eval_batch_size,
            )
        return result, prefer_device
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if not torch.cuda.is_available() or prefer_device.type != "cuda":
            raise
        msg = str(e)
        if "out of memory" not in msg.lower() and "OutOfMemory" not in msg:
            raise
        logger.warning("GPU OOM 于推理阶段 (%s)，自动回退到 CPU", msg.strip()[:120])
        torch.cuda.empty_cache()
        method.cpu()
        if method.is_supervised:
            result = method.supervised_predict(
                data=data, mode=mode, device=torch.device("cpu"),
                eval_num_neighbors=eval_num_neighbors,
                eval_batch_size=eval_batch_size,
            )
        else:
            result = method.infer_embeddings(
                data=data, mode=mode, device=torch.device("cpu"),
                eval_num_neighbors=eval_num_neighbors,
                eval_batch_size=eval_batch_size,
            )
        return result, torch.device("cpu")


def run_linear_eval_with_fallback(
    features: Tensor,
    labels: Tensor,
    train_idx: Tensor,
    val_idx: Tensor,
    num_classes: int,
    prefer_device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    logger: logging.Logger,
    early_stop: bool = False,
    patience: int = 20,
    min_delta: float = 1e-4,
) -> tuple:
    """训练线性分类头，优先 GPU，OOM 时回退 CPU。"""
    try:
        clf = train_linear_eval(
            features=features, labels=labels,
            train_idx=train_idx, val_idx=val_idx,
            num_classes=num_classes, device=prefer_device,
            epochs=epochs, lr=lr, weight_decay=weight_decay,
            batch_size=batch_size, logger=logger,
            early_stop=early_stop, patience=patience, min_delta=min_delta,
        )
        return clf, prefer_device
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if not torch.cuda.is_available() or prefer_device.type != "cuda":
            raise
        msg = str(e)
        if "out of memory" not in msg.lower() and "OutOfMemory" not in msg:
            raise
        logger.warning("GPU OOM 于线性评估阶段 (%s)，自动回退到 CPU", msg.strip()[:120])
        torch.cuda.empty_cache()
        clf = train_linear_eval(
            features=features, labels=labels,
            train_idx=train_idx, val_idx=val_idx,
            num_classes=num_classes, device=torch.device("cpu"),
            epochs=epochs, lr=lr, weight_decay=weight_decay,
            batch_size=batch_size, logger=logger,
            early_stop=early_stop, patience=patience, min_delta=min_delta,
        )
        return clf, torch.device("cpu")


def parse_dims(text: str) -> List[int]:
    """解析 MRL 多维输出。"""
    out = [int(x.strip()) for x in text.split(",") if x.strip()]
    if len(out) == 0:
        raise ValueError("--mrl-dims 至少提供 1 个维度，例如 64 或 64,128,256,512")
    if any(x <= 0 for x in out):
        raise ValueError("--mrl-dims 中所有维度必须 > 0")
    return out


def project_root() -> Path:
    return Path(__file__).resolve().parent


def resolve_checkpoints_root() -> Path:
    root = (project_root() / "checkpoints").resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def resolve_results_root(output_dir: str) -> Path:
    base = (project_root() / "results").resolve()
    base.mkdir(parents=True, exist_ok=True)
    raw = (output_dir or "").strip()
    if raw in {"", ".", "./", "results", "./results"}:
        root = base
    else:
        p = Path(raw)
        suffix = p.name if p.is_absolute() else str(p).lstrip("./")
        root = base / suffix
    root.mkdir(parents=True, exist_ok=True)
    return root


def stable_run_id(args: argparse.Namespace) -> str:
    payload = {
        k: v
        for k, v in vars(args).items()
        if k
        not in {
            "log_level",
            "output_dir",
            "train_seed",
            "runs",
            "seeds",
        }
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    digest = hashlib.sha1(blob).hexdigest()[:10]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{digest}"


def build_run_dirs(args: argparse.Namespace) -> tuple[Path, Path]:
    ckpt_root = resolve_checkpoints_root()
    ckpt_method_dir = ckpt_root / args.method / args.dataset
    ckpt_method_dir.mkdir(parents=True, exist_ok=True)

    results_root = resolve_results_root(args.output_dir)
    results_method_dir = results_root / args.dataset / args.method
    results_method_dir.mkdir(parents=True, exist_ok=True)

    tag = f"hd{int(args.hidden_dim)}_l{int(args.num_layers)}"
    if is_nested_method(args.method):
        tag = f"{tag}_mrl{max(parse_dims(args.mrl_dims))}"
    run_name = f"{stable_run_id(args)}_{tag}"

    ckpt_run_dir = ckpt_method_dir / run_name
    ckpt_run_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_run_dir / "model").mkdir(parents=True, exist_ok=True)

    results_run_dir = results_method_dir / run_name
    results_run_dir.mkdir(parents=True, exist_ok=True)
    (results_run_dir / "eval").mkdir(parents=True, exist_ok=True)

    return ckpt_run_dir, results_run_dir


def dump_run_config(run_dir: Path, args: argparse.Namespace, train_seed: int, eval_seeds: List[int]) -> None:
    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(project_root()),
        "train_seed": int(train_seed),
        "eval_seeds": [int(x) for x in eval_seeds],
        "args": vars(args),
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": getattr(torch, "__version__", "unknown"),
    }
    p = run_dir / "config.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


# ============================================================================
# 2. 数据加载（仅本地，不触发自动下载）
# ============================================================================


@dataclass
class DatasetBundle:
    """图数据和官方划分。"""

    data: Data
    train_idx: Tensor
    val_idx: Tensor
    test_idx: Tensor
    num_classes: int


def _assert_dir_non_empty(path: Path, desc: str) -> None:
    """确保目录存在且非空。"""
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"{desc} 不存在: {path}")
    if not any(path.iterdir()):
        raise FileNotFoundError(f"{desc} 为空: {path}")


def ensure_local_dataset_ready(dataset: str, root: str) -> None:
    """校验本地数据目录，防止触发下载。"""
    root_path = Path(root).resolve()
    if dataset == "arxiv":
        _assert_dir_non_empty(root_path / "ogbn_arxiv", "ogbn-arxiv 数据目录")
    elif dataset == "products":
        _assert_dir_non_empty(root_path / "ogbn_products", "ogbn-products 数据目录")
    elif dataset == "mag":
        _assert_dir_non_empty(root_path / "ogbn_mag", "ogbn-mag 数据目录")
    elif dataset == "reddit2":
        # 兼容 data/reddit2 和 data/reddit 两种目录。
        d1 = root_path / "reddit2"
        d2 = root_path / "reddit"
        if not (d1.exists() and any(d1.iterdir())) and not (d2.exists() and any(d2.iterdir())):
            raise FileNotFoundError(
                f"reddit2 数据目录缺失，请确认 {d1} 或 {d2} 存在且非空"
            )
    else:
        raise ValueError(f"不支持的数据集: {dataset}")


def load_dataset(
    dataset: str,
    root: str,
    logger: Optional[logging.Logger] = None,
    normalize_ogb_features: bool = True,
) -> DatasetBundle:
    """加载本地数据和官方 split。"""
    ensure_local_dataset_ready(dataset, root)
    x, edge_index, y, train_idx, val_idx, test_idx = get_dataset(
        name=dataset,
        root=root,
        return_splits=True,
    )
    # OGB 数据集（arxiv/mag/products）的特征未经 T.NormalizeFeatures 处理，
    # 对依赖余弦相似度的对比学习方法（GRACE/DGI/CCA-SSG），
    # 行归一化能显著提升训练稳定性。
    if normalize_ogb_features and dataset in ("arxiv", "mag", "products"):
        x = torch.nn.functional.normalize(x, p=2, dim=1)
    data = Data(x=x, edge_index=edge_index, y=y.long())
    num_classes = int(data.y.max().item()) + 1
    n = data.num_nodes
    if logger is not None:
        logger.info(
            "数据集=%s | 节点=%d | 边=%d | 特征维=%d | 类别数=%d",
            dataset,
            n,
            data.num_edges,
            data.num_features,
            num_classes,
        )
        logger.info(
            "划分: train=%d(%.2f%%), val=%d(%.2f%%), test=%d(%.2f%%)",
            train_idx.numel(),
            train_idx.numel() * 100.0 / n,
            val_idx.numel(),
            val_idx.numel() * 100.0 / n,
            test_idx.numel(),
            test_idx.numel() * 100.0 / n,
        )
    return DatasetBundle(
        data=data,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        num_classes=num_classes,
    )


def auto_mode(dataset: str, mode: str) -> str:
    """自动选择 full/neighbor。"""
    if mode != "auto":
        return mode
    return "neighbor"


# ============================================================================
# 3. 共享组件：线性分类头与评估
# ============================================================================


class LinearClassifier(nn.Module):
    """统一线性分类头。"""

    def __init__(self, in_dim: int, num_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc(x)


def iterate_index_batches(index: Tensor, batch_size: int, shuffle: bool) -> List[Tensor]:
    """索引分批。"""
    if shuffle:
        index = index[torch.randperm(index.numel())]
    return [index[i:i + batch_size] for i in range(0, index.numel(), batch_size)]


@torch.no_grad()
def predict_on_index(
    classifier: nn.Module,
    features: Tensor,
    index: Tensor,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """在线性头上进行分批预测。"""
    classifier.eval()
    preds: List[Tensor] = []
    for sub_idx in iterate_index_batches(index, batch_size=batch_size, shuffle=False):
        logits = classifier(features[sub_idx].to(device))
        preds.append(logits.argmax(dim=-1).cpu())
    return torch.cat(preds, dim=0).numpy()


def compute_metrics(preds: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """统一指标计算。"""
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision_macro": precision_score(labels, preds, average="macro", zero_division=0),
        "recall_macro": recall_score(labels, preds, average="macro", zero_division=0),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
    }


def train_linear_eval(
    features: Tensor,
    labels: Tensor,
    train_idx: Tensor,
    val_idx: Tensor,
    num_classes: int,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    logger: logging.Logger,
    early_stop: bool = False,
    patience: int = 20,
    min_delta: float = 1e-4,
) -> nn.Module:
    """冻结编码器后训练统一线性头。"""
    clf = LinearClassifier(features.size(1), num_classes).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_state: Optional[Dict[str, Tensor]] = None
    best_val = -1.0
    best_epoch = 0
    stop_epoch = 0
    patience_left = int(patience)
    for ep in range(1, epochs + 1):
        clf.train()
        total_loss = 0.0
        total_correct = 0
        total = 0
        for sub_idx in iterate_index_batches(train_idx, batch_size, shuffle=True):
            x = features[sub_idx].to(device)
            y = labels[sub_idx].to(device)
            opt.zero_grad()
            logits = clf(x)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * int(sub_idx.numel())
            total_correct += int((logits.argmax(dim=-1) == y).sum().item())
            total += int(sub_idx.numel())

        train_loss = total_loss / max(total, 1)
        train_acc = total_correct / max(total, 1)
        val_pred = predict_on_index(clf, features, val_idx, device, batch_size)
        val_acc = accuracy_score(labels[val_idx].cpu().numpy(), val_pred)
        logger.info(
            "Linear Epoch %03d | loss=%.4f | train_acc=%.4f | val_acc=%.4f",
            ep,
            train_loss,
            train_acc,
            val_acc,
        )
        if val_acc > best_val + float(min_delta):
            best_val = val_acc
            best_epoch = ep
            patience_left = int(patience)
            best_state = {k: v.detach().cpu().clone() for k, v in clf.state_dict().items()}
        elif early_stop:
            patience_left -= 1
            if patience_left <= 0:
                stop_epoch = ep
                logger.info(
                    "Linear EarlyStop | best_val=%.4f @ epoch=%d | stop_epoch=%d",
                    best_val,
                    best_epoch,
                    stop_epoch,
                )
                break

    if best_state is None:
        raise RuntimeError("线性分类头训练失败，best_state 为空。")
    clf.load_state_dict(best_state)
    return clf


# ============================================================================
# 4. 方法构建（方法类已拆分到独立文件）
# ============================================================================


def build_method(args: argparse.Namespace, in_dim: int, num_classes: int) -> BaseMethod:
    """按命令行参数创建具体方法类。"""
    mrl_dims = parse_dims(args.mrl_dims)
    if args.method == "gcn":
        return SupervisedGCNMethod(
            in_dim=in_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            num_classes=num_classes,
        )
    if args.method == "dgi":
        return DGIMethod(in_dim, args.hidden_dim, args.num_layers, args.dropout)
    if args.method == "ccassg":
        return CCASSGMethod(
            in_dim=in_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            lam=args.lam,
            p_feat_mask_1=args.p_feat_mask_1,
            p_edge_drop_1=args.p_edge_drop_1,
            p_feat_mask_2=args.p_feat_mask_2,
            p_edge_drop_2=args.p_edge_drop_2,
        )
    if args.method == "grace":
        return GRACEMethod(
            in_dim=in_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            tau=args.tau,
            p_feat_mask_1=args.p_feat_mask_1,
            p_edge_drop_1=args.p_edge_drop_1,
            p_feat_mask_2=args.p_feat_mask_2,
            p_edge_drop_2=args.p_edge_drop_2,
        )
    if args.method == "grace_mrl":
        return GRACEWithMRLMethod(
            in_dim=in_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            tau=args.tau,
            p_feat_mask_1=args.p_feat_mask_1,
            p_edge_drop_1=args.p_edge_drop_1,
            p_feat_mask_2=args.p_feat_mask_2,
            p_edge_drop_2=args.p_edge_drop_2,
            mrl_dims=mrl_dims,
            mrl_weight=args.mrl_weight,
        )
    if args.method == "grace_ml":
        return GRACEWithMRLMutualLearningMethod(
            in_dim=in_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            tau=args.tau,
            p_feat_mask_1=args.p_feat_mask_1,
            p_edge_drop_1=args.p_edge_drop_1,
            p_feat_mask_2=args.p_feat_mask_2,
            p_edge_drop_2=args.p_edge_drop_2,
            mrl_dims=mrl_dims,
            mrl_weight=args.mrl_weight,
            ml_weight=args.ml_weight,
            grace_only_epochs=args.grace_only_epochs,
            grace_ml_epochs=args.grace_ml_epochs,
            ml_module=args.ml_module,
            cdmd_tau=args.cdmd_tau,
            verbose=True,  # 默认开启 verbose 模式打印损失
        )
    if args.method in MCNE_METHODS:
        use_cdmd = bool(args.use_cdmd) and args.method != "mcne_no_cdmd"
        use_hpem = bool(args.use_hpem) and args.method != "mcne_no_hpem"
        use_dals = bool(args.use_dals) and args.method != "mcne_no_dals"
        disabled = [
            name
            for name, enabled in (("cdmd", use_cdmd), ("hpem", use_hpem), ("dals", use_dals))
            if not enabled
        ]
        expected_alias = "mcne" if not disabled else "mcne_no_" + "_no_".join(disabled)
        ablation_name = args.method if args.method == expected_alias else expected_alias
        return MCNEMethod(
            in_dim=in_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            tau=args.tau,
            p_feat_mask_1=args.p_feat_mask_1,
            p_edge_drop_1=args.p_edge_drop_1,
            p_feat_mask_2=args.p_feat_mask_2,
            p_edge_drop_2=args.p_edge_drop_2,
            mrl_dims=mrl_dims,
            mrl_weight=args.mrl_weight,
            ml_weight=args.ml_weight,
            ml_module=args.ml_module,
            cdmd_tau=args.cdmd_tau,
            hpem_beta_init=args.hpem_beta_init,
            hpem_tau_0=args.hpem_tau_0,
            hpem_phi_1_init=args.hpem_phi_1_init,
            hpem_phi_2_init=args.hpem_phi_2_init,
            dals_lambda_init=args.dals_lambda_init,
            use_cdmd=use_cdmd,
            use_hpem=use_hpem,
            use_dals=use_dals,
            ablation_name=ablation_name,
            warmup_epochs=args.grace_only_epochs,
            full_epochs=args.grace_ml_epochs,
            verbose=True,
        )
    if args.method == "ssge":
        return SSGEMethod(
            in_dim=in_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            lam=args.lam,
            p_feat_mask_1=args.p_feat_mask_1,
            p_edge_drop_1=args.p_edge_drop_1,
            p_feat_mask_2=args.p_feat_mask_2,
            p_edge_drop_2=args.p_edge_drop_2,
        )
    if args.method in GRAPHCL_METHODS:
        common = dict(
            in_dim=in_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            proj_dim=args.hidden_dim,
            tau=args.graphcl_tau,
            aug_1=args.graphcl_aug_1,
            aug_2=args.graphcl_aug_2,
            aug_ratio_1=args.graphcl_aug_ratio_1,
            aug_ratio_2=args.graphcl_aug_ratio_2,
            symmetric_loss=args.graphcl_symmetric_loss,
        )
        return GraphCLMethod(**common)
    raise ValueError(f"未知方法: {args.method}")


# ============================================================================
# 6. 单次运行与多次统计
# ============================================================================


def summarize_values(values: List[float]) -> Dict[str, float]:
    """统计均值/方差/标准差。"""
    arr = np.array(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "variance": float(arr.var(ddof=1 if len(arr) > 1 else 0)),
        "std": float(arr.std(ddof=1 if len(arr) > 1 else 0)),
    }


def attach_splits(data: Data, bundle: DatasetBundle) -> None:
    data.train_idx = bundle.train_idx
    data.val_idx = bundle.val_idx
    data.test_idx = bundle.test_idx


def save_model_checkpoint(
    run_dir: Path,
    args: argparse.Namespace,
    train_seed: int,
    method: BaseMethod,
    output_dim: int,
    num_classes: int,
    in_dim: int,
) -> Path:
    ckpt_path = run_dir / "model" / "model.pt"
    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "method": method.method_name,
        "dataset": args.dataset,
        "train_seed": int(train_seed),
        "args": vars(args),
        "output_dim": int(output_dim),
        "num_classes": int(num_classes),
        "in_dim": int(in_dim),
        "state_dict": method.state_dict(),
        "torch_version": getattr(torch, "__version__", "unknown"),
    }
    torch.save(payload, ckpt_path)
    return ckpt_path


def torch_load_unsafe(path: Path) -> Any:
    kwargs: Dict[str, Any] = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = False
    return torch.load(path, **kwargs)


def load_model_checkpoint(
    checkpoint_path: Path,
    args: argparse.Namespace,
    bundle: DatasetBundle,
    device: torch.device,
    logger: logging.Logger,
) -> BaseMethod:
    obj = torch_load_unsafe(checkpoint_path)
    if not isinstance(obj, dict) or "state_dict" not in obj:
        raise RuntimeError(f"无效 checkpoint: {checkpoint_path}")
    method = build_method(args, in_dim=bundle.data.num_features, num_classes=bundle.num_classes).to(device)
    method.load_state_dict(obj["state_dict"], strict=True)
    logger.info("加载模型参数: %s", str(checkpoint_path))
    return method


def train_once(args: argparse.Namespace, train_seed: int, bundle: DatasetBundle, logger: logging.Logger, run_dir: Path) -> Dict[str, Any]:
    set_seed(train_seed)
    device = resolve_device(args.gpu_id)
    mode = auto_mode(args.dataset, args.mode)
    data = bundle.data
    attach_splits(data, bundle)

    method = build_method(args, in_dim=data.num_features, num_classes=bundle.num_classes).to(device)
    out_dim = int(method.output_dim())
    logger.info(
        "训练开始 | 方法=%s | 数据集=%s | 表示维度=%d | 设备=%s | mode=%s | train_seed=%d",
        method.method_name,
        args.dataset,
        out_dim,
        device,
        mode,
        int(train_seed),
    )

    if method.is_supervised:
        optimizer = torch.optim.Adam(
            method.parameters(),
            lr=args.supervised_lr,
            weight_decay=args.supervised_weight_decay,
        )
        best_state: Optional[Dict[str, Tensor]] = None
        best_val = -1.0
        best_epoch = 0
        stop_epoch = 0
        patience_left = int(args.supervised_patience)
        last_epoch = 0
        for ep in range(1, args.supervised_epochs + 1):
            last_epoch = ep
            if mode == "full":
                loss = method.supervised_train_step_full(data, device, optimizer)
            else:
                loss = method.supervised_train_step_neighbor(
                    data=data,
                    train_idx=bundle.train_idx,
                    num_neighbors=args.num_neighbors,
                    batch_size=args.batch_size,
                    device=device,
                    optimizer=optimizer,
                )
            if args.supervised_early_stop and (ep % int(args.supervised_eval_every) == 0 or ep == args.supervised_epochs):
                logits_ep = method.supervised_predict(
                    data=data,
                    mode=mode,
                    device=device,
                    eval_num_neighbors=args.eval_num_neighbors,
                    eval_batch_size=args.eval_batch_size,
                )
                val_pred_ep = logits_ep[bundle.val_idx].argmax(dim=-1).numpy()
                val_acc_ep = float(accuracy_score(data.y[bundle.val_idx].cpu().numpy(), val_pred_ep))
                logger.info("Supervised Epoch %03d | loss=%.4f | val_acc=%.4f", ep, loss, val_acc_ep)
                if val_acc_ep > best_val + float(args.supervised_min_delta):
                    best_val = val_acc_ep
                    best_epoch = ep
                    patience_left = int(args.supervised_patience)
                    best_state = {k: v.detach().cpu().clone() for k, v in method.state_dict().items()}
                else:
                    patience_left -= 1
                    if patience_left <= 0:
                        stop_epoch = ep
                        logger.info(
                            "Supervised EarlyStop | best_val=%.4f @ epoch=%d | stop_epoch=%d",
                            best_val,
                            best_epoch,
                            stop_epoch,
                        )
                        break
            else:
                logger.info("Supervised Epoch %03d | loss=%.4f", ep, loss)

        if args.supervised_early_stop:
            if stop_epoch == 0:
                stop_epoch = int(last_epoch)
            if best_state is not None:
                method.load_state_dict(best_state)
        ckpt_path = save_model_checkpoint(
            run_dir=run_dir,
            args=args,
            train_seed=train_seed,
            method=method,
            output_dim=out_dim,
            num_classes=bundle.num_classes,
            in_dim=int(data.num_features),
        )
        return {
            "train_seed": int(train_seed),
            "method": method.method_name,
            "dataset": args.dataset,
            "hidden_dim": int(args.hidden_dim),
            "num_layers": int(args.num_layers),
            "output_dim": out_dim,
            "checkpoint_path": str(ckpt_path),
            "train_mode": "supervised",
            "supervised_early_stop": bool(args.supervised_early_stop),
            "supervised_best_val_acc": float(best_val) if args.supervised_early_stop else None,
            "supervised_best_epoch": int(best_epoch) if args.supervised_early_stop else None,
            "supervised_stop_epoch": int(stop_epoch) if args.supervised_early_stop else None,
        }
    else:
        # Check if method manages its own optimizer(s) (e.g., SGRL dual-optimizer)
        if hasattr(method, 'manages_own_optimizer') and method.manages_own_optimizer:
            optimizer = None  # Will be ignored; method handles optimization internally
        else:
            optimizer = torch.optim.Adam(
                method.parameters(),
                lr=float(args.pretrain_lr),
                weight_decay=args.pretrain_weight_decay,
            )
        best_ssl = float("inf")
        best_ssl_epoch = 0
        best_ssl_state: Optional[Dict[str, Tensor]] = None
        ssl_stop_epoch = 0
        ssl_patience_left = int(args.pretrain_patience)
        effective_ssl_early_stop = bool(args.pretrain_early_stop)
        for ep in range(1, args.pretrain_epochs + 1):
            # 更新 epoch（用于 verbose 模式打印）
            if hasattr(method, "epoch"):
                method.epoch = ep

            # grace_ML / MCNE 两阶段：进入第二阶段时切换学习率
            if (
                (args.method == "grace_ml" or args.method in MCNE_METHODS)
                and args.grace_ml_lr is not None
                and ep == args.grace_only_epochs + 1
            ):
                for pg in optimizer.param_groups:
                    pg["lr"] = float(args.grace_ml_lr)
                logger.info(
                    "ML 阶段开始 | lr 切换: %.6f -> %.6f",
                    float(args.pretrain_lr),
                    float(args.grace_ml_lr),
                )

            if mode == "full":
                loss = method.ssl_train_step_full(data, device, optimizer)
            else:
                input_nodes = None if args.pretrain_input == "all" else bundle.train_idx
                loss = method.ssl_train_step_neighbor(
                    data=data,
                    input_nodes=input_nodes,
                    num_neighbors=args.num_neighbors,
                    batch_size=args.batch_size,
                    device=device,
                    optimizer=optimizer,
                )
            # 构建详细损失信息（MRL/ML 方法会提供分量损失）
            extra_info = ""
            if hasattr(method, "get_last_mrl_dim_losses"):
                d = method.get_last_mrl_dim_losses()
                if d:
                    parts = []
                    for k, v in d.items():
                        if k.startswith("dim_"):
                            continue  # 维度级损失太多，跳过
                        parts.append(f"{k}={v:.4f}")
                    if parts:
                        extra_info = " | " + " | ".join(parts)
            if effective_ssl_early_stop and (ep % int(args.pretrain_eval_every) == 0 or ep == args.pretrain_epochs):
                logger.info("Pretrain Epoch %03d | loss=%.4f%s", ep, loss, extra_info)
                if loss < best_ssl - float(args.pretrain_min_delta):
                    best_ssl = float(loss)
                    best_ssl_epoch = int(ep)
                    best_ssl_state = {
                        key: value.detach().cpu().clone()
                        for key, value in method.state_dict().items()
                    }
                    ssl_patience_left = int(args.pretrain_patience)
                else:
                    ssl_patience_left -= 1
                    if ssl_patience_left <= 0:
                        ssl_stop_epoch = int(ep)
                        logger.info(
                            "Pretrain EarlyStop | best_loss=%.6f @ epoch=%d | stop_epoch=%d",
                            best_ssl,
                            best_ssl_epoch,
                            ssl_stop_epoch,
                        )
                        break
            else:
                logger.info("Pretrain Epoch %03d | loss=%.4f%s", ep, loss, extra_info)
        if effective_ssl_early_stop and best_ssl_state is not None:
            method.load_state_dict(best_ssl_state)
            logger.info("恢复预训练最佳模型 | epoch=%d | loss=%.6f", best_ssl_epoch, best_ssl)
        # 训练结束后打印诊断信息（embedding 范数、logit 尺度、饱和度）
        if hasattr(method, "diagnose"):
            try:
                method.diagnose(
                    data=data,
                    mode=mode,
                    device=device,
                    eval_num_neighbors=args.eval_num_neighbors,
                    eval_batch_size=args.eval_batch_size,
                    logger=logger,
                )
            except Exception as exc:
                logger.warning("DIAG 失败: %s", exc)
        ckpt_path = save_model_checkpoint(
            run_dir=run_dir,
            args=args,
            train_seed=train_seed,
            method=method,
            output_dim=out_dim,
            num_classes=bundle.num_classes,
            in_dim=int(data.num_features),
        )
        return {
            "train_seed": int(train_seed),
            "method": method.method_name,
            "dataset": args.dataset,
            "hidden_dim": int(args.hidden_dim),
            "num_layers": int(args.num_layers),
            "output_dim": out_dim,
            "checkpoint_path": str(ckpt_path),
            "train_mode": "ssl",
            "pretrain_early_stop": effective_ssl_early_stop,
            "pretrain_best_loss": float(best_ssl) if effective_ssl_early_stop else None,
            "pretrain_best_epoch": int(best_ssl_epoch) if effective_ssl_early_stop else None,
            "pretrain_stop_epoch": int(ssl_stop_epoch if ssl_stop_epoch > 0 else min(args.pretrain_epochs, best_ssl_epoch))
            if effective_ssl_early_stop
            else None,
        }


def evaluate_once(
    args: argparse.Namespace,
    eval_seed: int,
    bundle: DatasetBundle,
    logger: logging.Logger,
    checkpoint_path: Path,
) -> Dict[str, Any]:
    set_seed(eval_seed)
    device = resolve_device(args.gpu_id)
    infer_device = resolve_infer_device(args.infer_device)
    mode = auto_mode(args.dataset, args.mode)
    data = bundle.data
    attach_splits(data, bundle)

    method = load_model_checkpoint(checkpoint_path, args=args, bundle=bundle, device=infer_device, logger=logger)
    out_dim = int(method.output_dim())
    logger.info(
        "评估开始 | 方法=%s | 数据集=%s | 表示维度=%d | 设备=%s | mode=%s | eval_seed=%d",
        method.method_name,
        args.dataset,
        out_dim,
        infer_device,
        mode,
        int(eval_seed),
    )

    mrl_dim_test_accuracy: Dict[str, float] = {}

    if method.is_supervised:
        logits = method.supervised_predict(
            data=data,
            mode=mode,
            device=infer_device,
            eval_num_neighbors=args.eval_num_neighbors,
            eval_batch_size=args.eval_batch_size,
        )
        val_pred = logits[bundle.val_idx].argmax(dim=-1).numpy()
        test_pred = logits[bundle.test_idx].argmax(dim=-1).numpy()
    else:
        features = method.infer_embeddings(
            data=data,
            mode=mode,
            device=infer_device,
            eval_num_neighbors=args.eval_num_neighbors,
            eval_batch_size=args.eval_batch_size,
        )
        clf = train_linear_eval(
            features=features,
            labels=data.y.cpu(),
            train_idx=bundle.train_idx.cpu(),
            val_idx=bundle.val_idx.cpu(),
            num_classes=bundle.num_classes,
            device=infer_device,
            epochs=args.cls_epochs,
            lr=args.cls_lr,
            weight_decay=args.cls_weight_decay,
            batch_size=args.cls_batch_size,
            logger=logger,
            early_stop=bool(args.cls_early_stop),
            patience=int(args.cls_patience),
            min_delta=float(args.cls_min_delta),
        )
        val_pred = predict_on_index(clf, features, bundle.val_idx.cpu(), infer_device, args.cls_batch_size)
        test_pred = predict_on_index(clf, features, bundle.test_idx.cpu(), infer_device, args.cls_batch_size)

        mrl_dims_for_eval: Optional[List[int]] = None
        if hasattr(method, "mrl") and hasattr(method.mrl, "dims"):
            mrl_dims_for_eval = [int(d) for d in method.mrl.dims]
        elif hasattr(method, "mrl_dims"):
            mrl_dims_for_eval = [int(d) for d in getattr(method, "mrl_dims")]

        if mrl_dims_for_eval:
            dims = sorted(set(mrl_dims_for_eval))
            test_labels_np = data.y[bundle.test_idx].cpu().numpy()
            for dim in dims:
                dim_key = f"dim_{dim}"
                dim_features = features[:, :dim]
                dim_clf = train_linear_eval(
                    features=dim_features,
                    labels=data.y.cpu(),
                    train_idx=bundle.train_idx.cpu(),
                    val_idx=bundle.val_idx.cpu(),
                    num_classes=bundle.num_classes,
                    device=infer_device,
                    epochs=args.cls_epochs,
                    lr=args.cls_lr,
                    weight_decay=args.cls_weight_decay,
                    batch_size=args.cls_batch_size,
                    logger=logger,
                    early_stop=bool(args.cls_early_stop),
                    patience=int(args.cls_patience),
                    min_delta=float(args.cls_min_delta),
                )
                dim_test_pred = predict_on_index(dim_clf, dim_features, bundle.test_idx.cpu(), infer_device, args.cls_batch_size)
                dim_test_acc = float(accuracy_score(test_labels_np, dim_test_pred))
                mrl_dim_test_accuracy[dim_key] = dim_test_acc
                logger.info("MRL维度 %s | test_acc=%.4f", dim_key, dim_test_acc)

    val_labels = data.y[bundle.val_idx].cpu().numpy()
    test_labels = data.y[bundle.test_idx].cpu().numpy()
    val_metrics = compute_metrics(val_pred, val_labels)
    test_metrics = compute_metrics(test_pred, test_labels)
    result = {
        "eval_seed": int(eval_seed),
        "method": method.method_name,
        "dataset": args.dataset,
        "hidden_dim": int(args.hidden_dim),
        "num_layers": int(args.num_layers),
        "output_dim": out_dim,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "checkpoint_path": str(checkpoint_path),
    }
    if is_nested_method(args.method):
        result["mrl_dims"] = parse_dims(args.mrl_dims)
    if mrl_dim_test_accuracy:
        result["mrl_dim_test_accuracy"] = mrl_dim_test_accuracy
    return result


def save_results(
    run_dir: Path,
    args: argparse.Namespace,
    train_meta: Dict[str, Any],
    eval_results: List[Dict[str, Any]],
    logger: logging.Logger,
) -> None:
    eval_dir = run_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    method = train_meta["method"]
    dataset = args.dataset
    out_dim = int(train_meta.get("output_dim", -1))

    for item in eval_results:
        seed = item["eval_seed"]
        p = eval_dir / f"eval_seed{seed}.metrics.json"
        with open(p, "w", encoding="utf-8") as f:
            json.dump(item, f, ensure_ascii=False, indent=2)

    acc_values = [float(x["test_metrics"]["accuracy"]) for x in eval_results]
    stats = summarize_values(acc_values)

    # MRL维度统计
    mrl_dim_accuracy_values: Dict[str, List[float]] = {}
    for item in eval_results:
        dim_acc = item.get("mrl_dim_test_accuracy")
        if not dim_acc:
            continue
        for dim_key, value in dim_acc.items():
            mrl_dim_accuracy_values.setdefault(dim_key, []).append(float(value))

    mrl_dim_accuracy_stats: Dict[str, Dict[str, Any]] = {}
    for dim_key in sorted(mrl_dim_accuracy_values.keys()):
        dim_stats = summarize_values(mrl_dim_accuracy_values[dim_key])
        mrl_dim_accuracy_stats[dim_key] = {
            "values": mrl_dim_accuracy_values[dim_key],
            "mean": dim_stats["mean"],
            "variance": dim_stats["variance"],
            "std": dim_stats["std"],
            "mean_pm_variance": f"{dim_stats['mean']:.4f} ± {dim_stats['variance']:.6f}",
        }

    summary = {
        "dataset": dataset,
        "method": method,
        "hidden_dim": int(args.hidden_dim),
        "num_layers": int(args.num_layers),
        "output_dim": out_dim,
        "train_meta": train_meta,
        "num_eval_runs": len(eval_results),
        "eval_seeds": [int(x["eval_seed"]) for x in eval_results],
        "accuracy_values": acc_values,
        "accuracy_mean": stats["mean"],
        "accuracy_variance": stats["variance"],
        "accuracy_std": stats["std"],
        "mean_pm_variance": f"{stats['mean']:.4f} ± {stats['variance']:.6f}",
        "per_eval_results": eval_results,
    }
    if is_nested_method(args.method):
        summary["mrl_dims"] = parse_dims(args.mrl_dims)
    if mrl_dim_accuracy_stats:
        summary["mrl_dim_accuracy_stats"] = mrl_dim_accuracy_stats
    summary_json = run_dir / "summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    summary_txt = run_dir / "accuracy_summary.txt"
    with open(summary_txt, "w", encoding="utf-8") as f:
        dim_tag = f"dim{out_dim}" if out_dim > 0 else f"hd{int(args.hidden_dim)}"
        f.write(f"{dataset}-{method}-{dim_tag}: {stats['mean']:.4f} +- {stats['variance']:.6f}\n")
        if mrl_dim_accuracy_stats:
            for dim_key in sorted(mrl_dim_accuracy_stats.keys()):
                f.write(
                    f"{dataset}-{method}-{dim_key}: "
                    f"{mrl_dim_accuracy_stats[dim_key]['mean']:.4f} +- "
                    f"{mrl_dim_accuracy_stats[dim_key]['variance']:.6f}\n"
                )
    logger.info("结果保存: %s", str(summary_json))
    dataset_dir = run_dir.parent.parent
    dataset_summary = dataset_dir / "accuracy_summary.txt"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    with open(dataset_summary, "a", encoding="utf-8") as f:
        method_tag = f"{method}_hd{int(args.hidden_dim)}"
        f.write(f"{method_tag}: {stats['mean']:.4f} +- {stats['variance']:.6f}\n")
        if mrl_dim_accuracy_stats:
            for dim_key in sorted(mrl_dim_accuracy_stats.keys()):
                f.write(
                    f"{method}_{dim_key}: "
                    f"{mrl_dim_accuracy_stats[dim_key]['mean']:.4f} +- "
                    f"{mrl_dim_accuracy_stats[dim_key]['variance']:.6f}\n"
                )
        f.write("\n")

    logger.info("准确率(均值+-方差): %s", summary["mean_pm_variance"])



# ============================================================================
# 7. 命令行入口
# ============================================================================


def parse_args() -> argparse.Namespace:
    """解析参数。"""
    p = argparse.ArgumentParser(
        description="统一训练：监督 GCN 与多种图自监督表示学习方法",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        "--method",
        type=str,
        required=True,
        choices=[
            "gcn",
            "dgi",
            "ccassg",
            "grace",
            "grace_mrl",
            "grace_ml",
            "mcne",
            "mcne_no_cdmd",
            "mcne_no_hpem",
            "mcne_no_dals",
            "ssge",
            "graphcl",
        ],
    )
    p.add_argument("--dataset", type=str, required=True, choices=["arxiv", "reddit2", "products", "mag"])
    p.add_argument("--config", type=str, default=None,
                   help="YAML 配置文件；命令行显式参数优先于 YAML")
    p.add_argument("--root", type=str, default="./data")
    p.add_argument("--mode", type=str, default="auto", choices=["auto", "full", "neighbor"])
    p.add_argument("--gpu-id", type=str, default="auto", help="auto/cpu/0/1...")
    p.add_argument("--infer-device", type=str, default="auto", help="推理设备 auto/cpu/0/1... 优先GPU，OOM自动回退CPU")

    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--run-dims", type=str, default=None,
                   help="依次独立训练多个维度，例如 32,64,128,256,384,512,768")
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--tau", type=float, default=0.5, help="GRACE 温度系数")
    p.add_argument("--lam", type=float, default=1e-3, help="CCA-SSG λ")

    p.add_argument("--p-feat-mask-1", type=float, default=0.3)
    p.add_argument("--p-edge-drop-1", type=float, default=0.2)
    p.add_argument("--p-feat-mask-2", type=float, default=0.4)
    p.add_argument("--p-edge-drop-2", type=float, default=0.4)

    p.add_argument("--pretrain-epochs", type=int, default=200)
    p.add_argument("--grace-only-epochs", type=int, default=None,
                   help="grace_ML 第一阶段（纯 GRACE）训练轮数；未设置时取 pretrain-epochs 的一半")
    p.add_argument("--grace-ml-epochs", type=int, default=None,
                   help="grace_ML 第二阶段（GRACE+ML）训练轮数；未设置时取 pretrain-epochs 的一半")
    p.add_argument("--pretrain-lr", type=float, default=1e-3)
    p.add_argument("--grace-ml-lr", type=float, default=None,
                   help="grace_ML 第二阶段（ML）学习率；未设置时沿用 pretrain-lr")
    p.add_argument("--pretrain-weight-decay", type=float, default=0.0)
    p.add_argument("--pretrain-input", type=str, default="all", choices=["all", "train"])
    p.add_argument("--pretrain-early-stop", action="store_true", help="自监督预训练按 loss 提前停止")
    p.add_argument("--pretrain-patience", type=int, default=20, help="预训练提前停止 patience（按评估次数计）")
    p.add_argument("--pretrain-min-delta", type=float, default=1e-4, help="预训练 loss 最小下降阈值")
    p.add_argument("--pretrain-eval-every", type=int, default=1, help="每隔多少个 epoch 检查一次预训练 loss（用于早停）")

    p.add_argument("--supervised-epochs", type=int, default=200)
    p.add_argument("--supervised-lr", type=float, default=1e-2)
    p.add_argument("--supervised-weight-decay", type=float, default=5e-4)
    p.add_argument("--supervised-early-stop", action="store_true", help="监督训练按验证集准确率提前停止")
    p.add_argument("--supervised-patience", type=int, default=20, help="提前停止 patience（按评估次数计）")
    p.add_argument("--supervised-min-delta", type=float, default=1e-4, help="验证集准确率最小提升阈值")
    p.add_argument("--supervised-eval-every", type=int, default=5, help="每隔多少个 epoch 评估一次验证集（用于早停）")

    p.add_argument("--cls-epochs", type=int, default=200)
    p.add_argument("--cls-lr", type=float, default=1e-2)
    p.add_argument("--cls-weight-decay", type=float, default=0.0)
    p.add_argument("--cls-batch-size", type=int, default=8192)
    p.add_argument("--cls-early-stop", action="store_true", help="线性评估按验证集准确率提前停止")
    p.add_argument("--cls-patience", type=int, default=20, help="线性评估提前停止 patience（按 epoch 计）")
    p.add_argument("--cls-min-delta", type=float, default=1e-4, help="线性评估验证集准确率最小提升阈值")

    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--eval-batch-size", type=int, default=4096)
    p.add_argument("--num-neighbors", type=int, nargs="+", default=[15, 10])
    p.add_argument("--eval-num-neighbors", type=int, nargs="+", default=[-1, -1])

    p.add_argument("--mrl-dims", type=str, default="64,128,256,512", help="MRL 维度列表（逗号分隔）")
    p.add_argument("--mrl-weight", type=float, default=1.0)
    p.add_argument("--ml-weight", type=float, default=5.0, help="互学习损失权重")
    p.add_argument("--ml-module", type=str, default="ml2", choices=["ml", "ml2"],
                   help="CDMD 对齐方式: ml2=所有低维向最高维学习（默认）, ml=相邻维度学习")
    p.add_argument("--cdmd-tau", type=float, default=0.5,
                   help="兼容旧命令的保留参数；ReLU-CDMD 中不参与计算")
    p.add_argument("--disable-cdmd", dest="use_cdmd", action="store_false", default=True,
                   help="关闭 CDMD；可与其他消融开关组合")
    p.add_argument("--disable-hpem", dest="use_hpem", action="store_false", default=True,
                   help="关闭 HPEM；此时自动以标准 GRACE+MRL InfoNCE 替代")
    p.add_argument("--disable-dals", dest="use_dals", action="store_false", default=True,
                   help="关闭 DALS；HPEM 使用固定 τ_0 且各维损失等权")
    p.add_argument("--hpem-beta-init", type=float, default=0.1,
                   help="HPEM 的 β 参数初始值（可学习）")
    p.add_argument("--hpem-tau-0", type=float, default=None,
                   help="HPEM/DALS 基础温度 τ_0；默认沿用 --tau")
    p.add_argument("--hpem-phi-1-init", type=float, default=0.0,
                   help="维度自适应温度参数 φ_1 的初始值")
    p.add_argument("--hpem-phi-2-init", type=float, default=0.0,
                   help="维度自适应温度参数 φ_2 的初始值")
    p.add_argument("--dals-lambda-init", type=float, default=0.0,
                   help="DALS 可学习参数 λ 的初始值")

    # ---- GraphCL 特有参数 ----
    graphcl_aug_choices = sorted(["none", "node_drop", "edge_perturb", "subgraph", "attr_mask", "random"])
    p.add_argument("--graphcl-tau", type=float, default=0.2,
                   help="GraphCL InfoNCE 温度；原仓库默认 0.2")
    p.add_argument("--graphcl-aug-1", type=str, default="edge_perturb", choices=graphcl_aug_choices,
                   help="GraphCL 第一视图增强")
    p.add_argument("--graphcl-aug-2", type=str, default="attr_mask", choices=graphcl_aug_choices,
                   help="GraphCL 第二视图增强")
    p.add_argument("--graphcl-aug-ratio-1", type=float, default=0.2,
                   help="第一视图增强强度，范围 [0,1)")
    p.add_argument("--graphcl-aug-ratio-2", type=float, default=0.2,
                   help="第二视图增强强度，范围 [0,1)")
    p.add_argument("--graphcl-symmetric-loss", action="store_true", default=False,
                   help="使用双向 GraphCL 损失；默认关闭以贴近原仓库 loss_cal")
    p.add_argument("--eval-only", action="store_true",
                   help="仅评估模式：跳过训练，直接从 --checkpoint 加载模型进行标准评估")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="eval-only 模式下使用的 checkpoint 路径")

    p.add_argument("--output-dir", type=str, default="./results")
    p.add_argument("--log-level", type=str, default="INFO")

    args = p.parse_args()
    if args.config:
        config_path = Path(args.config).expanduser().resolve()
        with config_path.open("r", encoding="utf-8") as handle:
            raw_config = yaml.safe_load(handle) or {}
        merged: Dict[str, Any] = {}
        merged.update(raw_config.get("default", {}))
        merged.update(raw_config.get(args.dataset, {}))

        explicit = set()
        argv = sys.argv[1:]
        for action in p._actions:
            if any(
                token == option or token.startswith(option + "=")
                for token in argv
                for option in action.option_strings
            ):
                explicit.add(action.dest)

        for key, value in merged.items():
            if not hasattr(args, key):
                raise ValueError(f"配置文件包含未知参数: {key}")
            if key not in explicit:
                setattr(args, key, value)
    return args


def run_dimension_sweep(dimensions: str) -> None:
    """Run each representation dimension in an isolated process."""
    dims = parse_dims(dimensions)
    argv = sys.argv[1:]
    child_args: List[str] = []
    skip_next = False
    for token in argv:
        if skip_next:
            skip_next = False
            continue
        if token in {"--run-dims", "--hidden-dim"}:
            skip_next = True
            continue
        if token.startswith("--run-dims=") or token.startswith("--hidden-dim="):
            continue
        child_args.append(token)

    for dim in dims:
        print(f"\n{'=' * 24} representation dimension={dim} {'=' * 24}", flush=True)
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                *child_args,
                "--hidden-dim",
                str(dim),
            ],
            cwd=str(project_root()),
            check=True,
        )


def main() -> None:
    """程序入口。"""
    try:
        args = parse_args()
        if args.run_dims:
            run_dimension_sweep(args.run_dims)
            return
        t_start = time.time()

        # grace_ML / MCNE 两阶段 epoch 处理：支持显式指定两个阶段的训练轮数
        if args.grace_only_epochs is not None or args.grace_ml_epochs is not None:
            goe = args.grace_only_epochs if args.grace_only_epochs is not None else args.pretrain_epochs // 2
            gme = args.grace_ml_epochs if args.grace_ml_epochs is not None else args.pretrain_epochs // 2
            args.grace_only_epochs = goe
            args.grace_ml_epochs = gme
            args.pretrain_epochs = goe + gme
        elif args.method == "grace_ml" or args.method in MCNE_METHODS:
            args.grace_only_epochs = args.pretrain_epochs // 2
            args.grace_ml_epochs = args.pretrain_epochs - args.grace_only_epochs

        enable_torch26_compat()
        bundle = load_dataset(
            args.dataset,
            args.root,
            logger=None,
            normalize_ogb_features=True,
        )

        train_seed = 0
        eval_seeds = [0, 1, 2, 3, 4]

        if args.eval_only:
            if not args.checkpoint:
                print("错误: --eval-only 需要同时指定 --checkpoint", file=sys.stderr)
                sys.exit(1)
            checkpoint_path = Path(args.checkpoint).resolve()
            if not checkpoint_path.exists():
                print(f"错误: checkpoint 文件不存在: {checkpoint_path}", file=sys.stderr)
                sys.exit(1)
            result_dir = Path(args.output_dir).resolve()
            result_dir.mkdir(parents=True, exist_ok=True)
            logger = setup_logger(args.log_level, log_file=result_dir / "run.log")
            logger.info("EvalOnly模式 | checkpoint=%s | result_dir=%s", str(checkpoint_path), str(result_dir))
            train_meta = {"method": args.method, "output_dim": int(args.hidden_dim), "checkpoint_path": str(checkpoint_path)}
        else:
            ckpt_dir, result_dir = build_run_dirs(args)
            logger = setup_logger(args.log_level, log_file=result_dir / "run.log")
            logger.info("CheckpointDir=%s", str(ckpt_dir))
            logger.info("ResultDir=%s", str(result_dir))
            logger.info("方法=%s | 数据集=%s | train_seed=%d | eval_seeds=%s", args.method, args.dataset, int(train_seed), eval_seeds)
            dump_run_config(ckpt_dir, args=args, train_seed=int(train_seed), eval_seeds=[int(x) for x in eval_seeds])
            train_meta = train_once(args, train_seed=int(train_seed), bundle=bundle, logger=logger, run_dir=ckpt_dir)
            checkpoint_path = Path(train_meta["checkpoint_path"]).resolve()

        device = resolve_device(args.gpu_id)
        infer_device = resolve_infer_device(args.infer_device)
        mode = auto_mode(args.dataset, args.mode)
        data = bundle.data
        attach_splits(data, bundle)
        method = load_model_checkpoint(checkpoint_path, args=args, bundle=bundle, device=infer_device, logger=logger)
        method.eval()

        eval_results: List[Dict[str, Any]] = []

        if method.is_supervised:
            # 判断是否使用全图评估模式（所有eval_num_neighbors都为-1）
            is_full_graph_eval = all(n == -1 for n in args.eval_num_neighbors)
            
            if is_full_graph_eval:
                # 全图模式：只推理一次，结果重复5次（确定性结果）
                logits, actual_eval_device = run_infer_with_fallback(
                    method=method, data=data, mode=mode, prefer_device=infer_device,
                    eval_num_neighbors=args.eval_num_neighbors,
                    eval_batch_size=args.eval_batch_size, logger=logger,
                )
                val_pred = logits[bundle.val_idx].argmax(dim=-1).numpy()
                test_pred = logits[bundle.test_idx].argmax(dim=-1).numpy()
                val_labels = data.y[bundle.val_idx].cpu().numpy()
                test_labels = data.y[bundle.test_idx].cpu().numpy()
                val_metrics = compute_metrics(val_pred, val_labels)
                test_metrics = compute_metrics(test_pred, test_labels)
                base = {
                    "method": method.method_name,
                    "dataset": args.dataset,
                    "hidden_dim": int(args.hidden_dim),
                    "num_layers": int(args.num_layers),
                    "output_dim": int(method.output_dim()),
                    "val_metrics": val_metrics,
                    "test_metrics": test_metrics,
                    "checkpoint_path": str(checkpoint_path),
                }
                for seed in eval_seeds:
                    logger.info("%s 评估 eval_seed=%d %s", "=" * 20, int(seed), "=" * 20)
                    eval_results.append({**base, "eval_seed": int(seed)})
            else:
                # 邻居采样模式：执行5次独立推理（存在随机性）
                for seed in eval_seeds:
                    logger.info("%s 评估 eval_seed=%d %s", "=" * 20, int(seed), "=" * 20)
                    set_seed(int(seed))
                    logger.info(
                        "评估开始 | 方法=%s | 数据集=%s | 表示维度=%d | 设备=%s | mode=%s | eval_seed=%d",
                        method.method_name,
                        args.dataset,
                        int(method.output_dim()),
                        infer_device,
                        mode,
                        int(seed),
                    )
                    logits, actual_eval_device = run_infer_with_fallback(
                        method=method, data=data, mode=mode, prefer_device=infer_device,
                        eval_num_neighbors=args.eval_num_neighbors,
                        eval_batch_size=args.eval_batch_size, logger=logger,
                    )
                    val_pred = logits[bundle.val_idx].argmax(dim=-1).numpy()
                    test_pred = logits[bundle.test_idx].argmax(dim=-1).numpy()
                    val_labels = data.y[bundle.val_idx].cpu().numpy()
                    test_labels = data.y[bundle.test_idx].cpu().numpy()
                    val_metrics = compute_metrics(val_pred, val_labels)
                    test_metrics = compute_metrics(test_pred, test_labels)
                    eval_results.append({
                        "eval_seed": int(seed),
                        "method": method.method_name,
                        "dataset": args.dataset,
                        "hidden_dim": int(args.hidden_dim),
                        "num_layers": int(args.num_layers),
                        "output_dim": int(method.output_dim()),
                        "val_metrics": val_metrics,
                        "test_metrics": test_metrics,
                        "checkpoint_path": str(checkpoint_path),
                    })
        else:
            features, actual_infer_device = run_infer_with_fallback(
                method=method, data=data, mode=mode, prefer_device=infer_device,
                eval_num_neighbors=args.eval_num_neighbors,
                eval_batch_size=args.eval_batch_size, logger=logger,
            )

            mrl_dims_for_eval: Optional[List[int]] = None
            if hasattr(method, "mrl") and hasattr(method.mrl, "dims"):
                mrl_dims_for_eval = [int(d) for d in method.mrl.dims]
            elif hasattr(method, "mrl_dims"):
                mrl_dims_for_eval = [int(d) for d in getattr(method, "mrl_dims")]

            for seed in eval_seeds:
                logger.info("%s 评估 eval_seed=%d %s", "=" * 20, int(seed), "=" * 20)
                set_seed(int(seed))
                logger.info(
                    "评估开始 | 方法=%s | 数据集=%s | 表示维度=%d | 设备=%s | mode=%s | eval_seed=%d",
                    method.method_name,
                    args.dataset,
                    int(method.output_dim()),
                    actual_infer_device,
                    mode,
                    int(seed),
                )

                clf, actual_infer_device = run_linear_eval_with_fallback(
                    features=features,
                    labels=data.y.cpu(),
                    train_idx=bundle.train_idx.cpu(),
                    val_idx=bundle.val_idx.cpu(),
                    num_classes=bundle.num_classes,
                    prefer_device=actual_infer_device,
                    epochs=args.cls_epochs,
                    lr=args.cls_lr,
                    weight_decay=args.cls_weight_decay,
                    batch_size=args.cls_batch_size,
                    logger=logger,
                    early_stop=bool(args.cls_early_stop),
                    patience=int(args.cls_patience),
                    min_delta=float(args.cls_min_delta),
                )
                val_pred = predict_on_index(clf, features, bundle.val_idx.cpu(), actual_infer_device, args.cls_batch_size)
                test_pred = predict_on_index(clf, features, bundle.test_idx.cpu(), actual_infer_device, args.cls_batch_size)

                mrl_dim_test_accuracy: Dict[str, float] = {}
                if mrl_dims_for_eval:
                    dims = sorted(set(mrl_dims_for_eval))
                    test_labels_np = data.y[bundle.test_idx].cpu().numpy()
                    for dim in dims:
                        dim_key = f"dim_{dim}"
                        dim_features = features[:, :dim]
                        dim_clf, _ = run_linear_eval_with_fallback(
                            features=dim_features,
                            labels=data.y.cpu(),
                            train_idx=bundle.train_idx.cpu(),
                            val_idx=bundle.val_idx.cpu(),
                            num_classes=bundle.num_classes,
                            prefer_device=actual_infer_device,
                            epochs=args.cls_epochs,
                            lr=args.cls_lr,
                            weight_decay=args.cls_weight_decay,
                            batch_size=args.cls_batch_size,
                            logger=logger,
                            early_stop=bool(args.cls_early_stop),
                            patience=int(args.cls_patience),
                            min_delta=float(args.cls_min_delta),
                        )
                        dim_test_pred = predict_on_index(
                            dim_clf, dim_features, bundle.test_idx.cpu(), actual_infer_device, args.cls_batch_size
                        )
                        dim_test_acc = float(accuracy_score(test_labels_np, dim_test_pred))
                        mrl_dim_test_accuracy[dim_key] = dim_test_acc
                        logger.info("MRL维度 %s | test_acc=%.4f", dim_key, dim_test_acc)

                val_labels = data.y[bundle.val_idx].cpu().numpy()
                test_labels = data.y[bundle.test_idx].cpu().numpy()
                val_metrics = compute_metrics(val_pred, val_labels)
                test_metrics = compute_metrics(test_pred, test_labels)
                result: Dict[str, Any] = {
                    "eval_seed": int(seed),
                    "method": method.method_name,
                    "dataset": args.dataset,
                    "hidden_dim": int(args.hidden_dim),
                    "num_layers": int(args.num_layers),
                    "output_dim": int(method.output_dim()),
                    "val_metrics": val_metrics,
                    "test_metrics": test_metrics,
                    "checkpoint_path": str(checkpoint_path),
                }
                if is_nested_method(args.method):
                    result["mrl_dims"] = parse_dims(args.mrl_dims)
                if mrl_dim_test_accuracy:
                    result["mrl_dim_test_accuracy"] = mrl_dim_test_accuracy
                eval_results.append(result)

        save_results(run_dir=result_dir, args=args, train_meta=train_meta, eval_results=eval_results, logger=logger)

        # ---- 记录总耗时 ----
        t_end = time.time()
        elapsed = int(t_end - t_start)
        h, rem = divmod(elapsed, 3600)
        m, s = divmod(rem, 60)
        time_dir = project_root() / "time"
        time_dir.mkdir(parents=True, exist_ok=True)
        time_file = time_dir / "timecost.txt"
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        tag = f"{args.method}_{args.dataset}_hd{int(args.hidden_dim)}"
        line = f"[{ts}] {tag} | {h}h {m}m {s}s\n"
        with open(time_file, "a", encoding="utf-8") as f:
            f.write(line)
        logger.info("总耗时: %dh %dm %ds → %s", h, m, s, str(time_file))
    except KeyboardInterrupt:
        print("\n检测到用户中断，程序已退出。", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:  # noqa: BLE001
        print(f"运行失败: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
