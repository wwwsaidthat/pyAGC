import inspect
from typing import Any, Dict, List, Optional

import torch_frame
from torch import Tensor
from torch.nn import Module
try:
    from torch_frame import stype as stype
except Exception:
    try:
        import torch_frame.stype as stype  # type: ignore[no-redef]
    except Exception:
        stype = None  # type: ignore[assignment]

try:
    from torch_frame import TensorFrame as TensorFrame
except Exception:
    try:
        from torch_frame.data import TensorFrame as TensorFrame  # type: ignore[no-redef]
    except Exception:
        TensorFrame = None  # type: ignore[assignment]

try:
    from torch_frame.data.stats import StatType as StatType
except Exception:
    StatType = None  # type: ignore[assignment]
from torch_frame.nn.models import ResNet
from torch_geometric.typing import Adj, OptTensor


def _get_stype_member(name: str) -> Any:
    if hasattr(torch_frame, name):
        return getattr(torch_frame, name)
    if stype is not None and hasattr(stype, name):
        return getattr(stype, name)
    return None


def _require_torch_frame() -> None:
    if stype is None or TensorFrame is None or StatType is None:
        raise ImportError("torch_frame 版本不兼容：缺少 stype/TensorFrame/StatType")
    required = [
        "categorical",
        "numerical",
        "multicategorical",
        "embedding",
        "timestamp",
    ]
    missing = [k for k in required if _get_stype_member(k) is None]
    if missing:
        raise ImportError(f"torch_frame 版本不兼容：缺少 stype 成员 {missing}")


class TabularEncoder(Module):
    r"""
    Tabular encoder using PyTorch Frame. It maps a single TensorFrame into embeddings.

    Args:
        channels (int): Output embedding dimension.
        col_names_dict (Dict[torch_frame.stype, List[str]]):
            A mapping from stype → column names.
        col_stats (Dict[str, Dict[StatType, Any]]):
            Column statistics computed from the training set only.
        torch_frame_model_cls (defaults to ResNet):
            TorchFrame encoder class to use.
        torch_frame_model_kwargs (Dict[str, Any]): Keyword arguments for
            :class:`torch_frame_model_cls` class. Default keyword argument is
            set specific for :class:`torch_frame.nn.ResNet`. Expect it to
            be changed for different :class:`torch_frame_model_cls`.
        default_stype_encoder_cls_kwargs (Dict[torch_frame.stype, Any]):
            A dictionary mapping from :obj:`torch_frame.stype` object into a
            tuple specifying :class:`torch_frame.nn.StypeEncoder` class and its
            keyword arguments :obj:`kwargs`.
    """

    def __init__(
        self,
        channels: int,
        col_names_dict: Dict[Any, List[str]],
        col_stats: Dict[str, Dict[Any, Any]],
        torch_frame_model_cls=ResNet,
        torch_frame_model_kwargs: Optional[Dict[str, Any]] = None,
        default_stype_encoder_cls_kwargs: Optional[Dict[Any, Any]] = None,
    ):
        super().__init__()
        _require_torch_frame()

        if torch_frame_model_kwargs is None:
            torch_frame_model_kwargs = {
                "channels": 128,
                "num_layers": 2,
            }
        if default_stype_encoder_cls_kwargs is None:
            default_stype_encoder_cls_kwargs = {
                _get_stype_member("categorical"): (torch_frame.nn.EmbeddingEncoder, {}),
                _get_stype_member("numerical"): (torch_frame.nn.LinearEncoder, {}),
                _get_stype_member("multicategorical"): (
                    torch_frame.nn.MultiCategoricalEmbeddingEncoder,
                    {},
                ),
                _get_stype_member("embedding"): (torch_frame.nn.LinearEmbeddingEncoder, {}),
                _get_stype_member("timestamp"): (torch_frame.nn.TimestampEncoder, {}),
            }

        # Build stype → Encoder module dict
        stype_encoder_dict = {
            st: cls(**kwargs)
            for st, (cls, kwargs) in default_stype_encoder_cls_kwargs.items()
            if st in col_names_dict  # only keep stypes present in this table
        }

        # The actual TorchFrame model (e.g., ResNet)
        self.encoder = torch_frame_model_cls(
            **torch_frame_model_kwargs,
            out_channels=channels,
            col_stats=col_stats,
            col_names_dict=col_names_dict,
            stype_encoder_dict=stype_encoder_dict,
        )

    def reset_parameters(self):
        self.encoder.reset_parameters()

    def forward(self, tf: Any) -> Tensor:
        """
        Args:
            tf: TensorFrame.

        Returns:
            Tensor of shape [num_samples, channels].
        """
        return self.encoder(tf)


class TabularGraphEncoder(Module):
    r"""
    A two-stage encoder for Tabular Graphs:

    1. Encode node tabular attributes with a :class:`TabularEncoder`.
    2. Encode graph structure with a PyG GNN model.

    This module is useful when each node is associated with a row-like
    tabular feature representation (stored as a :class:`torch_frame.TensorFrame`)
    and graph connectivity should be exploited afterwards.

    Args:
        tabular_encoder (torch.nn.Module): A tabular encoder using PyTorch Frame.
            It maps a single TensorFrame into embeddings.
        graph_encoder (torch.nn.Module): A graph encoder that consumes node embeddings
            and graph connectivity. Typical examples are:
            :class:`torch_geometric.nn.models.GCN`,
            :class:`torch_geometric.nn.models.GraphSAGE`, etc.
    """

    def __init__(
        self,
        tabular_encoder: Module,
        graph_encoder: Module,
    ):
        super().__init__()

        self.tabular_encoder = tabular_encoder
        self.graph_encoder = graph_encoder

    def reset_parameters(self):
        """Reset parameters of both the tabular encoder and the graph encoder."""
        if hasattr(self.tabular_encoder, "reset_parameters"):
            self.tabular_encoder.reset_parameters()
        if hasattr(self.graph_encoder, "reset_parameters"):
            self.graph_encoder.reset_parameters()

    def encode_tabular(self, tf: Any) -> Tensor:
        r"""
        Encode node tabular attributes into dense node embeddings.

        Args:
            tf (torch_frame.TensorFrame): Node features in TensorFrame format.

        Returns:
            Tensor: Node embeddings of shape [num_nodes, channels].
        """
        return self.tabular_encoder(tf)

    def _filter_supported_kwargs(self, module: Module,
                                 kwargs: Dict[str, Any]) -> Dict[str, Any]:
        r"""
        Filter keyword arguments according to the forward signature of a module.

        If the module forward accepts **kwargs, all arguments are kept.
        Otherwise, only explicitly declared parameters are retained.
        """
        signature = inspect.signature(module.forward)
        parameters = signature.parameters

        accepts_var_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in parameters.values()
        )
        if accepts_var_kwargs:
            return kwargs

        return {
            key: value
            for key, value in kwargs.items()
            if key in parameters
        }

    def encode_graph(
        self,
        x: Tensor,
        edge_index: Adj,
        edge_weight: OptTensor = None,
        edge_attr: OptTensor = None,
        batch: OptTensor = None,
        batch_size: Optional[int] = None,
        num_sampled_nodes_per_hop: Optional[List[int]] = None,
        num_sampled_edges_per_hop: Optional[List[int]] = None,
    ) -> Tensor:
        r"""
        Apply the graph encoder on node embeddings.

        Only arguments supported by the graph encoder's forward method
        will be passed through.
        """
        kwargs = {
            "x": x,
            "edge_index": edge_index,
            "edge_weight": edge_weight,
            "edge_attr": edge_attr,
            "batch": batch,
            "batch_size": batch_size,
            "num_sampled_nodes_per_hop": num_sampled_nodes_per_hop,
            "num_sampled_edges_per_hop": num_sampled_edges_per_hop,
        }

        kwargs = self._filter_supported_kwargs(self.graph_encoder, kwargs)

        return self.graph_encoder(**kwargs)

    def forward(
        self,
        x: torch_frame.TensorFrame,
        edge_index: Adj,
        edge_weight: OptTensor = None,
        edge_attr: OptTensor = None,
        batch: OptTensor = None,
        batch_size: Optional[int] = None,
        num_sampled_nodes_per_hop: Optional[List[int]] = None,
        num_sampled_edges_per_hop: Optional[List[int]] = None,
    ) -> Tensor:
        r"""
        Full forward pass:
        tabular node attributes -> tabular embeddings -> graph encoder output.
        """
        x = self.encode_tabular(x)

        x = self.encode_graph(
            x=x,
            edge_index=edge_index,
            edge_weight=edge_weight,
            edge_attr=edge_attr,
            batch=batch,
            batch_size=batch_size,
            num_sampled_nodes_per_hop=num_sampled_nodes_per_hop,
            num_sampled_edges_per_hop=num_sampled_edges_per_hop,
        )
        return x

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"tabular_encoder={self.tabular_encoder.__class__.__name__}, "
            f"graph_encoder={self.graph_encoder.__class__.__name__})"
        )
