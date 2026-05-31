"""统一方法模块导出。"""

from .base_method import BaseMethod
from .ccassg_method import CCASSGMethod
from .ccassg_mrl_method import CCASSGWithMRLMethod
from .dgi_method import DGIMethod
from .dgi_mrl_method import DGIWithMRLMethod
from .gcn_supervised_method import SupervisedGCNMethod
from .grace_method import GRACEMethod
from .grace_mrl_method import GRACEWithMRLMethod
from .mrl_module import MRLFusionModule

__all__ = [
    "BaseMethod",
    "MRLFusionModule",
    "SupervisedGCNMethod",
    "DGIMethod",
    "CCASSGMethod",
    "GRACEMethod",
    "DGIWithMRLMethod",
    "CCASSGWithMRLMethod",
    "GRACEWithMRLMethod",
]
