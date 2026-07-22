"""统一方法模块导出。"""

from .base_method import BaseMethod
from .ccassg_method import CCASSGMethod
from .dgi_method import DGIMethod
from .gcn_supervised_method import SupervisedGCNMethod
from .grace_method import GRACEMethod
from .grace_mrl_method import GRACEWithMRLMethod
from .grace_ML import GRACEWithMRLMutualLearningMethod
from .mcne_method import MCNEMethod
from .ssge_method import SSGEMethod
from .graphcl_method import GraphCLMethod
from .pagcl_method import PaGCLMethod
from .pagcl_mrl_method import PaGCLWithMRLMethod
from .pagcl_mcne_method import PaGCLMCNEMethod

__all__ = [
    "BaseMethod",
    "SupervisedGCNMethod",
    "DGIMethod",
    "CCASSGMethod",
    "GRACEMethod",
    "GRACEWithMRLMethod",
    "GRACEWithMRLMutualLearningMethod",
    "MCNEMethod",
    "SSGEMethod",
    "GraphCLMethod",
    "PaGCLMethod",
    "PaGCLWithMRLMethod",
    "PaGCLMCNEMethod",
]
