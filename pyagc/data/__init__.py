from .datasets import get_dataset
from .graphland import GraphLandDataset
try:
    from .tabular_graphland import GraphLandTensorFrameDataset, get_tabular_graphland_dataset
except Exception:
    GraphLandTensorFrameDataset = None
    get_tabular_graphland_dataset = None

__all__ = ['get_dataset', 'GraphLandDataset']
if GraphLandTensorFrameDataset is not None and get_tabular_graphland_dataset is not None:
    __all__.extend(['GraphLandTensorFrameDataset', 'get_tabular_graphland_dataset'])
