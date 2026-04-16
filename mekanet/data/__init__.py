"""
Data processing module for MekaNet

Contains utilities for:
- Feature extraction from detected megakaryocytes
- Data preprocessing and augmentation
- Dataset classes for training and evaluation
"""

from .feature_extraction import FeatureExtractor, extract_morphological_features

__all__ = [
    "FeatureExtractor", 
    "extract_morphological_features",
]

try:
    from .preprocessing import preprocess_image, normalize_features
except ModuleNotFoundError:
    preprocess_image = None
    normalize_features = None
else:
    __all__.extend(["preprocess_image", "normalize_features"])

try:
    from .dataset import MegakaryocyteDataset
except ModuleNotFoundError:
    MegakaryocyteDataset = None
else:
    __all__.append("MegakaryocyteDataset")
