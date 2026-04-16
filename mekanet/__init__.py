"""
MekaNet: A deep learning framework for megakaryocyte detection 
and myeloproliferative neoplasm classification with enhanced feature extraction

This package provides tools for:
- Megakaryocyte detection using YOLO and SAHI
- Morphological feature extraction from detected cells
- Binary and multi-class classification for MPN diagnosis
"""

__version__ = "1.0.0"
__author__ = "Byung-Sun Won, Young-eun Lee, Jae-Hyun Baek, Sang Mee Hwang, Jon-Lark Kim"

from .models import MPNClassifier
from .data import FeatureExtractor, extract_morphological_features
from .utils import visualize_detections, calculate_metrics

__all__ = [
    "MPNClassifier",
    "FeatureExtractor",
    "extract_morphological_features",
    "visualize_detections",
    "calculate_metrics"
]

try:
    from .models import YoloSahiDetector
except ImportError:
    YoloSahiDetector = None
else:
    __all__.append("YoloSahiDetector")

try:
    from .models import CellularityEstimator
except ImportError:
    CellularityEstimator = None
else:
    __all__.append("CellularityEstimator")

try:
    from .data import MegakaryocyteDataset
except ImportError:
    MegakaryocyteDataset = None
else:
    __all__.append("MegakaryocyteDataset")
