"""
MekaNet Detection Experiments
TESSD (Tiling-Enhanced Semi-Supervised Detection) Framework

This module provides reproducible detection experiments for megakaryocyte detection
and validation across multiple institutions for the MekaNet paper.
"""

__version__ = "1.0.0"
__author__ = "MekaNet Research Team"

__all__ = []

try:
    from .detection_evaluator import DetectionEvaluator
except ModuleNotFoundError:
    DetectionEvaluator = None
else:
    __all__.append("DetectionEvaluator")

try:
    from .institutional_validator import InstitutionalValidator
except ModuleNotFoundError:
    InstitutionalValidator = None
else:
    __all__.append("InstitutionalValidator")

try:
    from .tessd_framework import TESSDFramework
except ModuleNotFoundError:
    TESSDFramework = None
else:
    __all__.append("TESSDFramework")

try:
    from .semi_supervised_trainer import SemiSupervisedTrainer
except ModuleNotFoundError:
    SemiSupervisedTrainer = None
    DetectionTrainer = None
else:
    DetectionTrainer = SemiSupervisedTrainer
    __all__.extend(["SemiSupervisedTrainer", "DetectionTrainer"])
