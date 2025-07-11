"""
MekaNet Detection Experiments
TESSD (Tiling-Enhanced Semi-Supervised Detection) Framework

This module provides reproducible detection experiments for megakaryocyte detection
and validation across multiple institutions for the MekaNet paper.
"""

__version__ = "1.0.0"
__author__ = "MekaNet Research Team"

from .tessd_framework import TESSDFramework
from .detection_trainer import DetectionTrainer
from .detection_evaluator import DetectionEvaluator
from .institutional_validator import InstitutionalValidator

__all__ = [
    "TESSDFramework",
    "DetectionTrainer", 
    "DetectionEvaluator",
    "InstitutionalValidator"
] 