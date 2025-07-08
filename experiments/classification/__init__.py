"""
MekaNet Classification Experiments

This module contains comprehensive classification experiments for hematological disease diagnosis
with cross-institutional validation framework.

Experiments:
- Enhanced RFECV Feature Selection
- Cross-Dataset Validation  
- Three-Tier Modeling Analysis

For detailed usage instructions, see the README.md file.
"""

from .rfecv_feature_selection import RFECVFeatureSelector
from .institutional_validation import InstitutionalValidator  
from .comprehensive_modeling import ComprehensiveModeling

__version__ = "1.0.0"
__author__ = "MekaNet Research Team"

__all__ = [
    "RFECVFeatureSelector",
    "InstitutionalValidator", 
    "ComprehensiveModeling"
]