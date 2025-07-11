# MekaNet Detection Experiments

[![Paper](https://img.shields.io/badge/Paper-TESSD%20Framework-blue.svg)](https://github.com/LeGenAI/mekanet-release)
[![Reproducible](https://img.shields.io/badge/Reproducible-Research-green.svg)](./REPRODUCIBILITY.md)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)]()

## 🎯 Overview

This module implements the **TESSD (Tiling-Enhanced Semi-Supervised Detection)** framework for megakaryocyte detection in bone marrow histopathology images, as described in the MekaNet paper.

### 🔬 Key Components

1. **TESSD Framework** - Core architecture for tiling-enhanced detection
2. **Semi-supervised Training** - Self-training on partially labeled data
3. **Cross-institutional Validation** - B hospital (training) → S hospital (validation)
4. **Reproducible Experiments** - Complete pipeline for paper results

## 🚀 Quick Start

### Basic Detection Inference
```bash
cd experiments/detection
python inference_demo.py --image_path /path/to/image.jpg --model_path ../../weights/epoch60.pt
```

### Reproduce Paper Results
```bash
# Complete detection experiments
python run_detection_experiments.py --config configs/paper_reproduction.yaml

# Institutional validation
python institutional_validation.py --b_hospital_data /path/to/b_data --s_hospital_data /path/to/s_data
```

## 📊 Experimental Design

### 🏥 Datasets
- **B Hospital**: 100 partially labeled images (training)
- **S Hospital**: 5 fully labeled images (external validation)
- **Demo Data**: Representative cases for testing

### 🎯 Metrics
- **mAP (mean Average Precision)**: Primary detection metric
- **Precision/Recall**: At multiple IoU thresholds
- **F1-Score**: Harmonic mean of precision/recall
- **Confidence Analysis**: Threshold optimization (0.15 vs 0.20)

### 🔧 TESSD Configuration
- **Tiling Strategy**: 640×640 pixels with 20% overlap
- **Confidence Thresholds**: 0.15, 0.20 (paper comparison)
- **Semi-supervised Iterations**: Progressive pseudo-labeling
- **Cross-validation**: 5-fold for robustness

## 📁 Structure

```
detection/
├── README.md                     # This file
├── configs/                      # Experiment configurations
│   ├── paper_reproduction.yaml   # Settings for paper results
│   ├── threshold_analysis.yaml   # Confidence threshold experiments
│   └── institutional_val.yaml    # Cross-hospital validation
├── tessd_framework.py            # Core TESSD implementation
├── detection_trainer.py          # Semi-supervised training
├── detection_evaluator.py        # Comprehensive evaluation
├── institutional_validator.py    # Cross-hospital validation
├── inference_demo.py             # Single image inference
├── run_detection_experiments.py  # Complete experiment pipeline
└── results/                      # Generated outputs
    ├── training_logs/
    ├── evaluation_metrics/
    └── visualizations/
```

## 🔬 TESSD Framework Details

### Architecture
```python
# Example usage
from experiments.detection import TESSDFramework

# Initialize TESSD
tessd = TESSDFramework(
    model_path="../../weights/epoch60.pt",
    confidence_threshold=0.20,
    tile_size=640,
    overlap_ratio=0.2
)

# Perform detection
detections = tessd.detect(image, use_tiling=True)

# Extract features for classification
features = tessd.extract_morphological_features(detections, image_shape)
```

### Semi-supervised Learning
1. **Initial Training**: Limited labeled data from B hospital
2. **Pseudo-labeling**: High-confidence predictions on unlabeled data
3. **Iterative Refinement**: Progressive model improvement
4. **Validation**: External testing on S hospital data

## 📈 Performance Targets

Based on notebook experiments and paper requirements:

| Metric | B Hospital | S Hospital | Target |
|--------|------------|------------|---------|
| mAP@0.5 | >0.85 | >0.80 | Maintain cross-institutional performance |
| Precision | >0.90 | >0.85 | High precision for clinical use |
| Recall | >0.80 | >0.75 | Detect majority of megakaryocytes |
| F1-Score | >0.85 | >0.80 | Balanced performance |

## 🔄 Reproducibility

### Environment Setup
```bash
# Install dependencies
pip install -r ../../requirements.txt

# Verify setup
python -c "from experiments.detection import TESSDFramework; print('Setup OK')"
```

### Exact Reproduction
```bash
# Set random seeds for reproducibility
export PYTHONHASHSEED=42

# Run complete pipeline
python run_detection_experiments.py --seed 42 --deterministic True
```

## 📚 References

- Original SAHI paper: [Slicing Aided Hyper Inference](https://github.com/obss/sahi)
- YOLOv8: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- Semi-supervised Learning: Self-training methodology 