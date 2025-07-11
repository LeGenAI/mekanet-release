# MekaNet TESSD Framework - Deployment Guide

[![Paper](https://img.shields.io/badge/Paper-TESSD%20Framework-blue.svg)](https://github.com/LeGenAI/mekanet-release)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)]()
[![Detection](https://img.shields.io/badge/mAP50-0.85-brightgreen.svg)]()
[![Speed](https://img.shields.io/badge/Speed-15%20slices/sec-blue.svg)]()

## 🚀 Quick Start (One-Click Deployment)

### Option 1: Run Demo Experiments (Recommended for testing)
```bash
cd mekanet-release
python deploy_detection_experiments.py --experiment-type demo
```
**Runtime:** ~2-5 minutes  
**Purpose:** Quick validation with demo data

### Option 2: Run Optimal Threshold Analysis  
```bash
python deploy_detection_experiments.py --experiment-type threshold
```
**Runtime:** ~5-10 minutes  
**Purpose:** Exploratory threshold (0.15) vs conservative threshold (0.20) analysis

### Option 3: Run Complete Paper Reproduction
```bash
python deploy_detection_experiments.py --experiment-type complete
```
**Runtime:** ~10-15+ minutes  
**Purpose:** Full TESSD framework validation and multi-institutional testing

## 📋 Prerequisites

### Environment Requirements
- **Python:** 3.8+ 
- **GPU:** Recommended (CUDA-compatible)
- **RAM:** 8GB+ recommended
- **Storage:** 2GB+ free space

### Dependencies
The deployment script will automatically check and install:
```
torch>=1.13.0
ultralytics>=8.0.0
opencv-python>=4.7.0
sahi>=0.11.0
scikit-learn>=1.2.0
pandas>=1.5.0
matplotlib>=3.6.0
seaborn>=0.12.0
scipy>=1.9.0
PyYAML>=6.0
Pillow>=9.3.0
```

## 🎯 Experiment Types

### 1. Demo Detection (`demo`)
- **Purpose:** Quick functionality test
- **Data:** Uses sample images from `data/sample_images/`
- **Features:** 
  - Basic detection with TESSD framework
  - Morphological feature extraction
  - Simple performance metrics
- **Output:** `results/demo_results/`

### 2. Threshold Analysis (`threshold`)
- **Purpose:** Exploratory vs conservative threshold analysis
- **Comparison:** 0.15 (exploratory) vs 0.20 (conservative) confidence thresholds
- **Features:**
  - Detection performance comparison
  - Recall analysis (0.986 vs 0.925)
  - Precision-recall curves
  - Optimal threshold validation
- **Output:** `results/threshold_analysis/`

### 3. Complete Experiments (`complete`)
- **Purpose:** Full TESSD framework validation
- **Features:**
  - Multi-institutional validation (B Hospital n=100, S Hospital n=73)
  - TESSD performance metrics (mAP50: 0.85, F1: 0.77)
  - Clinical utility assessment
  - Processing speed evaluation (~15 slices/sec)
- **Output:** `results/paper_reproduction/`

### 4. Institutional Validation (`institutional`)
```bash
cd experiments/detection
python run_detection_experiments.py --config configs/institutional_validation.yaml
```
- **Purpose:** Multi-institutional TESSD validation
- **Features:**
  - B hospital (training, n=100) vs S hospital (external, n=73)
  - Threshold generalizability analysis
  - Clinical utility assessment across institutions
- **Output:** `results/institutional_validation/`

## 🔧 Manual Setup (Advanced Users)

### Step 1: Environment Verification
```bash
cd mekanet-release/experiments/detection
python setup_detection_experiments.py --comprehensive
```

### Step 2: Check Model Weights
```bash
cd ../../weights
python download_weights.py  # If weights are missing
```

### Step 3: Verify Demo Data
```bash
# Demo data should be in:
# - data/sample_images/
# - data/demo_data/
ls -la data/sample_images/
```

### Step 4: Run Specific Experiments
```bash
cd experiments/detection

# Demo only
python run_detection_experiments.py --config configs/paper_reproduction.yaml --demo-only

# Threshold analysis
python run_detection_experiments.py --config configs/threshold_analysis.yaml

# Complete pipeline
python run_detection_experiments.py --config configs/paper_reproduction.yaml --complete
```

## 📊 Expected Results

### Demo Results
```
results/demo_results/
├── demo_detection_results.csv          # Detection summary
├── demo_visualizations/                # Detection overlays
│   ├── SC2_cell_detections.jpg
│   ├── SC7_cell_detections.jpg
│   └── SP23_cell_detections.jpg
└── demo_report.txt                     # Performance summary
```

### Threshold Analysis Results
```
results/threshold_analysis/
├── threshold_comparison.csv            # Threshold performance
├── optimal_threshold_report.txt        # Recommendation
├── precision_recall_curves.png         # PR curves
└── detection_count_analysis.png        # Count vs threshold
```

### Complete Experiment Results
```
results/paper_reproduction/
├── comprehensive_report.txt            # Full experimental report
├── evaluation_metrics/
│   ├── detection_performance.csv       # mAP, precision, recall
│   ├── morphological_features.csv      # 21 extracted features
│   └── statistical_significance.csv    # Statistical tests
├── visualizations/
│   ├── detection_examples.png          # Sample detections
│   ├── feature_distributions.png       # Morphological analysis
│   └── institutional_comparison.png    # B vs S hospital
└── institutional_validation/
    ├── cross_hospital_performance.csv  # Performance comparison
    └── bias_analysis_report.txt        # Institutional bias
```

## 🔬 Technical Details

### TESSD Framework Architecture
- **Base Model:** YOLOv8 with tiling enhancement
- **Tiling Strategy:** 640×640 pixels, 20% overlap
- **Semi-supervised:** Exploratory pseudo-labeling
- **Processing Speed:** ~15 slices per second
- **Optimal Threshold:** 0.15 (exploratory) vs 0.20 (conservative)

### Validated Performance Metrics
| Metric | Value | Validation |
|--------|-------|------------|
| mAP@0.5 | 0.85 | Multi-institutional |
| F1-Score | 0.77 | Precision: 0.84, Recall: 0.77 |
| Exploratory Recall | 0.986 | Threshold: 0.15 |
| Conservative Recall | 0.925 | Threshold: 0.20 |

### Clinical Utility Assessment
1. **Technical Innovation:** TESSD achieves superior detection performance (mAP50: 0.85, F1: 0.77)
2. **Clinical Validation:** Classical markers (PLT, Hb) consistently outperform AI morphological features
3. **Key Finding:** While TESSD is technically superior, morphological features don't add clinical value beyond standard biomarkers
4. **Multi-institutional:** Consistent performance across B Hospital (n=100) and S Hospital (n=73)
5. **Processing Efficiency:** Real-time analysis at ~15 slices per second
6. **Scientific Discovery:** Dual contribution - technical innovation + rigorous clinical validation

## 🐛 Troubleshooting

### Common Issues

#### 1. Model Weights Not Found
```bash
# Error: epoch60.pt not found
cd weights
python download_weights.py
```

#### 2. Demo Data Missing
```bash
# Check for demo images
find . -name "*.jpg" | grep -E "(SC2|SC7|SP23)"

# If missing, check alternative locations:
# - data/sample_images/
# - experiments/detection/demo_data/
```

#### 3. Memory Issues
```bash
# Reduce tile size in configs
# Edit configs/paper_reproduction.yaml:
# tile_size: 512  # Instead of 640
```

#### 4. CUDA/GPU Issues
```bash
# Force CPU mode
export CUDA_VISIBLE_DEVICES=""
python deploy_detection_experiments.py --experiment-type demo
```

#### 5. Permission Errors
```bash
# Make scripts executable
chmod +x deploy_detection_experiments.py
chmod +x experiments/detection/*.py
```

### Environment Issues

#### Python Version
```bash
# Check Python version (requires 3.8+)
python --version

# If using conda:
conda create -n mekanet python=3.9
conda activate mekanet
```

#### Dependencies
```bash
# Install missing packages manually
pip install torch ultralytics opencv-python sahi scikit-learn
pip install pandas matplotlib seaborn scipy PyYAML Pillow
```

### Validation Issues

#### No Detections Found
- Check confidence threshold (try lowering to 0.1)
- Verify image quality and format
- Ensure model weights are correct version

#### Performance Below Targets
- Verify using correct demo data
- Check for data preprocessing issues
- Confirm model weights are from epoch 60

## 📝 Configuration Customization

### Custom Confidence Thresholds
```yaml
# Edit configs/threshold_analysis.yaml
detection:
  confidence_thresholds: [0.1, 0.15, 0.20, 0.25, 0.30]
  # Key thresholds from paper:
  # 0.15: Exploratory threshold (0.986 recall)
  # 0.20: Conservative threshold (0.925 recall)
```

### Custom Evaluation Metrics
```yaml
# Edit configs/paper_reproduction.yaml
evaluation:
  primary_metrics:
    - "mAP@0.5"
    - "mAP@0.75"
    - "mAP@0.95"  # Add stricter IoU
    - "precision"
    - "recall"
```

### Custom Output Directory
```yaml
# Edit any config file
output:
  base_dir: "./custom_results"  # Change output location
```

## 🔗 Integration with Paper

### Key Research Findings
The experiments validate the paper's dual contribution:
1. **Technical Innovation:** TESSD framework achieves mAP50 of 0.85 with superior detection performance and optimal threshold discovery (0.15 exploratory vs 0.20 conservative)
2. **Scientific Discovery:** Clinical utility assessment reveals classical biomarkers (PLT, Hb) consistently outperform AI morphological features across institutions
3. **Clinical Impact:** While TESSD is technically superior for detection, morphological features don't add clinical value beyond standard biomarkers for MPN diagnosis
4. **Multi-institutional Validation:** Robust performance across B Hospital (n=100) and S Hospital (n=73)
5. **Processing Efficiency:** Real-time capability at ~15 slices per second

### Figures for Paper
The experiments generate publication-ready figures:
- `tessd_detection_examples.png` → TESSD framework sample detections
- `threshold_analysis.png` → Exploratory (0.15) vs Conservative (0.20) threshold comparison
- `institutional_validation.png` → Multi-institutional validation results
- `clinical_utility_assessment.png` → Classical markers vs AI features comparison

### Tables for Paper
CSV files ready for paper tables:
- `tessd_performance.csv` → TESSD detection metrics (mAP50: 0.85, F1: 0.77, Precision: 0.84, Recall: 0.77)
- `threshold_analysis.csv` → Confidence threshold optimization results (0.15 exploratory: 0.986 recall vs 0.20 conservative: 0.925 recall)
- `institutional_validation.csv` → Multi-institutional validation (B Hospital: n=100, S Hospital: n=73)
- `clinical_utility_assessment.csv` → Classical biomarkers (PLT, Hb) vs AI morphological features comparison
- `processing_efficiency.csv` → Real-time performance metrics (~15 slices per second)

### Reproducibility
All experiments use fixed random seeds and deterministic settings:
```python
# Set in all config files
random_seed: 42
deterministic: true
pytorch_deterministic: true
```

## 📧 Support

For issues or questions:
1. Check this troubleshooting guide
2. Review experiment logs in `results/*/logs/`
3. Verify configuration files in `configs/`
4. Check the main README.md for general setup

## 🎉 Success Indicators

Your deployment is successful when you see:

✅ **Environment Setup Complete**
- All dependencies installed
- Model weights downloaded
- Demo data accessible

✅ **Experiments Running**
- Real-time detection progress
- No error messages in logs
- Results being generated

✅ **Results Generated**
- CSV files with metrics
- PNG visualizations
- Comprehensive reports

Example successful output:
```
🎉 TESSD Deployment completed successfully in 4.2 minutes!
📁 Results saved to: experiments/detection/results/demo_results
📊 TESSD Results Summary:
   - 3 images processed
   - mAP@0.5: 0.85 (validated)
   - F1-Score: 0.77 (Precision: 0.84, Recall: 0.77)
   - Processing speed: ~15 slices per second
   - Optimal threshold: 0.15 (exploratory) vs 0.20 (conservative)
``` 