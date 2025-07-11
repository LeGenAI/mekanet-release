# MekaNet Detection Experiments - Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying and running the MekaNet detection experiments framework. The framework implements TESSD (Tiling-Enhanced Semi-Supervised Detection) for megakaryocyte detection in bone marrow images with cross-institutional validation capabilities.

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone and navigate to the project
cd mekanet-release/

# Install dependencies
pip install -r requirements.txt

# Verify installation
python3 -c "import torch, sahi, ultralytics; print('Dependencies OK')"
```

### 2. Quick Test Run

```bash
# Run quick test configuration (dry-run mode)
python3 run_paper_reproduction.py --quick --dry-run

# Check configuration
python3 -c "
import yaml
with open('experiments/detection/configs/paper_reproduction_quick.yaml', 'r') as f:
    config = yaml.safe_load(f)
print(f'Quick config: {len(config)} experiments loaded')
"
```

## 📋 Prerequisites

### System Requirements
- **OS**: macOS 10.15+, Ubuntu 18.04+, Windows 10+
- **Python**: 3.8+ (tested with 3.12.0)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 10GB free space for models and results
- **GPU**: Optional but recommended for faster inference

### Dependencies
The framework requires the following key dependencies:

**Core ML/CV Libraries:**
- `torch>=1.11.0` - PyTorch for deep learning
- `ultralytics>=8.0.0` - YOLOv8 implementation
- `sahi>=0.11.0` - Slicing Aided Hyper Inference
- `opencv-python>=4.6.0` - Computer vision operations

**Data Processing:**
- `numpy>=1.21.0` - Numerical computations
- `pandas>=1.5.0` - Data manipulation
- `scikit-learn>=1.0.0` - Machine learning utilities
- `PyYAML>=6.0` - Configuration management

**Visualization:**
- `matplotlib>=3.5.0` - Plotting
- `seaborn>=0.11.0` - Statistical visualization

See `requirements.txt` for complete list.

## 🏗️ Installation

### Option 1: Standard Installation

```bash
# 1. Install core dependencies
pip install numpy pandas scipy scikit-learn matplotlib seaborn PyYAML

# 2. Install deep learning frameworks
pip install torch torchvision

# 3. Install computer vision libraries
pip install opencv-python ultralytics sahi

# 4. Install additional utilities
pip install tqdm joblib pathlib2 click
```

### Option 2: Complete Installation

```bash
# Install all dependencies from requirements.txt
pip install -r requirements.txt
```

### Option 3: Development Installation

```bash
# Install with development tools
pip install -r requirements.txt
pip install pytest pytest-cov black flake8
```

## 📁 Project Structure

```
mekanet-release/
├── experiments/detection/          # Detection experiment modules
│   ├── configs/                   # YAML configuration files
│   │   ├── paper_reproduction_full.yaml    # Complete experiment suite
│   │   └── paper_reproduction_quick.yaml   # Quick test configuration
│   ├── paper_reproduction_runner.py        # Main experiment runner
│   ├── benchmark_comparison.py             # Baseline method comparison
│   ├── results_aggregator.py              # Results processing
│   ├── external_validation_processor.py    # External dataset validation
│   └── tessd_framework.py                 # Core TESSD implementation
├── mekanet/                       # Core MekaNet modules
│   ├── models/                    # Model implementations
│   │   ├── yolo_sahi.py          # YOLOv8+SAHI integration
│   │   └── classifier.py         # MPN classification models
│   └── data/                      # Data processing utilities
├── run_paper_reproduction.py      # Main entry point
├── deploy_detection_experiments.py # Deployment script
├── requirements.txt               # Python dependencies
├── setup.py                      # Package setup
└── README.md                     # Project documentation
```

## 🎯 Usage Examples

### 1. Paper Reproduction (Complete)

```bash
# Run full experiment suite (6-8 experiments)
python3 run_paper_reproduction.py --config experiments/detection/configs/paper_reproduction_full.yaml

# Monitor progress
tail -f experiments/detection/results/*/experiment_summary.txt
```

### 2. Quick Testing

```bash
# Quick validation (4 experiments)
python3 run_paper_reproduction.py --quick

# Dry run (no actual processing)
python3 run_paper_reproduction.py --quick --dry-run
```

### 3. Custom Configuration

```bash
# Create custom config
cp experiments/detection/configs/paper_reproduction_quick.yaml my_config.yaml
# Edit my_config.yaml as needed

# Run custom experiments
python3 run_paper_reproduction.py --config my_config.yaml
```

### 4. Resume Interrupted Experiments

```bash
# Resume from checkpoint
python3 run_paper_reproduction.py --resume

# Resume specific experiment
python3 run_paper_reproduction.py --resume --experiment baseline_evaluation
```

## ⚙️ Configuration

### Experiment Configuration (YAML)

```yaml
baseline_evaluation:
  enabled: true
  type: "comprehensive_evaluation"
  models: ["yolov8n", "yolov8s", "yolov8m"]
  metrics: ["mAP", "precision", "recall", "f1"]
  confidence_thresholds: [0.15, 0.20, 0.25]
  datasets:
    training: "data/B_hospital_100_images"
    validation: "data/S_hospital_5_images"
  expected_runtime_minutes: 15
  output_dir: "results/baseline_evaluation"
```

### Performance Targets

**B Hospital (Training Data):**
- mAP@0.5: >0.85
- Precision: >0.88
- Recall: >0.82
- Processing time: <10 min/experiment

**S Hospital (Validation Data):**
- mAP@0.5: >0.80
- Generalization score: >0.85
- Performance drop: <15%
- Cross-institutional consistency: >0.75

## 🧪 Testing and Validation

### 1. Module Testing

```bash
# Test basic imports and structure
python3 -c "
import sys
sys.path.append('.')
import experiments.detection.paper_reproduction_runner as prr
print('✓ Modules import successfully')
"
```

### 2. Configuration Validation

```bash
# Validate YAML configs
python3 -c "
import yaml
with open('experiments/detection/configs/paper_reproduction_quick.yaml', 'r') as f:
    config = yaml.safe_load(f)
print(f'✓ Quick config: {len(config)} experiments')
print(f'✓ Experiment types: {[exp[\"type\"] for exp in config.values()]}')
"
```

### 3. Dry Run Testing

```bash
# Test without actual processing
python3 run_paper_reproduction.py --quick --dry-run --verbose

# Expected output:
# ✓ Configuration loaded
# ✓ Datasets validated
# ✓ Models initialized
# ✓ Experiment pipeline ready
```

## 🔧 Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Error: ModuleNotFoundError: No module named 'sahi'
pip install sahi>=0.11.0

# Error: No module named 'ultralytics'
pip install ultralytics>=8.0.0
```

**2. CUDA/GPU Issues**
```bash
# Check PyTorch CUDA availability
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Force CPU usage if needed
export CUDA_VISIBLE_DEVICES=""
```

**3. Memory Issues**
```bash
# Reduce batch size in configs
# Edit YAML files to use smaller models: yolov8n instead of yolov8m
# Use smaller image tile sizes: 416x416 instead of 640x640
```

**4. Configuration Errors**
```bash
# Validate YAML syntax
python3 -c "
import yaml
try:
    with open('experiments/detection/configs/paper_reproduction_quick.yaml', 'r') as f:
        yaml.safe_load(f)
    print('✓ YAML syntax OK')
except yaml.YAMLError as e:
    print(f'✗ YAML error: {e}')
"
```

### Performance Optimization

**For Faster Execution:**
1. Use GPU acceleration: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
2. Reduce image resolution in configs
3. Use smaller YOLOv8 models (yolov8n vs yolov8m)
4. Enable parallel processing: Set `num_workers: 4` in configs

**For Lower Memory Usage:**
1. Use `yolov8n` model
2. Reduce batch sizes
3. Use smaller tile sizes (416x416)
4. Process datasets sequentially

## 📊 Output and Results

### Result Directory Structure

```
experiments/detection/results/
├── results_YYYYMMDD_HHMMSS/          # Timestamped results
│   ├── experiment_summary.txt        # Overall summary
│   ├── baseline_evaluation/          # Individual experiment results
│   │   ├── metrics.json              # Numerical results
│   │   ├── visualizations/           # Plots and figures
│   │   └── detailed_report.txt       # Comprehensive analysis
│   └── statistical_analysis/         # Cross-experiment statistics
├── paper_reproduction_report.pdf     # Publication-ready report
└── aggregated_results.json          # Machine-readable summary
```

### Key Output Files

1. **experiment_summary.txt** - High-level results overview
2. **metrics.json** - Detailed performance metrics
3. **paper_reproduction_report.pdf** - Publication-ready figures and tables
4. **statistical_significance.csv** - Statistical analysis results

## 🔍 Monitoring and Logging

### Real-time Monitoring

```bash
# Monitor experiment progress
tail -f experiments/detection/results/*/experiment_summary.txt

# Check resource usage
top -p $(pgrep -f "paper_reproduction")
```

### Log Files

- **Experiment logs**: `experiments/detection/results/*/logs/`
- **Error logs**: `experiments/detection/results/*/errors/`
- **Performance logs**: `experiments/detection/results/*/performance/`

## 🚀 Production Deployment

### Docker Deployment (Optional)

```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python3", "run_paper_reproduction.py", "--config", "experiments/detection/configs/paper_reproduction_full.yaml"]
```

### Batch Processing

```bash
# Submit to SLURM cluster
sbatch --job-name=mekanet_detection \
       --time=04:00:00 \
       --mem=16G \
       --gres=gpu:1 \
       run_detection_experiments.slurm
```

## 📝 Next Steps

1. **Complete Integration Testing**
   ```bash
   # Install remaining dependencies
   pip install torch ultralytics sahi opencv-python
   
   # Run full test suite
   python3 run_paper_reproduction.py --quick
   ```

2. **Data Preparation**
   - Prepare B hospital training dataset (100 images)
   - Prepare S hospital validation dataset (5 images)
   - Ensure proper annotation format (YOLO format)

3. **Model Weights**
   - Download pre-trained YOLOv8 weights
   - Configure model paths in YAML files

4. **Production Deployment**
   - Set up monitoring and alerting
   - Configure automatic result archiving
   - Implement error recovery mechanisms

## 📞 Support

For issues and questions:
- Check troubleshooting section above
- Review experiment logs in `results/*/logs/`
- Validate configuration files
- Ensure all dependencies are installed

## 📄 License

See LICENSE file for licensing information.

---

**Last Updated**: January 2025  
**Version**: 1.0.0  
**Compatible with**: MekaNet Detection Framework v12+ 