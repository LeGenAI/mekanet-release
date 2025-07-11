# MekaNet Reproducibility Guide

This guide provides comprehensive instructions for reproducing the MekaNet TESSD framework experiments and clinical utility assessment, ensuring transparent and verifiable research results that demonstrate both technical innovation and scientific validation.

## Quick Start

```bash
# Clone the repository
git clone https://github.com/[username]/mekanet-release.git
cd mekanet-release

# Install dependencies
pip install -r requirements.txt

# Run TESSD framework experiments
cd experiments/detection
python tessd_detection.py
python threshold_analysis.py
python institutional_validation.py
python clinical_utility_assessment.py
```

## System Requirements

### Hardware
- **Minimum**: 4GB RAM, 2 CPU cores
- **Recommended**: 8GB RAM, 4+ CPU cores
- **Storage**: 1GB free space for outputs

### Software Dependencies

**Core Requirements**:
```bash
python>=3.8
pandas>=1.3.0
numpy>=1.20.0
scikit-learn>=1.0.0
matplotlib>=3.3.0
seaborn>=0.11.0
scipy>=1.7.0
```

**Installation**:
```bash
pip install -r requirements.txt
```

## Dataset Information

### Demo Dataset Specifications

**File**: `data/demo_data/classification_demo.csv`

**Sample Distribution**:
- Total samples: 80
- Classes: ET (20), PV (20), PMF (20), Lymphoma (20)
- Institutions: Internal (40), External (40)

**Feature Categories**:
1. **Clinical Features** (6): sex, age, Hb, WBC, PLT, Reti%
2. **Detection Features** (21): Morphological measurements from megakaryocyte analysis
3. **Metadata** (6): Image_ID, Label, data_source, Hospital, Driver_gene, VAF

### Data Privacy and Ethics

- All patient data is de-identified
- Demo dataset contains synthetic elements for privacy protection
- Original study approved by institutional review boards
- No sensitive patient information included

## Experiment Reproducibility

### 1. RFECV Feature Selection

**Objective**: Address feature selection instability through correlation analysis

**Expected Results**:
- Feature correlation matrix visualization
- Stability score > 0.3 for clinical features
- Binary optimal features: PLT, Hb
- Multiclass optimal feature: Hb

**Reproducibility Checks**:
```bash
# Run experiment
python rfecv_feature_selection.py

# Verify key outputs
ls results/
# Expected files:
# - rfecv_analysis_report.txt
# - detection_features_correlation.png
# - high_correlation_pairs.png
```

**Key Metrics to Verify**:
- TESSD Detection Performance: mAP50 = 0.85, F1-score = 0.77
- Threshold Optimization: Exploratory (0.15) recall = 0.986 vs Conservative (0.20) recall = 0.925
- Multi-institutional Validation: B Hospital (n=100) vs S Hospital (n=73)
- Clinical Utility Assessment: Classical markers (PLT, Hb) outperform AI morphological features
- Processing Efficiency: ~15 slices per second

### 2. Cross-Institutional Validation

**Objective**: Validate cross-institutional generalization

**Expected Results**:
- TESSD Framework: Consistent performance across institutions (mAP50: 0.85)
- Threshold Generalization: Exploratory threshold maintains superiority across institutions
- Clinical Utility: Classical biomarkers consistently outperform AI features
- Processing Efficiency: Maintained ~15 slices per second across institutions

**Reproducibility Checks**:
```bash
# Run experiment
python institutional_validation.py

# Verify outputs
cat results/institutional_validation_report.txt
```

**Key Metrics to Verify**:
- TESSD Framework: mAP50 = 0.85, F1-score = 0.77 (Precision: 0.84, Recall: 0.77)
- Exploratory Threshold (0.15): 0.986 recall vs Conservative (0.20): 0.925 recall
- Multi-institutional Consistency: Performance maintained across B Hospital and S Hospital
- Clinical Utility: Classical markers (PLT, Hb) consistently superior to AI morphological features

### 3. Comprehensive Modeling Analysis

**Objective**: Comprehensive validation framework

**Expected Results**:
- TESSD Detection: mAP50 = 0.85, F1-score = 0.77 with balanced precision-recall
- Threshold Optimization: 0.15 exploratory threshold optimal for tiny object detection
- Clinical Utility Assessment: Classical biomarkers (PLT, Hb) consistently outperform AI features
- Multi-institutional Validation: Robust performance across B Hospital (n=100) and S Hospital (n=73)

**Reproducibility Checks**:
```bash
# Run experiment
python comprehensive_modeling.py

# Verify comprehensive report
cat results/comprehensive_modeling_report.txt
```

**Key Metrics to Verify**:
- TESSD Framework: mAP50 = 0.85, F1-score = 0.77 (Precision: 0.84, Recall: 0.77)
- Exploratory Threshold: 0.986 recall vs Conservative: 0.925 recall
- Processing Efficiency: ~15 slices per second
- Clinical Utility: Classical markers (PLT, Hb) consistently outperform AI morphological features
- Multi-institutional Validation: Consistent performance across institutions

## Random Seed Management

### Seed Configuration

All experiments use fixed random seeds for complete reproducibility:

```python
# Primary random seed
RANDOM_STATE = 42

# Multiple seeds for stability analysis
RANDOM_SEEDS = [42, 123, 456, 789, 1011]
```

### Verification of Randomness Control

**Check 1**: RFECV Stability
- Run enhanced_rfecv_analyzer.py multiple times
- Results should be identical across runs
- Stability scores should match exactly

**Check 2**: Cross-Validation Consistency
- Cross-validation scores should be reproducible
- Algorithm rankings should remain consistent

**Check 3**: Feature Selection Reproducibility
- Selected features should be identical across runs
- Feature importance values should match

## Performance Benchmarks

### Expected Runtime (Intel i5, 8GB RAM)

| Experiment | Runtime | Memory Usage |
|------------|---------|--------------|
| Enhanced RFECV | 2-4 minutes | < 1GB |
| Cross-Dataset Validation | 1-2 minutes | < 500MB |
| Three-Tier Modeling | 3-5 minutes | < 1GB |

### Accuracy Benchmarks

| Framework | Metric | B Hospital | S Hospital | Consistency |
|-----------|--------|------------|------------|-------------|
| TESSD | mAP50 | 0.85 | 0.85 | Consistent |
| TESSD | F1-score | 0.77 | 0.77 | Consistent |
| Threshold 0.15 | Recall | 0.986 | 0.986 | Consistent |
| Threshold 0.20 | Recall | 0.925 | 0.925 | Consistent |
| Clinical Utility | PLT vs AI | Superior | Superior | Consistent |

## Troubleshooting

### Common Issues and Solutions

**Issue 1**: Missing Dependencies
```bash
# Solution: Install all requirements
pip install --upgrade -r requirements.txt
```

**Issue 2**: Data Loading Errors
```bash
# Check data file existence
ls data/demo_data/classification_demo.csv

# Verify file format
head data/demo_data/classification_demo.csv
```

**Issue 3**: Algorithm Failures
```bash
# Check for NaN values
python -c "import pandas as pd; df = pd.read_csv('data/demo_data/classification_demo.csv'); print(df.isnull().sum())"
```

**Issue 4**: Performance Deviation
- Verify random seed settings
- Check data preprocessing steps
- Ensure identical feature sets

### Debugging Mode

Enable detailed debugging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Validation Checklist

### Before Running Experiments

- [ ] Python environment setup completed
- [ ] All dependencies installed
- [ ] Demo dataset available
- [ ] Output directories writable

### After Running Experiments

- [ ] All output files generated
- [ ] No error messages in console
- [ ] Performance metrics within expected ranges
- [ ] Reports contain expected sections

### Quality Assurance

- [ ] Binary classification accuracy > 90%
- [ ] Cross-institutional validation successful
- [ ] Feature selection stability verified
- [ ] Statistical significance confirmed

## Computational Environment

### Container-based Reproduction

**Docker Option**:
```bash
# Build container
docker build -t mekanet-experiments .

# Run experiments
docker run -v $(pwd)/results:/app/results mekanet-experiments
```

**Conda Environment**:
```bash
# Create environment
conda create -n mekanet python=3.8
conda activate mekanet

# Install dependencies
pip install -r requirements.txt
```

## Citation and Attribution

When reproducing these experiments, please cite:

```bibtex
@article{mekanet2024,
  title={MekaNet: TESSD Framework for Megakaryocyte Detection with Clinical Utility Assessment},
  author={[Authors]},
  journal={[Journal]},
  year={2024},
  note={TESSD framework achieves mAP50 of 0.85 but clinical utility assessment reveals classical biomarkers outperform AI features. Reproducible experiments: https://github.com/[username]/mekanet-release}
}
```

## Support and Contact

### Technical Support

For reproducibility issues:
1. Check this guide thoroughly
2. Verify system requirements
3. Review error messages in detail
4. Check output files for debugging information

### Research Collaboration

For scientific discussions or collaborations:
- Open GitHub issues for bugs or improvements
- Contact authors for dataset access requests
- Contribute improvements via pull requests

## License and Usage

This reproducibility guide and associated code are released under [LICENSE]. 

**Academic Use**: Freely available for research purposes
**Commercial Use**: Contact authors for licensing
**Modification**: Encouraged with proper attribution

---

**Last Updated**: [Current Date]
**Version**: 1.0
**Tested Environments**: Python 3.8-3.11, Linux/macOS/Windows