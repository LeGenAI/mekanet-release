# MekaNet Classification Experiments

This directory contains the comprehensive classification experiments for the MekaNet framework, focusing on hematological disease diagnosis with cross-institutional validation.

## Overview

The classification experiments implement a comprehensive validation framework addressing key clinical and methodological concerns:
- **Feature Selection**: RFECV-based objective feature selection with stability analysis
- **Cross-Institutional Validation**: Robustness testing across multiple hospitals
- **Comprehensive Modeling**: Multi-tier performance and interpretability analysis

## Experiments

### 1. RFECV Feature Selection (`rfecv_feature_selection.py`)

**Purpose**: Address feature selection instability through correlation analysis and feature stabilization.

**Key Features**:
- Detection feature correlation analysis (threshold: |r| > 0.7)
- PCA-based feature group stabilization
- Stability metrics monitoring (Coefficient of Variation)
- Robust RFECV with multiple random seeds

**Usage**:
```bash
cd experiments/classification
python rfecv_feature_selection.py
```

**Expected Output**:
- Correlation analysis visualizations
- Feature stability metrics
- RFECV results for binary and multiclass tasks
- Comprehensive analysis report

### 2. Cross-Institutional Validation (`institutional_validation.py`)

**Purpose**: Complete cross-institutional validation demonstrating universal biological signatures.

**Key Features**:
- Stage 1: Direct generalization (B Hospital → S Hospital)
- Stage 2: Independent replication (External RFECV)
- Complete NaN handling for all algorithms
- Multi-algorithm robustness testing

**Usage**:
```bash
python institutional_validation.py
```

**Expected Output**:
- Generalization performance metrics
- Cross-institutional validation results
- Algorithm comparison across institutions
- Success rate analysis

### 3. Comprehensive Modeling (`comprehensive_modeling.py`)

**Purpose**: Multi-tier analysis addressing performance, interpretability, and robustness.

**Key Features**:
- Tier 1: Multiple algorithms with confidence intervals
- Tier 2: Interpretable models with feature importance
- Tier 3: Cross-dataset robustness analysis
- Statistical significance testing

**Usage**:
```bash
python comprehensive_modeling.py
```

**Expected Output**:
- Performance excellence metrics with 95% CI
- Clinical interpretability analysis
- Cross-dataset robustness validation
- Comprehensive multi-tier report

## Data Requirements

### Demo Dataset (`../../data/demo_data/classification.csv`)

The demo dataset contains 80 balanced samples:
- **Classes**: ET (20), PV (20), PMF (20), Lymphoma (20)
- **Sources**: Internal (40), External (40)
- **Features**: Clinical features + Detection features

### Required Columns

**Essential Columns**:
- `Label`: Disease class (ET, PV, PMF, Lymphoma)
- `data_source`: Institution (internal, external)
- `Image_ID`: Unique sample identifier

**Clinical Features**:
- `sex`: Patient gender
- `age`: Patient age (years)
- `Hb`: Hemoglobin level (g/dL)
- `WBC`: White blood cell count (×10³/μL)
- `PLT`: Platelet count (×10³/μL)
- `Reti%`: Reticulocyte percentage

**Detection Features**:
- Various morphological features extracted from megakaryocyte analysis
- Automatically detected from column names (not in essential/clinical lists)

## Reproducibility Guidelines

### Environment Setup

1. **Python Requirements**:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn scipy
   ```

2. **Random Seed Control**:
   All experiments use fixed random seeds for reproducibility:
   - Primary seed: 42
   - Multiple seeds: [42, 123, 456, 789, 1011] for stability analysis

### Expected Runtime

- **RFECV Feature Selection**: ~2-5 minutes
- **Institutional Validation**: ~1-3 minutes  
- **Comprehensive Modeling**: ~3-7 minutes

### Output Files

Each experiment generates:
- Results directory with analysis outputs
- Comprehensive text reports
- Visualization files (when applicable)
- Performance metrics with statistical validation

### Verification Metrics

**Key Performance Indicators**:
- Binary classification accuracy: >90% (internal), >85% (external)
- Multiclass classification accuracy: >75% (internal), >70% (external)
- Cross-institutional generalization: >80% relative performance
- Feature selection stability: CV < 0.5 for clinical features

## Algorithm Coverage

**Implemented Algorithms**:
- Decision Tree (interpretable)
- Random Forest (ensemble robustness)
- Gradient Boosting (high performance)
- Logistic Regression (clinical standard)
- SVM (non-linear boundaries)

## Statistical Validation

**Robust Evaluation**:
- Stratified K-fold cross-validation (k=5)
- Confidence intervals (95% CI using t-distribution)
- Multiple random seed validation
- Statistical significance testing

## Error Handling

**Comprehensive NaN Management**:
- Automatic missing value imputation (median)
- Feature availability checking
- Graceful algorithm failure handling
- Data type conversion with error catching

## Clinical Interpretability

**Interpretable Outputs**:
- Feature importance rankings
- Decision boundary visualizations
- Top contributing features identification
- Clinical significance assessment

## Troubleshooting

### Common Issues

1. **Missing Features**: 
   - Check demo dataset column names
   - Verify feature availability before analysis

2. **Algorithm Failures**:
   - Review NaN handling in data preparation
   - Check sample size sufficiency for each class

3. **Performance Issues**:
   - Reduce random seed count for faster execution
   - Use smaller feature sets for initial testing

### Support

For technical issues or questions about the experiments:
1. Check the output reports for detailed error messages
2. Verify data format matches requirements
3. Ensure all dependencies are properly installed

## Research Impact

### Key Contributions

1. **Objective Feature Selection**: RFECV eliminates arbitrary feature choices
2. **Cross-Institutional Validation**: First comprehensive external validation in MPN AI
3. **Clinical Interpretability**: Balance between performance and explainability
4. **Statistical Rigor**: Comprehensive confidence intervals and significance testing

### Clinical Significance

- **Perfect Binary Classification**: 100% accuracy distinguishing MPN from controls
- **Robust Subtype Classification**: Reliable ET, PV, PMF distinction
- **Cross-Hospital Applicability**: Validated performance across institutions
- **Clinical Integration**: Compatible with existing pathology workflows

## Citation

When using these experiments, please cite:

```bibtex
@article{won2024mekanet,
  title={MekaNet: A deep learning framework for megakaryocyte detection and myeloproliferative neoplasm classification with enhanced feature extraction},
  author={Won, Byung-Sun and Lee, Young-eun and Baek, Jae-Hyun and Hwang, Sang Mee and Kim, Jon-Lark},
  journal={Under Review},
  year={2024},
  note={Enhanced MPN classification through AI-powered morphological analysis achieving 100\% binary classification accuracy}
}
```

## Contact Information

**Primary Contact**: Jae-Hyun Baek
- **Email**: jhbaek@sogang.ac.kr
- **Affiliation**: Department of Mathematics, Sogang University
- **Lab**: CICAGO Lab (Computational Intelligence, Cryptography, Algorithms, Graph theory, Optimization)

**Clinical Contact**: Sang Mee Hwang
- **Email**: sangmee1@snu.ac.kr  
- **Affiliation**: Department of Pathology, Seoul National University Bundang Hospital

## License

This software is released under the MIT License. All patient data has been de-identified and the study was approved by the Seoul National University Bundang Hospital Institutional Review Board (B-2401-876-104).