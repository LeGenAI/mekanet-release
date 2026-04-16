# Classification Validation Module

This directory contains the manuscript-facing classification analyses for MekaNet.

## Scope

The classification code covers:

- feature extraction quality checks
- RFECV-style feature selection
- binary control-vs-MPN modeling
- multi-class subtype modeling
- cross-dataset validation routines

## Public Data Status

The current public release does **not** include the classification table used in the manuscript.

Not publicly available in this repository:

- The control cases referenced in the manuscript classification section
- The complete patient-level dataset and full master cohort
- The public-ready `data/demo_data/classification.csv` expected by experiment runners
- Any feature-engineered CSV export derived from private patient data

## Scripts

- `rfecv_feature_selection.py`: feature stability and correlation analysis
- `institutional_validation.py`: stage-1/2 generalization checks
- `comprehensive_modeling.py`: multi-tier benchmark reporting
- `run_all_experiments.py`: wrapper over the three analyses
- `setup_paths.py`: minimal path/bootstrap helper

## Expected Input Schema

The experiment runners expect a CSV with at least:

- `Image_ID`
- `Label`
- `data_source` for cross-institution validation
- clinical columns such as `sex`, `age`, `Hb`, `WBC`, `PLT`, `Reti%`
- morphological columns such as `Avg_Size`, `Std_Size`, `Num_Megakaryocytes`, `Avg_NND`, `Avg_Local_Density`, `Num_Clusters`

## Running

```bash
cd experiments/classification
python run_all_experiments.py
```

If `../../data/demo_data/classification.csv` is absent, the runner now fails explicitly and does not pretend to use a bundled demo dataset.

## Affiliation Note

Reviewer-facing documentation should use `Department of Laboratory Medicine` for the clinical affiliation, consistent with the private manuscript.

## Cleanup Note

Standalone demo scripts and redundant per-folder dependency files were removed so that this directory reflects only the reproducibility-critical classification analyses.
