"""
MekaNet Classification Experiments - Complete Pipeline
=====================================================

Run all classification experiments in sequence for comprehensive validation
and reproducibility testing. This script provides a unified entry point for
all MekaNet classification analyses.

Author: MekaNet Research Team
License: MIT
"""

import sys
import warnings
import time
from pathlib import Path
import pandas as pd

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Import path setup utility
from setup_paths import setup_paths, verify_imports


def _load_classification_data():
    """Load the shared classification CSV once per experiment."""
    data_path = Path("../../data/demo_data/classification.csv")
    if not data_path.exists():
        raise FileNotFoundError(
            f"Demo dataset not found at {data_path}. Add the CSV or update the path before running experiments."
        )

    data = pd.read_csv(data_path)
    return data_path, data

def print_section(title):
    """Print formatted section header"""
    print("\n" + "="*80)
    print(f"🧪 {title}")
    print("="*80)

def run_rfecv_experiment():
    """
    Run RFECV Feature Selection Analysis
    
    This experiment performs Recursive Feature Elimination with Cross-Validation
    to identify optimal feature subsets for classification tasks.
    
    Returns:
        bool: True if experiment completed successfully
    """
    print_section("EXPERIMENT 1: RFECV Feature Selection Analysis")
    
    try:
        # Import required modules
        sys.path.insert(0, str(Path(__file__).parent))
        from rfecv_feature_selection import RFECVFeatureSelector
        
        # Load data
        print("📊 Loading dataset...")
        _, data = _load_classification_data()
        
        print(f"Dataset loaded: {data.shape[0]} samples, {data.shape[1]} features")
        
        # Initialize analyzer
        print("⚙️ Initializing RFECV analyzer...")
        analyzer = RFECVFeatureSelector(random_seeds=[42, 43, 44, 45, 46])
        
        # Create results directory
        results_dir = Path('results')
        results_dir.mkdir(exist_ok=True)
        
        # Run correlation analysis
        print("🔍 Analyzing feature correlations...")
        analyzer.analyze_feature_correlations(data, save_dir=results_dir)

        clinical_features = ['sex', 'age', 'Hb', 'WBC', 'PLT', 'Reti%']
        available_clinical = [f for f in clinical_features if f in data.columns]
        if not available_clinical:
            raise ValueError("No expected clinical features were found in the classification dataset")

        binary_df = data[data['Label'].isin(['ET', 'PV', 'PMF', 'Lymphoma'])].copy()
        binary_df['binary_target'] = binary_df['Label'].apply(lambda x: 0 if x == 'Lymphoma' else 1)
        multiclass_df = data[data['Label'].isin(['ET', 'PV', 'PMF'])].copy()
        multiclass_labels = sorted(multiclass_df['Label'].dropna().unique())
        label_to_idx = {label: idx for idx, label in enumerate(multiclass_labels)}
        multiclass_df['multiclass_target'] = multiclass_df['Label'].map(label_to_idx)

        print("🎯 Running RFECV analysis...")
        analyzer.run_enhanced_rfecv(binary_df[available_clinical], binary_df['binary_target'], 'binary_clinical', 'binary')
        analyzer.run_enhanced_rfecv(
            multiclass_df[available_clinical],
            multiclass_df['multiclass_target'],
            'multiclass_clinical',
            'multiclass'
        )
        
        # Generate report
        print("📝 Generating analysis report...")
        analyzer.generate_report(results_dir / "enhanced_rfecv_analysis_report.txt")
        
        print("✅ RFECV experiment completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ RFECV experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_institutional_validation():
    """
    Run Cross-Institutional Validation Framework
    
    This experiment validates model performance across different institutions
    to ensure generalizability and robustness.
    
    Returns:
        bool: True if experiment completed successfully
    """
    print_section("EXPERIMENT 2: Cross-Institutional Validation")
    
    try:
        # Import required modules
        from institutional_validation import InstitutionalValidator
        
        # Load data
        print("📊 Loading multi-institutional dataset...")
        _, data = _load_classification_data()
        if 'data_source' not in data.columns:
            raise ValueError("classification.csv is missing the 'data_source' column required for institutional validation")
        df_internal = data[data['data_source'] == 'internal'].copy()
        df_external = data[data['data_source'] == 'external'].copy()
        
        # Initialize validator
        print("⚙️ Initializing institutional validator...")
        validator = InstitutionalValidator()

        clinical_features = ['sex', 'age', 'Hb', 'WBC', 'PLT', 'Reti%']
        available_features = [f for f in clinical_features if f in data.columns]
        if not available_features:
            raise ValueError("No expected clinical features were found in the classification dataset")
        
        # Run validation
        print("🏥 Running cross-institutional validation...")
        validator.run_stage1_validation(df_internal, df_external, available_features, 'binary')
        validator.run_stage1_validation(df_internal, df_external, available_features, 'multiclass')
        validator.run_stage2_validation(df_external, available_features, 'binary')
        validator.run_stage2_validation(df_external, available_features, 'multiclass')
        
        # Generate report
        results_dir = Path('results')
        results_dir.mkdir(exist_ok=True)
        validator.generate_report(results_dir / "cross_dataset_validation_report.txt")
        
        print("✅ Institutional validation completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Institutional validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_comprehensive_modeling():
    """
    Run Comprehensive Modeling Analysis
    
    This experiment performs comprehensive model evaluation including
    multiple algorithms, metrics, and validation strategies.
    
    Returns:
        bool: True if experiment completed successfully
    """
    print_section("EXPERIMENT 3: Comprehensive Modeling Analysis")
    
    try:
        # Import required modules
        from comprehensive_modeling import ComprehensiveModeling
        
        # Load data
        print("📊 Loading dataset for comprehensive modeling...")
        _, data = _load_classification_data()
        
        # Initialize comprehensive modeling
        print("⚙️ Initializing comprehensive modeling framework...")
        modeler = ComprehensiveModeling()

        clinical_features = ['sex', 'age', 'Hb', 'WBC', 'PLT', 'Reti%']
        available_clinical = [f for f in clinical_features if f in data.columns]
        if not available_clinical:
            raise ValueError("No expected clinical features were found in the classification dataset")

        feature_sets = {
            'binary_optimal': [f for f in ['PLT', 'Hb'] if f in data.columns],
            'multiclass_optimal': [f for f in ['Hb'] if f in data.columns],
            'clinical_comprehensive': available_clinical,
        }
        feature_sets = {name: features for name, features in feature_sets.items() if features}
        
        # Run comprehensive analysis
        print("🔬 Running comprehensive modeling analysis...")
        modeler.tier1_performance_excellence(data, feature_sets)
        modeler.tier2_clinical_interpretability(data, available_clinical)
        modeler.tier3_cross_dataset_robustness(data, available_clinical)
        
        # Generate report
        results_dir = Path('results')
        results_dir.mkdir(exist_ok=True)
        modeler.generate_comprehensive_report(results_dir / "three_tier_modeling_report.txt")
        
        print("✅ Comprehensive modeling completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Comprehensive modeling failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """
    Main execution function for MekaNet classification experiments.
    
    This function orchestrates the complete experimental pipeline including
    environment setup, data verification, and sequential experiment execution.
    
    Returns:
        bool: True if all experiments completed successfully
    """
    print_section("MEKANET CLASSIFICATION EXPERIMENTS - COMPLETE PIPELINE")
    
    # Environment setup and verification
    print("⚙️ Setting up environment...")
    if not setup_paths():
        print("❌ Path setup failed!")
        return False
    
    if not verify_imports():
        print("❌ Import verification failed!")
        print("Please install required dependencies: pip install -r requirements.txt")
        return False
    
    # Verify data availability
    data_path = Path("../../data/demo_data/classification.csv")
    if not data_path.exists():
        print(f"❌ Demo dataset not found at {data_path}")
        print("Please ensure the demo dataset is available before running experiments.")
        print("Alternatively, update the data path in the experiment functions.")
        return False
    
    # Create results directory
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    print(f"\n✅ Data verified: {data_path}")
    print(f"📁 Results directory: {results_dir.absolute()}")
    
    # Define experiments to run
    experiments = [
        ("RFECV Feature Selection", run_rfecv_experiment),
        ("Cross-Institutional Validation", run_institutional_validation),
        ("Comprehensive Modeling", run_comprehensive_modeling)
    ]
    
    # Track experiment results
    results = {}
    total_start_time = time.time()
    
    # Run each experiment
    for exp_name, exp_func in experiments:
        print(f"\n🎬 Starting {exp_name}...")
        start_time = time.time()
        
        success = exp_func()
        
        end_time = time.time()
        runtime = end_time - start_time
        
        results[exp_name] = success
        
        if success:
            print(f"⏱️ {exp_name} completed in {runtime:.1f} seconds")
        else:
            print(f"⚠️ {exp_name} failed after {runtime:.1f} seconds")
            print("Continuing with remaining experiments...")
    
    # Final summary
    total_end_time = time.time()
    total_runtime = total_end_time - total_start_time
    
    print_section("EXPERIMENT PIPELINE SUMMARY")
    
    successful_experiments = sum(results.values())
    total_experiments = len(experiments)
    
    print(f"\n📊 Experiments completed: {successful_experiments}/{total_experiments}")
    print(f"⏱️ Total runtime: {total_runtime:.1f} seconds")
    print(f"📁 Results saved to: {results_dir.absolute()}")
    
    print("\n📋 Individual experiment results:")
    for exp_name, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"  {exp_name:<35} {status}")
    
    # Verify output files
    print("\n📄 Generated output files:")
    if results_dir.exists():
        output_files = list(results_dir.glob("*"))
        if output_files:
            for file_path in sorted(output_files):
                print(f"  📄 {file_path.name}")
        else:
            print("  No output files found in results directory")
    
    # Final recommendations
    print("\n🚀 Next steps:")
    if successful_experiments == total_experiments:
        print("  🎉 ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
        print("  📖 1. Review generated reports in results/ directory")
        print("  📊 2. Examine visualizations for insights")
        print("  📈 3. Compare metrics with expected benchmarks")
        print("  📝 4. Use results for manuscript preparation")
        print("  🔬 5. Consider additional validation experiments")
    else:
        print("  ⚠️ Some experiments failed. Please:")
        print("  🔍 1. Check error messages for failed experiments")
        print("  📋 2. Verify data format and dependencies")
        print("  🔄 3. Re-run failed experiments individually")
        print("  💬 4. Contact support if issues persist")
        print("  📖 5. Check README.md for troubleshooting guide")
    
    return successful_experiments == total_experiments

if __name__ == "__main__":
    print("MekaNet Classification Experiments Runner")
    print("========================================")
    print("Starting comprehensive experimental pipeline...\n")
    
    # Run all experiments
    success = main()
    
    # Exit with appropriate code
    if success:
        print("\n🎉 All experiments completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Some experiments failed. Check logs above for details.")
        sys.exit(1)
