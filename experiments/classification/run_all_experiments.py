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

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Import path setup utility
from setup_paths import setup_paths, verify_imports

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
        from rfecv_feature_selection import RFECVAnalyzer
        from utils.data_loader import MekaNetDataLoader
        
        # Load data
        print("📊 Loading dataset...")
        loader = MekaNetDataLoader('../../data/demo_data/classification.csv')
        data = loader.load_raw_data()
        data = loader.create_binary_labels(data)
        
        print(f"Dataset loaded: {data.shape[0]} samples, {data.shape[1]} features")
        
        # Initialize analyzer
        print("⚙️ Initializing RFECV analyzer...")
        analyzer = RFECVAnalyzer(random_seeds=[42, 43, 44, 45, 46])
        
        # Create results directory
        results_dir = Path('results')
        results_dir.mkdir(exist_ok=True)
        
        # Run correlation analysis
        print("🔍 Analyzing feature correlations...")
        corr_results = analyzer.analyze_feature_correlations(data, results_dir)
        
        # Create stabilized features
        print("📈 Creating stabilized feature groups...")
        stabilized_data = analyzer.create_stabilized_features(data)
        
        # Run RFECV analysis
        print("🎯 Running RFECV analysis...")
        binary_results = analyzer.run_stabilized_rfecv_analysis(stabilized_data, 'binary')
        multiclass_results = analyzer.run_stabilized_rfecv_analysis(stabilized_data, 'multiclass')
        
        # Generate report
        print("📝 Generating analysis report...")
        analyzer.generate_enhanced_report(results_dir)
        
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
        from utils.data_loader import MekaNetDataLoader
        
        # Load data
        print("📊 Loading multi-institutional dataset...")
        loader = MekaNetDataLoader('../../data/demo_data/classification.csv')
        data = loader.load_raw_data()
        
        # Initialize validator
        print("⚙️ Initializing institutional validator...")
        validator = InstitutionalValidator()
        
        # Run validation
        print("🏥 Running cross-institutional validation...")
        validation_results = validator.run_validation(data)
        
        # Generate report
        results_dir = Path('results')
        results_dir.mkdir(exist_ok=True)
        validator.generate_report(validation_results, results_dir)
        
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
        from utils.data_loader import MekaNetDataLoader
        
        # Load data
        print("📊 Loading dataset for comprehensive modeling...")
        loader = MekaNetDataLoader('../../data/demo_data/classification.csv')
        data = loader.load_raw_data()
        
        # Initialize comprehensive modeling
        print("⚙️ Initializing comprehensive modeling framework...")
        modeler = ComprehensiveModeling()
        
        # Run comprehensive analysis
        print("🔬 Running comprehensive modeling analysis...")
        modeling_results = modeler.run_analysis(data)
        
        # Generate report
        results_dir = Path('results')
        results_dir.mkdir(exist_ok=True)
        modeler.generate_report(modeling_results, results_dir)
        
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