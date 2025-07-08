"""
MekaNet Classification Experiments - Complete Pipeline
Run all classification experiments in sequence for comprehensive validation
"""

import os
import sys
import time
from pathlib import Path
import subprocess

def print_section(title):
    """Print section header"""
    print("\n" + "="*80)
    print(f" {title}")
    print("="*80)

def run_experiment(script_name, description):
    """Run individual experiment script"""
    print(f"\nRunning: {description}")
    print(f"Script: {script_name}")
    print("-" * 60)
    
    start_time = time.time()
    
    try:
        # Run the script
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=False, 
                              check=True,
                              cwd=Path(__file__).parent)
        
        end_time = time.time()
        runtime = end_time - start_time
        
        print(f"\nCompleted successfully in {runtime:.1f} seconds")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\nError running {script_name}: {e}")
        return False
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        return False

def main():
    """Main execution function"""
    print_section("MEKANET CLASSIFICATION EXPERIMENTS - COMPLETE PIPELINE")
    
    # Verify data availability
    data_path = Path("../../data/demo_data/classification.csv")
    if not data_path.exists():
        print(f"Error: Demo dataset not found at {data_path}")
        print("Please ensure the demo dataset is available before running experiments.")
        return False
    
    # Create results directory
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    print(f"\nData verified: {data_path}")
    print(f"Results directory: {results_dir.absolute()}")
    
    # Define experiments to run
    experiments = [
        ("rfecv_feature_selection.py", "RFECV Feature Selection Analysis"),
        ("institutional_validation.py", "Cross-Institutional Validation Framework"),
        ("comprehensive_modeling.py", "Comprehensive Modeling Analysis")
    ]
    
    # Track experiment results
    results = {}
    total_start_time = time.time()
    
    # Run each experiment
    for script_name, description in experiments:
        print_section(description)
        success = run_experiment(script_name, description)
        results[script_name] = success
        
        if not success:
            print(f"\nWarning: {script_name} failed. Continuing with remaining experiments...")
    
    # Summary
    total_end_time = time.time()
    total_runtime = total_end_time - total_start_time
    
    print_section("EXPERIMENT PIPELINE SUMMARY")
    
    successful_experiments = sum(results.values())
    total_experiments = len(experiments)
    
    print(f"\nExperiments completed: {successful_experiments}/{total_experiments}")
    print(f"Total runtime: {total_runtime:.1f} seconds")
    print(f"Results saved to: {results_dir.absolute()}")
    
    print("\nIndividual experiment results:")
    for script_name, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        print(f"  {script_name:<35} {status}")
    
    # Verify output files
    print("\nGenerated output files:")
    if results_dir.exists():
        for file_path in sorted(results_dir.glob("*")):
            print(f"  {file_path.name}")
    
    # Final recommendations
    print("\nNext steps:")
    if successful_experiments == total_experiments:
        print("  1. Review generated reports in results/ directory")
        print("  2. Examine visualizations for insights")
        print("  3. Compare metrics with expected benchmarks")
        print("  4. Use results for manuscript preparation")
    else:
        print("  1. Check error messages for failed experiments")
        print("  2. Verify data format and dependencies")
        print("  3. Re-run failed experiments individually")
        print("  4. Contact support if issues persist")
    
    return successful_experiments == total_experiments

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)