"""
MekaNet Classification - Path Setup Utility
===========================================

This utility resolves path configuration issues in the execution environment
to ensure proper module imports and system integration.

Author: MekaNet Research Team
License: MIT
"""

import sys
import os
from pathlib import Path

def setup_paths():
    """
    Configure system paths for MekaNet classification experiments.
    
    This function adds necessary directories to sys.path to ensure proper
    module imports, particularly for the config, utils, and experiments modules.
    
    Returns:
        bool: True if path setup completed successfully
    """
    # Get current script directory
    current_dir = Path(__file__).parent
    
    # Define paths to add to sys.path
    paths_to_add = [
        str(current_dir),  # classification folder
        str(current_dir.parent.parent / "mekanet_excellence_framework"),  # config parent folder
        str(current_dir / "utils"),  # utils folder
        str(current_dir / "experiments")  # experiments folder
    ]
    
    # Add paths to sys.path if they exist
    for path in paths_to_add:
        if path not in sys.path and os.path.exists(path):
            sys.path.insert(0, path)
            print(f"✅ Added to path: {path}")
    
    # Set PYTHONPATH environment variable
    os.environ['PYTHONPATH'] = ':'.join(paths_to_add)
    
    print("🚀 Path setup completed successfully!")
    return True

def verify_imports():
    """
    Verify that all required modules can be imported successfully.
    
    This function attempts to import core modules and reports any import failures.
    It's particularly useful for troubleshooting deployment and runtime issues.
    
    Returns:
        bool: True if all imports successful, False otherwise
    """
    print("\n🔍 Verifying module imports...")
    
    # Test core configuration imports
    try:
        from config.experiment_config import EXPERIMENT_CONFIG, FEATURE_CATEGORIES, FEATURE_COSTS
        print("✅ Configuration import successful")
    except ImportError as e:
        print(f"❌ Configuration import failed: {e}")
        return False
    
    # Test data loading utilities
    try:
        from utils.data_loader import MekaNetDataLoader
        print("✅ Data loader import successful")
    except ImportError as e:
        print(f"❌ Data loader import failed: {e}")
        return False
    
    # Test statistical utilities
    try:
        from utils.statistical_utils import calculate_confidence_intervals, bootstrap_metric
        print("✅ Statistical utilities import successful")
    except ImportError as e:
        print(f"❌ Statistical utilities import failed: {e}")
        return False
    
    # Test visualization utilities
    try:
        from utils.visualization_utils import create_performance_plots
        print("✅ Visualization utilities import successful")
    except ImportError as e:
        print(f"❌ Visualization utilities import failed: {e}")
        return False
    
    print("🎉 All module imports verified successfully!")
    return True

def diagnose_environment():
    """
    Perform comprehensive environment diagnostics.
    
    This function checks Python version, installed packages, and system paths
    to help troubleshoot any deployment or runtime issues.
    """
    print("\n🔧 Environment Diagnostics")
    print("-" * 40)
    
    # Python version
    print(f"Python version: {sys.version}")
    
    # Current working directory
    print(f"Working directory: {os.getcwd()}")
    
    # Python path
    print(f"Python path entries: {len(sys.path)}")
    for i, path in enumerate(sys.path[:5]):  # Show first 5 entries
        print(f"  {i+1}. {path}")
    if len(sys.path) > 5:
        print(f"  ... and {len(sys.path) - 5} more entries")
    
    # Key dependencies
    required_packages = ['numpy', 'pandas', 'scikit-learn', 'matplotlib', 'seaborn']
    print(f"\nRequired packages:")
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} (not installed)")

if __name__ == "__main__":
    print("MekaNet Classification - Environment Setup")
    print("=" * 50)
    
    # Run setup and verification
    setup_success = setup_paths()
    import_success = verify_imports()
    
    # Run diagnostics
    diagnose_environment()
    
    # Final status
    if setup_success and import_success:
        print("\n🎉 Environment setup completed successfully!")
        print("Ready to run MekaNet classification experiments.")
    else:
        print("\n⚠️ Environment setup encountered issues.")
        print("Please check the error messages above and resolve any missing dependencies.") 