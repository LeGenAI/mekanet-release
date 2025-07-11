#!/usr/bin/env python3
"""
Setup Script for MekaNet Detection Experiments
Prepares environment and verifies dependencies for TESSD framework deployment
"""

import sys
import subprocess
import importlib
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DetectionExperimentSetup:
    """
    Setup and verification for MekaNet detection experiments
    
    Performs comprehensive environment preparation including:
    - Dependency verification
    - Model weight availability
    - Demo data preparation
    - Directory structure setup
    """
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.detection_dir = Path(__file__).parent
        self.setup_results = {
            'dependencies': {},
            'model_weights': {},
            'demo_data': {},
            'directories': {},
            'environment': {}
        }
    
    def check_python_version(self) -> bool:
        """Check if Python version is compatible"""
        logger.info("Checking Python version...")
        
        version_info = sys.version_info
        required_major, required_minor = 3, 8
        
        if version_info.major < required_major or \
           (version_info.major == required_major and version_info.minor < required_minor):
            logger.error(f"Python {required_major}.{required_minor}+ required. Found: {version_info.major}.{version_info.minor}")
            self.setup_results['environment']['python_version'] = False
            return False
        
        logger.info(f"✅ Python version: {version_info.major}.{version_info.minor}.{version_info.micro}")
        self.setup_results['environment']['python_version'] = True
        return True
    
    def check_dependencies(self) -> bool:
        """Check if all required packages are available"""
        logger.info("Checking dependencies...")
        
        required_packages = [
            ('cv2', 'opencv-python'),
            ('numpy', 'numpy'),
            ('pandas', 'pandas'),
            ('torch', 'torch'),
            ('ultralytics', 'ultralytics'),
            ('sahi', 'sahi'),
            ('sklearn', 'scikit-learn'),
            ('matplotlib', 'matplotlib'),
            ('seaborn', 'seaborn'),
            ('scipy', 'scipy'),
            ('yaml', 'PyYAML'),
            ('PIL', 'Pillow')
        ]
        
        missing_packages = []
        available_packages = []
        
        for import_name, package_name in required_packages:
            try:
                importlib.import_module(import_name)
                available_packages.append(package_name)
                logger.info(f"✅ {package_name}")
            except ImportError:
                missing_packages.append(package_name)
                logger.warning(f"❌ {package_name} not found")
        
        self.setup_results['dependencies']['available'] = available_packages
        self.setup_results['dependencies']['missing'] = missing_packages
        
        if missing_packages:
            logger.error("Missing required packages. Install with:")
            logger.error(f"pip install {' '.join(missing_packages)}")
            return False
        
        logger.info("✅ All dependencies available")
        return True
    
    def check_model_weights(self) -> bool:
        """Check availability of model weights"""
        logger.info("Checking model weights...")
        
        weights_dir = self.project_root / "weights"
        model_files = [
            "epoch60.pt",
            "best.pt",
            "last.pt"
        ]
        
        available_weights = []
        missing_weights = []
        
        for model_file in model_files:
            model_path = weights_dir / model_file
            if model_path.exists():
                file_size = model_path.stat().st_size / (1024 * 1024)  # MB
                available_weights.append({
                    'file': model_file,
                    'path': str(model_path),
                    'size_mb': round(file_size, 2)
                })
                logger.info(f"✅ {model_file} ({file_size:.1f} MB)")
            else:
                missing_weights.append(model_file)
                logger.warning(f"❌ {model_file} not found")
        
        self.setup_results['model_weights']['available'] = available_weights
        self.setup_results['model_weights']['missing'] = missing_weights
        
        if not available_weights:
            logger.error("No model weights found!")
            logger.info("Download weights using:")
            logger.info("cd ../../weights && python download_weights.py")
            return False
        
        # Check if primary model (epoch60.pt) is available
        primary_model = weights_dir / "epoch60.pt"
        if primary_model.exists():
            logger.info("✅ Primary model (epoch60.pt) available")
            return True
        else:
            logger.warning("Primary model (epoch60.pt) not found, but other weights available")
            return len(available_weights) > 0
    
    def check_demo_data(self) -> bool:
        """Check availability of demo data"""
        logger.info("Checking demo data...")
        
        demo_data_paths = [
            self.project_root / "data" / "demo_data",
            self.project_root / "data" / "sample_images",
            self.detection_dir.parent.parent / "data" / "demo_data"
        ]
        
        available_images = []
        
        for demo_path in demo_data_paths:
            if demo_path.exists():
                for ext in ['*.jpg', '*.jpeg', '*.png']:
                    images = list(demo_path.glob(ext))
                    for img in images:
                        available_images.append({
                            'file': img.name,
                            'path': str(img),
                            'size_mb': round(img.stat().st_size / (1024 * 1024), 2)
                        })
        
        self.setup_results['demo_data']['available'] = available_images
        
        if not available_images:
            logger.warning("No demo images found")
            logger.info("Demo images expected in:")
            for path in demo_data_paths:
                logger.info(f"  - {path}")
            return False
        
        logger.info(f"✅ Found {len(available_images)} demo images")
        for img in available_images[:3]:  # Show first 3
            logger.info(f"   - {img['file']} ({img['size_mb']} MB)")
        
        if len(available_images) > 3:
            logger.info(f"   ... and {len(available_images) - 3} more")
        
        return True
    
    def setup_directories(self) -> bool:
        """Create necessary output directories"""
        logger.info("Setting up directories...")
        
        required_dirs = [
            self.detection_dir / "results",
            self.detection_dir / "results" / "demo_results",
            self.detection_dir / "results" / "threshold_analysis", 
            self.detection_dir / "results" / "institutional_validation",
            self.detection_dir / "results" / "visualizations"
        ]
        
        created_dirs = []
        
        for dir_path in required_dirs:
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                created_dirs.append(str(dir_path))
                logger.info(f"✅ {dir_path.relative_to(self.detection_dir)}")
            except Exception as e:
                logger.error(f"❌ Failed to create {dir_path}: {e}")
                return False
        
        self.setup_results['directories']['created'] = created_dirs
        return True
    
    def verify_configurations(self) -> bool:
        """Verify configuration files are available"""
        logger.info("Checking configuration files...")
        
        configs_dir = self.detection_dir / "configs"
        required_configs = [
            "paper_reproduction.yaml",
            "threshold_analysis.yaml"
        ]
        
        available_configs = []
        missing_configs = []
        
        for config_file in required_configs:
            config_path = configs_dir / config_file
            if config_path.exists():
                available_configs.append(str(config_path))
                logger.info(f"✅ {config_file}")
            else:
                missing_configs.append(config_file)
                logger.warning(f"❌ {config_file} not found")
        
        self.setup_results['configurations'] = {
            'available': available_configs,
            'missing': missing_configs
        }
        
        return len(available_configs) > 0
    
    def test_imports(self) -> bool:
        """Test if detection modules can be imported"""
        logger.info("Testing module imports...")
        
        test_modules = [
            ('experiments.detection.tessd_framework', 'TESSDFramework'),
            ('experiments.detection.detection_evaluator', 'DetectionEvaluator'),
            ('experiments.detection.institutional_validator', 'InstitutionalValidator')
        ]
        
        import_errors = []
        successful_imports = []
        
        # Add project root to path
        if str(self.project_root) not in sys.path:
            sys.path.insert(0, str(self.project_root))
        
        for module_path, class_name in test_modules:
            try:
                module = importlib.import_module(module_path)
                class_obj = getattr(module, class_name)
                successful_imports.append(f"{module_path}.{class_name}")
                logger.info(f"✅ {module_path}.{class_name}")
            except Exception as e:
                import_errors.append(f"{module_path}.{class_name}: {str(e)}")
                logger.error(f"❌ {module_path}.{class_name}: {str(e)}")
        
        self.setup_results['imports'] = {
            'successful': successful_imports,
            'failed': import_errors
        }
        
        return len(import_errors) == 0
    
    def run_comprehensive_setup(self) -> Dict[str, Any]:
        """Run complete setup and verification process"""
        logger.info("🚀 Starting MekaNet Detection Experiments Setup")
        logger.info("=" * 60)
        
        setup_steps = [
            ("Python Version", self.check_python_version),
            ("Dependencies", self.check_dependencies),
            ("Model Weights", self.check_model_weights),
            ("Demo Data", self.check_demo_data),
            ("Directories", self.setup_directories),
            ("Configurations", self.verify_configurations),
            ("Module Imports", self.test_imports)
        ]
        
        overall_success = True
        
        for step_name, step_function in setup_steps:
            logger.info(f"\n📋 {step_name}")
            logger.info("-" * 30)
            
            try:
                success = step_function()
                if not success:
                    overall_success = False
                    logger.warning(f"⚠️  {step_name} setup incomplete")
                else:
                    logger.info(f"✅ {step_name} setup complete")
            except Exception as e:
                logger.error(f"❌ {step_name} setup failed: {str(e)}")
                overall_success = False
        
        # Generate setup summary
        logger.info("\n" + "=" * 60)
        if overall_success:
            logger.info("🎉 SETUP COMPLETED SUCCESSFULLY!")
            logger.info("✅ Ready to run detection experiments")
            logger.info("\nNext steps:")
            logger.info("  python run_detection_experiments.py --config configs/paper_reproduction.yaml")
        else:
            logger.error("❌ SETUP INCOMPLETE")
            logger.error("Please resolve the issues above before running experiments")
        
        self.setup_results['overall_success'] = overall_success
        return self.setup_results
    
    def create_quick_test_script(self) -> str:
        """Create a quick test script for verification"""
        test_script_content = '''#!/usr/bin/env python3
"""
Quick test script for MekaNet detection experiments
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from experiments.detection.tessd_framework import TESSDFramework
    print("✅ TESSDFramework import successful")
    
    from experiments.detection.detection_evaluator import DetectionEvaluator
    print("✅ DetectionEvaluator import successful")
    
    from experiments.detection.institutional_validator import InstitutionalValidator
    print("✅ InstitutionalValidator import successful")
    
    print("\\n🎉 All detection modules imported successfully!")
    print("Ready to run detection experiments.")
    
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("Please run setup_detection_experiments.py first")
    sys.exit(1)
'''
        
        test_script_path = self.detection_dir / "test_imports.py"
        with open(test_script_path, 'w') as f:
            f.write(test_script_content)
        
        # Make executable
        test_script_path.chmod(0o755)
        
        logger.info(f"Created quick test script: {test_script_path}")
        return str(test_script_path)


def main():
    """Main setup function"""
    setup = DetectionExperimentSetup()
    
    # Run comprehensive setup
    results = setup.run_comprehensive_setup()
    
    # Create quick test script
    test_script = setup.create_quick_test_script()
    
    # Save setup results
    import json
    results_file = setup.detection_dir / "setup_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"\nSetup results saved to: {results_file}")
    logger.info(f"Quick test script created: {test_script}")
    
    return 0 if results['overall_success'] else 1


if __name__ == "__main__":
    exit(main()) 