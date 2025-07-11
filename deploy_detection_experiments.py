#!/usr/bin/env python3
"""
MekaNet Detection Experiments - One-Click Deployment
Automated setup and execution of TESSD detection experiments for paper reproduction
"""

import os
import sys
import subprocess
import time
import logging
from pathlib import Path
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MekaNetDetectionDeployment:
    """
    One-click deployment for MekaNet detection experiments
    
    Handles:
    - Environment setup and verification
    - Model weight checking
    - Demo data preparation  
    - Complete experiment pipeline execution
    """
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.detection_dir = self.project_root / "experiments" / "detection"
        self.deployment_results = {}
        
    def print_banner(self):
        """Print deployment banner"""
        banner = """
╔═══════════════════════════════════════════════════════════════╗
║                    MekaNet Detection Experiments              ║
║                    One-Click Deployment                       ║
║                                                               ║
║  🔬 TESSD Framework (Tiling-Enhanced Semi-Supervised)        ║
║  🏥 Cross-Institutional Validation                           ║
║  📊 Comprehensive Performance Analysis                       ║
╚═══════════════════════════════════════════════════════════════╝
        """
        print(banner)
        logger.info("Starting MekaNet Detection Experiments Deployment")
    
    def run_environment_setup(self) -> bool:
        """Run environment setup and verification"""
        logger.info("🔧 Setting up environment...")
        
        try:
            # Change to detection directory
            os.chdir(self.detection_dir)
            
            # Run setup script
            result = subprocess.run([
                sys.executable, "setup_detection_experiments.py", "--comprehensive"
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                logger.info("✅ Environment setup completed successfully")
                return True
            else:
                logger.error("❌ Environment setup failed")
                logger.error(f"Error: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("❌ Environment setup timed out")
            return False
        except Exception as e:
            logger.error(f"❌ Environment setup error: {e}")
            return False
    
    def run_detection_experiments(self, experiment_type: str = "demo") -> bool:
        """
        Run detection experiments
        
        Args:
            experiment_type: Type of experiment ('demo', 'threshold', 'complete')
        """
        logger.info(f"🚀 Running {experiment_type} detection experiments...")
        
        try:
            # Change to detection directory
            os.chdir(self.detection_dir)
            
            # Select configuration based on experiment type
            if experiment_type == "demo":
                config_file = "configs/paper_reproduction.yaml"
                cmd_args = ["--demo-only"]
            elif experiment_type == "threshold":
                config_file = "configs/threshold_analysis.yaml" 
                cmd_args = ["--threshold-analysis"]
            else:  # complete
                config_file = "configs/paper_reproduction.yaml"
                cmd_args = ["--complete"]
            
            # Run experiments
            cmd = [sys.executable, "run_detection_experiments.py", 
                   "--config", config_file] + cmd_args
            
            logger.info(f"Executing: {' '.join(cmd)}")
            
            # Run with real-time output
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT, 
                text=True,
                universal_newlines=True
            )
            
            # Stream output in real-time
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    print(output.strip())
            
            # Get final return code
            return_code = process.poll()
            
            if return_code == 0:
                logger.info("✅ Detection experiments completed successfully")
                return True
            else:
                logger.error("❌ Detection experiments failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Experiment execution error: {e}")
            return False
    
    def show_results_summary(self):
        """Display summary of experiment results"""
        results_dir = self.detection_dir / "results"
        
        if not results_dir.exists():
            logger.warning("No results directory found")
            return
        
        print("\n" + "="*60)
        print("📊 EXPERIMENT RESULTS SUMMARY")
        print("="*60)
        
        # List result directories
        for result_dir in results_dir.iterdir():
            if result_dir.is_dir():
                print(f"\n📁 {result_dir.name}:")
                
                # List key result files
                for result_file in result_dir.iterdir():
                    if result_file.suffix in ['.csv', '.png', '.pdf', '.txt']:
                        file_size = result_file.stat().st_size / 1024  # KB
                        print(f"   - {result_file.name} ({file_size:.1f} KB)")
        
        print(f"\n🔗 Full results available in: {results_dir}")
        print("="*60)
    
    def deploy(self, experiment_type: str = "demo") -> Dict[str, Any]:
        """
        Run complete deployment pipeline
        
        Args:
            experiment_type: 'demo' (5 min), 'threshold' (10 min), 'complete' (15+ min)
        """
        start_time = time.time()
        
        self.print_banner()
        
        # Step 1: Environment setup
        setup_success = self.run_environment_setup()
        if not setup_success:
            return {"success": False, "error": "Environment setup failed"}
        
        # Step 2: Run experiments
        experiment_success = self.run_detection_experiments(experiment_type)
        if not experiment_success:
            return {"success": False, "error": "Experiments failed"}
        
        # Step 3: Show results
        self.show_results_summary()
        
        # Calculate total time
        total_time = time.time() - start_time
        
        result = {
            "success": True,
            "experiment_type": experiment_type,
            "total_time_minutes": round(total_time / 60, 2),
            "results_dir": str(self.detection_dir / "results")
        }
        
        print(f"\n🎉 Deployment completed successfully in {result['total_time_minutes']} minutes!")
        print(f"📁 Results saved to: {result['results_dir']}")
        
        return result


def main():
    """Main deployment function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="MekaNet Detection Experiments - One-Click Deployment"
    )
    parser.add_argument(
        "--experiment-type", 
        choices=["demo", "threshold", "complete"],
        default="demo",
        help="Type of experiment to run (default: demo)"
    )
    parser.add_argument(
        "--setup-only",
        action="store_true",
        help="Only run environment setup, skip experiments"
    )
    
    args = parser.parse_args()
    
    # Create deployment instance
    deployment = MekaNetDetectionDeployment()
    
    if args.setup_only:
        deployment.print_banner()
        success = deployment.run_environment_setup()
        if success:
            print("✅ Environment setup completed successfully")
        else:
            print("❌ Environment setup failed")
            sys.exit(1)
    else:
        # Run complete deployment
        result = deployment.deploy(args.experiment_type)
        
        if not result["success"]:
            print(f"❌ Deployment failed: {result.get('error', 'Unknown error')}")
            sys.exit(1)


if __name__ == "__main__":
    main() 