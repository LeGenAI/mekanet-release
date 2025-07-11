#!/usr/bin/env python3
"""
Paper Reproduction Runner for TESSD Framework
Automated reproduction of all paper experiments with configurable settings
"""

import os
import sys
import json
import yaml
import time
import logging
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
from datetime import datetime
import traceback
import shutil

# Setup paths
sys.path.append(str(Path(__file__).parent.parent.parent))

from experiments.detection.tessd_framework import TESSDFramework
from experiments.detection.comprehensive_evaluator import ComprehensiveEvaluator, EvaluationConfig
from experiments.detection.institutional_validator import InstitutionalValidator, InstitutionalConfig
from experiments.detection.sahi_inference_module import SAHIInferenceModule, SAHIConfig
from experiments.detection.visualization_analyzer import DetectionVisualizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('paper_reproduction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment"""
    name: str
    description: str
    enabled: bool = True
    
    # Model settings
    model_path: str = "../../weights/epoch60.pt"
    
    # SAHI settings
    tile_size: int = 640
    overlap_ratio: float = 0.2
    confidence_thresholds: List[float] = None
    
    # Dataset settings
    datasets: List[str] = None
    
    # Evaluation settings
    iou_thresholds: List[float] = None
    
    # Output settings
    save_visualizations: bool = True
    save_detailed_results: bool = True
    
    # Resource settings
    max_images_per_dataset: Optional[int] = None  # For quick testing
    parallel_processing: bool = True
    
    def __post_init__(self):
        if self.confidence_thresholds is None:
            self.confidence_thresholds = [0.15, 0.20, 0.25]
        if self.datasets is None:
            self.datasets = ["B_hospital", "S_hospital"]
        if self.iou_thresholds is None:
            self.iou_thresholds = [0.5, 0.75]


@dataclass
class ReproductionResult:
    """Results from paper reproduction"""
    experiment_name: str
    status: str  # success, failed, skipped
    execution_time: float
    results_path: str
    key_metrics: Dict[str, Any]
    error_message: Optional[str] = None


class PaperReproductionRunner:
    """
    Comprehensive paper reproduction runner for TESSD experiments
    
    Features:
    - Configurable experiment pipeline
    - Automated result generation
    - Error handling and recovery
    - Progress tracking and checkpointing
    - Resource management
    - Reproducibility validation
    """
    
    def __init__(self, 
                 config_file: str = "configs/paper_reproduction_full.yaml",
                 output_dir: str = "./paper_reproduction_results",
                 checkpoint_file: str = None):
        """
        Initialize paper reproduction runner
        
        Args:
            config_file: Path to experiment configuration file
            output_dir: Output directory for all results
            checkpoint_file: Path to checkpoint file for resume capability
        """
        self.config_file = Path(config_file)
        self.output_dir = Path(output_dir)
        self.checkpoint_file = Path(checkpoint_file) if checkpoint_file else self.output_dir / "checkpoint.json"
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize experiment tracking
        self.experiment_configs = {}
        self.experiment_results = {}
        self.execution_order = []
        
        # Load configuration
        self._load_configuration()
        
        # Initialize checkpoint
        self._load_checkpoint()
        
        logger.info(f"Paper Reproduction Runner initialized")
        logger.info(f"Config: {self.config_file}")
        logger.info(f"Output: {self.output_dir}")
        logger.info(f"Experiments: {len(self.experiment_configs)}")
    
    def _load_configuration(self):
        """Load experiment configuration from YAML file"""
        try:
            with open(self.config_file, 'r') as f:
                config_data = yaml.safe_load(f)
            
            # Parse global settings
            self.global_settings = config_data.get('global_settings', {})
            
            # Parse experiment configurations
            experiments_data = config_data.get('experiments', {})
            
            for exp_name, exp_config in experiments_data.items():
                # Merge with global settings
                merged_config = {**self.global_settings, **exp_config}
                merged_config['name'] = exp_name
                
                # Create ExperimentConfig object
                self.experiment_configs[exp_name] = ExperimentConfig(**merged_config)
                
                if self.experiment_configs[exp_name].enabled:
                    self.execution_order.append(exp_name)
            
            logger.info(f"Loaded {len(self.experiment_configs)} experiment configurations")
            logger.info(f"Enabled experiments: {self.execution_order}")
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            self._create_default_config()
    
    def _create_default_config(self):
        """Create default configuration if none exists"""
        default_config = {
            'global_settings': {
                'model_path': '../../weights/epoch60.pt',
                'save_visualizations': True,
                'save_detailed_results': True,
                'parallel_processing': True
            },
            'experiments': {
                'baseline_evaluation': {
                    'description': 'Baseline TESSD evaluation on both hospitals',
                    'confidence_thresholds': [0.20],
                    'datasets': ['B_hospital', 'S_hospital'],
                    'enabled': True
                },
                'confidence_optimization': {
                    'description': 'Confidence threshold optimization study',
                    'confidence_thresholds': [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4],
                    'datasets': ['B_hospital', 'S_hospital'],
                    'enabled': True
                },
                'institutional_validation': {
                    'description': 'Cross-institutional validation study',
                    'confidence_thresholds': [0.20],
                    'datasets': ['B_hospital', 'S_hospital'],
                    'enabled': True
                },
                'sahi_parameter_study': {
                    'description': 'SAHI parameter optimization',
                    'tile_size': 640,
                    'overlap_ratio': 0.2,
                    'confidence_thresholds': [0.20],
                    'datasets': ['B_hospital'],
                    'enabled': True
                }
            }
        }
        
        # Save default config
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_file, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False, indent=2)
        
        logger.info(f"Created default configuration: {self.config_file}")
        
        # Reload with default config
        self._load_configuration()
    
    def _load_checkpoint(self):
        """Load checkpoint data for resume capability"""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r') as f:
                    checkpoint_data = json.load(f)
                
                self.experiment_results = checkpoint_data.get('experiment_results', {})
                completed_experiments = list(self.experiment_results.keys())
                
                logger.info(f"Loaded checkpoint with {len(completed_experiments)} completed experiments")
                
                # Filter execution order to skip completed experiments
                self.execution_order = [exp for exp in self.execution_order 
                                      if exp not in completed_experiments or 
                                      self.experiment_results[exp].get('status') != 'success']
                
                if completed_experiments:
                    logger.info(f"Will skip completed experiments: {completed_experiments}")
                
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}")
                self.experiment_results = {}
    
    def _save_checkpoint(self):
        """Save current progress to checkpoint file"""
        checkpoint_data = {
            'timestamp': datetime.now().isoformat(),
            'experiment_results': self.experiment_results,
            'execution_order': self.execution_order
        }
        
        try:
            with open(self.checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def run_all_experiments(self, 
                          resume: bool = True,
                          fail_fast: bool = False,
                          dry_run: bool = False) -> Dict[str, ReproductionResult]:
        """
        Run all configured experiments
        
        Args:
            resume: Whether to resume from checkpoint
            fail_fast: Whether to stop on first failure
            dry_run: Whether to only simulate execution
            
        Returns:
            Dictionary of experiment results
        """
        logger.info("🚀 Starting paper reproduction pipeline")
        logger.info(f"Experiments to run: {len(self.execution_order)}")
        
        if dry_run:
            logger.info("DRY RUN MODE - No actual execution")
        
        start_time = time.time()
        failed_experiments = []
        
        for i, exp_name in enumerate(self.execution_order):
            logger.info(f"\n📊 Running experiment {i+1}/{len(self.execution_order)}: {exp_name}")
            
            if dry_run:
                logger.info(f"[DRY RUN] Would execute: {exp_name}")
                continue
            
            try:
                result = self._run_single_experiment(exp_name)
                self.experiment_results[exp_name] = asdict(result)
                
                # Save checkpoint after each experiment
                self._save_checkpoint()
                
                if result.status == 'success':
                    logger.info(f"✅ {exp_name} completed successfully")
                else:
                    logger.error(f"❌ {exp_name} failed: {result.error_message}")
                    failed_experiments.append(exp_name)
                    
                    if fail_fast:
                        logger.error("Stopping due to fail_fast=True")
                        break
            
            except Exception as e:
                error_msg = f"Unexpected error in {exp_name}: {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                
                self.experiment_results[exp_name] = asdict(ReproductionResult(
                    experiment_name=exp_name,
                    status='failed',
                    execution_time=0,
                    results_path="",
                    key_metrics={},
                    error_message=error_msg
                ))
                
                failed_experiments.append(exp_name)
                self._save_checkpoint()
                
                if fail_fast:
                    break
        
        total_time = time.time() - start_time
        
        # Generate final report
        if not dry_run:
            self._generate_final_report(total_time, failed_experiments)
        
        logger.info(f"\n🎉 Paper reproduction completed in {total_time:.1f}s")
        logger.info(f"✅ Successful: {len(self.experiment_results) - len(failed_experiments)}")
        logger.info(f"❌ Failed: {len(failed_experiments)}")
        
        return self.experiment_results
    
    def _run_single_experiment(self, exp_name: str) -> ReproductionResult:
        """Run a single experiment"""
        config = self.experiment_configs[exp_name]
        exp_output_dir = self.output_dir / exp_name
        exp_output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Running: {config.description}")
        logger.info(f"Output: {exp_output_dir}")
        
        start_time = time.time()
        
        try:
            # Route to appropriate experiment type
            if 'baseline' in exp_name.lower():
                result = self._run_baseline_evaluation(config, exp_output_dir)
            elif 'confidence' in exp_name.lower():
                result = self._run_confidence_optimization(config, exp_output_dir)
            elif 'institutional' in exp_name.lower():
                result = self._run_institutional_validation(config, exp_output_dir)
            elif 'sahi' in exp_name.lower():
                result = self._run_sahi_parameter_study(config, exp_output_dir)
            else:
                # Default comprehensive evaluation
                result = self._run_comprehensive_evaluation(config, exp_output_dir)
            
            execution_time = time.time() - start_time
            
            return ReproductionResult(
                experiment_name=exp_name,
                status='success',
                execution_time=execution_time,
                results_path=str(exp_output_dir),
                key_metrics=result,
                error_message=None
            )
        
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Experiment failed: {str(e)}"
            
            return ReproductionResult(
                experiment_name=exp_name,
                status='failed',
                execution_time=execution_time,
                results_path=str(exp_output_dir),
                key_metrics={},
                error_message=error_msg
            )
    
    def _run_baseline_evaluation(self, config: ExperimentConfig, output_dir: Path) -> Dict[str, Any]:
        """Run baseline TESSD evaluation"""
        logger.info("Running baseline evaluation...")
        
        # Setup SAHI configuration
        sahi_config = SAHIConfig(
            tile_size=config.tile_size,
            overlap_ratio=config.overlap_ratio,
            confidence_threshold=config.confidence_thresholds[0]
        )
        
        # Setup evaluation configuration
        eval_config = EvaluationConfig()
        
        # Initialize evaluator
        evaluator = ComprehensiveEvaluator(
            model_path=config.model_path,
            sahi_config=sahi_config,
            eval_config=eval_config,
            experiment_name=config.name
        )
        
        # Create mock dataset info (in real implementation, load from config)
        datasets = {}
        for dataset_name in config.datasets:
            datasets[dataset_name] = {
                'institution': dataset_name,
                'images': self._get_dataset_images(dataset_name, config.max_images_per_dataset)
            }
        
        # Run evaluation on each dataset
        results = {}
        for dataset_name, dataset_info in datasets.items():
            dataset_output = output_dir / dataset_name
            eval_result = evaluator.evaluate_dataset(dataset_info, str(dataset_output))
            
            results[dataset_name] = {
                'map_50': eval_result.map_50,
                'map_75': eval_result.map_75,
                'precision_05': eval_result.precision_per_iou.get(0.5, 0),
                'recall_05': eval_result.recall_per_iou.get(0.5, 0),
                'f1_05': eval_result.f1_per_iou.get(0.5, 0),
                'total_images': eval_result.total_images,
                'total_detections': eval_result.total_detections,
                'processing_time': eval_result.average_processing_time
            }
        
        return results
    
    def _run_confidence_optimization(self, config: ExperimentConfig, output_dir: Path) -> Dict[str, Any]:
        """Run confidence threshold optimization study"""
        logger.info("Running confidence threshold optimization...")
        
        results = {}
        
        for dataset_name in config.datasets:
            dataset_results = {}
            dataset_images = self._get_dataset_images(dataset_name, config.max_images_per_dataset)
            
            for conf_threshold in config.confidence_thresholds:
                logger.info(f"Testing {dataset_name} with confidence {conf_threshold}")
                
                # Setup configurations
                sahi_config = SAHIConfig(
                    tile_size=config.tile_size,
                    overlap_ratio=config.overlap_ratio,
                    confidence_threshold=conf_threshold
                )
                
                evaluator = ComprehensiveEvaluator(
                    model_path=config.model_path,
                    sahi_config=sahi_config,
                    eval_config=EvaluationConfig(),
                    experiment_name=f"{config.name}_conf_{conf_threshold:.2f}"
                )
                
                # Run evaluation
                dataset_info = {
                    'institution': dataset_name,
                    'images': dataset_images
                }
                
                threshold_output = output_dir / dataset_name / f"conf_{conf_threshold:.2f}"
                eval_result = evaluator.evaluate_dataset(dataset_info, str(threshold_output))
                
                dataset_results[f"conf_{conf_threshold:.2f}"] = {
                    'confidence_threshold': conf_threshold,
                    'map_50': eval_result.map_50,
                    'precision_05': eval_result.precision_per_iou.get(0.5, 0),
                    'recall_05': eval_result.recall_per_iou.get(0.5, 0),
                    'f1_05': eval_result.f1_per_iou.get(0.5, 0),
                    'total_detections': eval_result.total_detections
                }
            
            results[dataset_name] = dataset_results
            
            # Find optimal threshold for this dataset
            f1_scores = [result['f1_05'] for result in dataset_results.values()]
            best_idx = np.argmax(f1_scores)
            best_conf = config.confidence_thresholds[best_idx]
            
            results[dataset_name]['optimal_threshold'] = best_conf
            results[dataset_name]['optimal_f1'] = f1_scores[best_idx]
        
        return results
    
    def _run_institutional_validation(self, config: ExperimentConfig, output_dir: Path) -> Dict[str, Any]:
        """Run cross-institutional validation"""
        logger.info("Running institutional validation...")
        
        # Setup institutional configuration
        institutional_config = InstitutionalConfig(
            training_institution=config.datasets[0],  # First dataset as training
            validation_institutions=config.datasets[1:],  # Rest as validation
            confidence_threshold_optimization=True,
            perform_statistical_tests=True
        )
        
        # Initialize validator
        validator = InstitutionalValidator(
            model_path=config.model_path,
            config=institutional_config,
            experiment_name=config.name
        )
        
        # Prepare datasets
        datasets = {}
        for dataset_name in config.datasets:
            datasets[dataset_name] = {
                'institution': dataset_name,
                'images': self._get_dataset_images(dataset_name, config.max_images_per_dataset)
            }
        
        # Run validation
        validation_results = validator.validate_across_institutions(datasets, str(output_dir))
        
        # Extract key metrics
        results = {
            'training_institution': institutional_config.training_institution,
            'validation_institutions': institutional_config.validation_institutions
        }
        
        for institution, val_result in validation_results.items():
            results[institution] = {
                'generalization_score': val_result.generalization_score,
                'performance_drop': val_result.performance_drop,
                'optimal_confidence': val_result.recommendations[0] if val_result.recommendations else "N/A"
            }
        
        return results
    
    def _run_sahi_parameter_study(self, config: ExperimentConfig, output_dir: Path) -> Dict[str, Any]:
        """Run SAHI parameter optimization study"""
        logger.info("Running SAHI parameter study...")
        
        # Test different tile sizes and overlap ratios
        tile_sizes = [512, 640, 768]
        overlap_ratios = [0.1, 0.2, 0.3]
        
        results = {}
        dataset_name = config.datasets[0]  # Use first dataset for parameter study
        dataset_images = self._get_dataset_images(dataset_name, min(10, config.max_images_per_dataset or 10))
        
        for tile_size in tile_sizes:
            for overlap_ratio in overlap_ratios:
                param_name = f"tile_{tile_size}_overlap_{overlap_ratio:.1f}"
                logger.info(f"Testing {param_name}")
                
                sahi_config = SAHIConfig(
                    tile_size=tile_size,
                    overlap_ratio=overlap_ratio,
                    confidence_threshold=config.confidence_thresholds[0]
                )
                
                evaluator = ComprehensiveEvaluator(
                    model_path=config.model_path,
                    sahi_config=sahi_config,
                    eval_config=EvaluationConfig(),
                    experiment_name=f"{config.name}_{param_name}"
                )
                
                dataset_info = {
                    'institution': dataset_name,
                    'images': dataset_images
                }
                
                param_output = output_dir / param_name
                eval_result = evaluator.evaluate_dataset(dataset_info, str(param_output))
                
                results[param_name] = {
                    'tile_size': tile_size,
                    'overlap_ratio': overlap_ratio,
                    'map_50': eval_result.map_50,
                    'processing_time': eval_result.average_processing_time,
                    'total_detections': eval_result.total_detections
                }
        
        # Find optimal parameters
        map_scores = [result['map_50'] for result in results.values()]
        best_param = list(results.keys())[np.argmax(map_scores)]
        
        results['optimal_parameters'] = best_param
        results['optimal_map'] = max(map_scores)
        
        return results
    
    def _run_comprehensive_evaluation(self, config: ExperimentConfig, output_dir: Path) -> Dict[str, Any]:
        """Run comprehensive evaluation (default)"""
        return self._run_baseline_evaluation(config, output_dir)
    
    def _get_dataset_images(self, dataset_name: str, max_images: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get dataset images (mock implementation)
        In real implementation, this would load from actual dataset configuration
        """
        # Mock dataset images
        mock_images = []
        
        # Create mock image entries
        num_images = min(max_images or 50, 50)  # Limit for testing
        
        for i in range(num_images):
            mock_images.append({
                'path': f"/path/to/{dataset_name}/image_{i:03d}.jpg",
                'annotations': [
                    {
                        'bbox': [100 + i*10, 100 + i*5, 50, 50],  # Mock bounding box
                        'category_id': 1,
                        'iscrowd': 0
                    }
                ]
            })
        
        return mock_images
    
    def _generate_final_report(self, total_time: float, failed_experiments: List[str]):
        """Generate comprehensive final report"""
        report_path = self.output_dir / "paper_reproduction_report.json"
        
        # Calculate summary statistics
        successful_experiments = [name for name in self.experiment_results.keys() 
                                if self.experiment_results[name].get('status') == 'success']
        
        summary = {
            'reproduction_metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_execution_time': total_time,
                'config_file': str(self.config_file),
                'output_directory': str(self.output_dir)
            },
            'experiment_summary': {
                'total_experiments': len(self.experiment_configs),
                'successful_experiments': len(successful_experiments),
                'failed_experiments': len(failed_experiments),
                'success_rate': len(successful_experiments) / len(self.experiment_configs) if self.experiment_configs else 0
            },
            'detailed_results': self.experiment_results,
            'failed_experiments': failed_experiments
        }
        
        # Save report
        with open(report_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Generate CSV summary for easy analysis
        self._generate_csv_summary()
        
        logger.info(f"📊 Final report saved: {report_path}")
    
    def _generate_csv_summary(self):
        """Generate CSV summary of key metrics"""
        summary_rows = []
        
        for exp_name, result_data in self.experiment_results.items():
            if result_data.get('status') == 'success':
                key_metrics = result_data.get('key_metrics', {})
                
                # Handle different result structures
                if isinstance(key_metrics, dict):
                    for dataset_name, metrics in key_metrics.items():
                        if isinstance(metrics, dict) and 'map_50' in metrics:
                            summary_rows.append({
                                'experiment': exp_name,
                                'dataset': dataset_name,
                                'map_50': metrics.get('map_50', 0),
                                'map_75': metrics.get('map_75', 0),
                                'precision_05': metrics.get('precision_05', 0),
                                'recall_05': metrics.get('recall_05', 0),
                                'f1_05': metrics.get('f1_05', 0),
                                'execution_time': result_data.get('execution_time', 0),
                                'status': result_data.get('status', 'unknown')
                            })
        
        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            csv_path = self.output_dir / "paper_reproduction_summary.csv"
            summary_df.to_csv(csv_path, index=False)
            logger.info(f"📈 CSV summary saved: {csv_path}")


def main():
    """Main function for paper reproduction runner"""
    parser = argparse.ArgumentParser(description="Paper Reproduction Runner for TESSD")
    parser.add_argument("--config", default="configs/paper_reproduction_full.yaml", 
                       help="Path to experiment configuration file")
    parser.add_argument("--output", default="./paper_reproduction_results", 
                       help="Output directory for results")
    parser.add_argument("--resume", action="store_true", 
                       help="Resume from checkpoint")
    parser.add_argument("--fail-fast", action="store_true", 
                       help="Stop on first failure")
    parser.add_argument("--dry-run", action="store_true", 
                       help="Simulate execution without running experiments")
    parser.add_argument("--checkpoint", help="Custom checkpoint file path")
    
    args = parser.parse_args()
    
    # Initialize runner
    runner = PaperReproductionRunner(
        config_file=args.config,
        output_dir=args.output,
        checkpoint_file=args.checkpoint
    )
    
    # Run experiments
    results = runner.run_all_experiments(
        resume=args.resume,
        fail_fast=args.fail_fast,
        dry_run=args.dry_run
    )
    
    # Print final summary
    successful = sum(1 for r in results.values() if r.get('status') == 'success')
    total = len(results)
    
    print(f"\n🎉 Paper reproduction completed!")
    print(f"✅ Success rate: {successful}/{total} ({successful/total*100:.1f}%)")
    print(f"📁 Results saved to: {args.output}")
    
    if successful < total:
        print(f"❌ Failed experiments: {total - successful}")
        failed_names = [name for name, result in results.items() 
                       if result.get('status') != 'success']
        for failed_name in failed_names:
            print(f"  - {failed_name}")


if __name__ == "__main__":
    main() 