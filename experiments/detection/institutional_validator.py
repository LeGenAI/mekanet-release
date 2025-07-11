#!/usr/bin/env python3
"""
Institutional Validation Framework for TESSD
Cross-hospital validation between B hospital and S hospital datasets
"""

import cv2
import numpy as np
import pandas as pd
import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

# Setup paths
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from experiments.detection.tessd_framework import TESSDFramework
from experiments.detection.comprehensive_evaluator import ComprehensiveEvaluator, EvaluationConfig
from experiments.detection.sahi_inference_module import SAHIConfig
from experiments.detection.visualization_analyzer import DetectionVisualizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class InstitutionalConfig:
    """Configuration for institutional validation"""
    training_institution: str = "B_hospital"
    validation_institutions: List[str] = None
    
    # Cross-validation settings
    perform_domain_adaptation: bool = False
    confidence_threshold_optimization: bool = True
    
    # Performance targets
    target_map_training: float = 0.85  # mAP@0.5 target for training institution
    target_map_validation: float = 0.80  # mAP@0.5 target for validation institutions
    
    # Statistical analysis
    perform_statistical_tests: bool = True
    significance_level: float = 0.05
    
    def __post_init__(self):
        if self.validation_institutions is None:
            self.validation_institutions = ["S_hospital"]


@dataclass 
class ValidationResult:
    """Results from institutional validation"""
    training_institution: str
    validation_institution: str
    
    # Performance metrics
    training_performance: Dict[str, float]
    validation_performance: Dict[str, float]
    performance_drop: Dict[str, float]
    
    # Statistical analysis
    statistical_tests: Dict[str, Any]
    
    # Generalization analysis
    generalization_score: float
    domain_gap_analysis: Dict[str, Any]
    
    # Recommendations
    recommendations: List[str]


class InstitutionalValidator:
    """
    Framework for cross-institutional validation of TESSD models
    
    Features:
    - Cross-hospital performance validation
    - Domain gap analysis
    - Statistical significance testing
    - Confidence threshold optimization per institution
    - Generalization assessment
    """
    
    def __init__(self, 
                 model_path: str,
                 config: InstitutionalConfig = None,
                 experiment_name: str = "institutional_validation"):
        """
        Initialize institutional validator
        
        Args:
            model_path: Path to trained model
            config: Institutional validation configuration
            experiment_name: Name for experiment tracking
        """
        self.model_path = model_path
        self.config = config if config is not None else InstitutionalConfig()
        self.experiment_name = experiment_name
        
        # Initialize evaluation components
        self.sahi_config = SAHIConfig(confidence_threshold=0.20)  # Initial threshold
        self.eval_config = EvaluationConfig()
        
        self.evaluator = ComprehensiveEvaluator(
            model_path=model_path,
            sahi_config=self.sahi_config,
            eval_config=self.eval_config,
            experiment_name=experiment_name
        )
        
        self.visualizer = DetectionVisualizer()
        
        logger.info(f"Institutional Validator initialized: {experiment_name}")
        logger.info(f"Training institution: {self.config.training_institution}")
        logger.info(f"Validation institutions: {self.config.validation_institutions}")
    
    def validate_across_institutions(self, 
                                   datasets: Dict[str, Dict[str, Any]],
                                   output_dir: str = None) -> Dict[str, ValidationResult]:
        """
        Perform comprehensive cross-institutional validation
        
        Args:
            datasets: Dictionary mapping institution names to dataset info
            output_dir: Output directory for results
            
        Returns:
            Dictionary of validation results
        """
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("Starting cross-institutional validation...")
        
        # First, evaluate on training institution
        training_data = datasets[self.config.training_institution]
        training_result = self.evaluator.evaluate_dataset(training_data, output_dir)
        
        validation_results = {}
        
        # Evaluate on each validation institution
        for val_institution in self.config.validation_institutions:
            if val_institution not in datasets:
                logger.warning(f"Dataset for {val_institution} not found")
                continue
            
            logger.info(f"Validating on {val_institution}...")
            
            val_data = datasets[val_institution]
            val_result = self.evaluator.evaluate_dataset(val_data, output_dir)
            
            # Perform institutional comparison
            validation_result = self._compare_institutional_performance(
                training_result, 
                val_result,
                self.config.training_institution,
                val_institution
            )
            
            # Optimize confidence threshold for validation institution
            if self.config.confidence_threshold_optimization:
                optimal_threshold = self._optimize_confidence_threshold(
                    val_data, val_institution, output_dir
                )
                validation_result.recommendations.append(
                    f"Optimal confidence threshold for {val_institution}: {optimal_threshold:.3f}"
                )
            
            validation_results[val_institution] = validation_result
        
        # Create comprehensive comparison
        if output_dir:
            self._create_institutional_comparison_report(
                training_result, 
                {inst: datasets[inst] for inst in self.config.validation_institutions},
                validation_results,
                output_path
            )
        
        return validation_results
    
    def _compare_institutional_performance(self, 
                                         training_result,
                                         validation_result,
                                         training_inst: str,
                                         validation_inst: str) -> ValidationResult:
        """Compare performance between training and validation institutions"""
        
        # Extract key metrics
        training_metrics = {
            'map_50': training_result.map_50,
            'map_75': training_result.map_75,
            'precision_05': training_result.precision_per_iou.get(0.5, 0),
            'recall_05': training_result.recall_per_iou.get(0.5, 0),
            'f1_05': training_result.f1_per_iou.get(0.5, 0)
        }
        
        validation_metrics = {
            'map_50': validation_result.map_50,
            'map_75': validation_result.map_75,
            'precision_05': validation_result.precision_per_iou.get(0.5, 0),
            'recall_05': validation_result.recall_per_iou.get(0.5, 0),
            'f1_05': validation_result.f1_per_iou.get(0.5, 0)
        }
        
        # Calculate performance drops
        performance_drop = {}
        for metric in training_metrics:
            if training_metrics[metric] > 0:
                drop = (training_metrics[metric] - validation_metrics[metric]) / training_metrics[metric]
                performance_drop[f"{metric}_drop"] = drop
            else:
                performance_drop[f"{metric}_drop"] = 0
        
        # Calculate generalization score
        generalization_score = self._calculate_generalization_score(
            training_metrics, validation_metrics
        )
        
        # Perform statistical tests
        statistical_tests = {}
        if self.config.perform_statistical_tests:
            statistical_tests = self._perform_statistical_analysis(
                training_result, validation_result
            )
        
        # Domain gap analysis
        domain_gap = self._analyze_domain_gap(training_result, validation_result)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            training_metrics, validation_metrics, performance_drop, generalization_score
        )
        
        return ValidationResult(
            training_institution=training_inst,
            validation_institution=validation_inst,
            training_performance=training_metrics,
            validation_performance=validation_metrics,
            performance_drop=performance_drop,
            statistical_tests=statistical_tests,
            generalization_score=generalization_score,
            domain_gap_analysis=domain_gap,
            recommendations=recommendations
        )
    
    def _calculate_generalization_score(self, 
                                      training_metrics: Dict[str, float],
                                      validation_metrics: Dict[str, float]) -> float:
        """
        Calculate overall generalization score (0-1, higher is better)
        
        Based on weighted combination of metric preservation across institutions
        """
        weights = {
            'map_50': 0.4,    # Most important
            'precision_05': 0.2,
            'recall_05': 0.2,
            'f1_05': 0.2
        }
        
        generalization_components = []
        
        for metric, weight in weights.items():
            if training_metrics[metric] > 0:
                preservation = validation_metrics[metric] / training_metrics[metric]
                # Cap at 1.0 (no bonus for exceeding training performance)
                preservation = min(1.0, preservation)
                generalization_components.append(weight * preservation)
        
        return sum(generalization_components) if generalization_components else 0
    
    def _perform_statistical_analysis(self, training_result, validation_result) -> Dict[str, Any]:
        """Perform statistical significance tests"""
        # This is a simplified statistical analysis
        # In practice, you would need multiple runs or bootstrap sampling
        
        statistical_tests = {
            'performance_difference_significant': False,
            'confidence_interval_map_50': None,
            'effect_size': 0,
            'notes': "Statistical tests require multiple evaluation runs for proper significance testing"
        }
        
        # Calculate effect size (Cohen's d approximation)
        map_50_diff = abs(training_result.map_50 - validation_result.map_50)
        pooled_std = 0.05  # Estimated standard deviation
        effect_size = map_50_diff / pooled_std if pooled_std > 0 else 0
        
        statistical_tests['effect_size'] = effect_size
        
        # Simple threshold-based significance (placeholder)
        if map_50_diff > 0.05:  # 5% difference threshold
            statistical_tests['performance_difference_significant'] = True
        
        return statistical_tests
    
    def _analyze_domain_gap(self, training_result, validation_result) -> Dict[str, Any]:
        """Analyze domain gap between institutions"""
        
        domain_gap = {
            'detection_density_ratio': 0,
            'confidence_distribution_shift': 0,
            'size_distribution_shift': 0,
            'spatial_distribution_shift': 0
        }
        
        # Detection density comparison
        train_density = training_result.total_detections / training_result.total_images
        val_density = validation_result.total_detections / validation_result.total_images
        
        if train_density > 0:
            domain_gap['detection_density_ratio'] = val_density / train_density
        
        # Confidence distribution shift (simplified)
        train_avg_conf = training_result.precision_recall_data.get('precision', [0])
        val_avg_conf = validation_result.precision_recall_data.get('precision', [0])
        
        if train_avg_conf and val_avg_conf:
            domain_gap['confidence_distribution_shift'] = abs(
                np.mean(train_avg_conf) - np.mean(val_avg_conf)
            )
        
        return domain_gap
    
    def _generate_recommendations(self, 
                                training_metrics: Dict[str, float],
                                validation_metrics: Dict[str, float],
                                performance_drop: Dict[str, float],
                                generalization_score: float) -> List[str]:
        """Generate actionable recommendations based on validation results"""
        
        recommendations = []
        
        # Performance targets check
        if validation_metrics['map_50'] < self.config.target_map_validation:
            recommendations.append(
                f"mAP@0.5 ({validation_metrics['map_50']:.3f}) below target "
                f"({self.config.target_map_validation:.3f}). Consider additional training data or domain adaptation."
            )
        
        # Large performance drops
        map_drop = performance_drop.get('map_50_drop', 0)
        if map_drop > 0.1:  # 10% drop
            recommendations.append(
                f"Significant mAP drop ({map_drop:.1%}). Consider institutional-specific fine-tuning."
            )
        
        precision_drop = performance_drop.get('precision_05_drop', 0)
        if precision_drop > 0.1:
            recommendations.append(
                f"High precision drop ({precision_drop:.1%}). Review false positive patterns."
            )
        
        recall_drop = performance_drop.get('recall_05_drop', 0)
        if recall_drop > 0.1:
            recommendations.append(
                f"High recall drop ({recall_drop:.1%}). Review false negative patterns."
            )
        
        # Generalization assessment
        if generalization_score < 0.8:
            recommendations.append(
                f"Low generalization score ({generalization_score:.3f}). "
                "Consider cross-institutional training or data augmentation."
            )
        elif generalization_score > 0.9:
            recommendations.append(
                f"Excellent generalization ({generalization_score:.3f}). Model generalizes well."
            )
        
        # Default recommendation if no issues
        if not recommendations:
            recommendations.append("Performance meets expectations across institutions.")
        
        return recommendations
    
    def _optimize_confidence_threshold(self, 
                                     dataset_info: Dict[str, Any],
                                     institution: str,
                                     output_dir: str = None) -> float:
        """Optimize confidence threshold for specific institution"""
        
        logger.info(f"Optimizing confidence threshold for {institution}...")
        
        # Test different confidence thresholds
        thresholds = np.arange(0.1, 0.9, 0.05)
        best_threshold = 0.2
        best_f1 = 0
        
        threshold_results = []
        
        for threshold in thresholds:
            # Update threshold
            self.sahi_config.confidence_threshold = threshold
            self.evaluator = ComprehensiveEvaluator(
                model_path=self.model_path,
                sahi_config=self.sahi_config,
                eval_config=self.eval_config,
                experiment_name=f"{self.experiment_name}_threshold_{threshold:.2f}"
            )
            
            # Quick evaluation (subset of data for speed)
            subset_data = {
                'institution': dataset_info['institution'],
                'images': dataset_info['images'][:min(20, len(dataset_info['images']))]  # Use subset
            }
            
            result = self.evaluator.evaluate_dataset(subset_data)
            f1_score = result.f1_per_iou.get(0.5, 0)
            
            threshold_results.append({
                'threshold': threshold,
                'f1_score': f1_score,
                'precision': result.precision_per_iou.get(0.5, 0),
                'recall': result.recall_per_iou.get(0.5, 0)
            })
            
            if f1_score > best_f1:
                best_f1 = f1_score
                best_threshold = threshold
        
        # Save threshold optimization results
        if output_dir:
            threshold_df = pd.DataFrame(threshold_results)
            threshold_csv = Path(output_dir) / f"threshold_optimization_{institution.lower()}.csv"
            threshold_df.to_csv(threshold_csv, index=False)
            
            # Plot optimization curve
            plt.figure(figsize=(10, 6))
            plt.plot(threshold_df['threshold'], threshold_df['f1_score'], 'b-o', label='F1-Score')
            plt.plot(threshold_df['threshold'], threshold_df['precision'], 'r--', label='Precision')
            plt.plot(threshold_df['threshold'], threshold_df['recall'], 'g--', label='Recall')
            plt.axvline(best_threshold, color='red', linestyle=':', 
                       label=f'Optimal: {best_threshold:.3f}')
            plt.xlabel('Confidence Threshold')
            plt.ylabel('Score')
            plt.title(f'Confidence Threshold Optimization - {institution}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plot_path = Path(output_dir) / f"threshold_optimization_{institution.lower()}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(f"Optimal threshold for {institution}: {best_threshold:.3f} (F1: {best_f1:.3f})")
        return best_threshold
    
    def _create_institutional_comparison_report(self, 
                                              training_result,
                                              validation_datasets: Dict[str, Dict[str, Any]],
                                              validation_results: Dict[str, ValidationResult],
                                              output_path: Path):
        """Create comprehensive institutional comparison report"""
        
        # Create summary report
        report_data = {
            'experiment_name': self.experiment_name,
            'model_path': self.model_path,
            'training_institution': self.config.training_institution,
            'validation_institutions': list(validation_results.keys()),
            'training_performance': {
                'map_50': training_result.map_50,
                'map_75': training_result.map_75,
                'total_images': training_result.total_images,
                'total_detections': training_result.total_detections
            },
            'validation_summary': {},
            'overall_generalization_score': 0,
            'key_findings': [],
            'recommendations': []
        }
        
        # Aggregate validation results
        generalization_scores = []
        all_recommendations = []
        
        for institution, result in validation_results.items():
            report_data['validation_summary'][institution] = {
                'performance': result.validation_performance,
                'performance_drop': result.performance_drop,
                'generalization_score': result.generalization_score,
                'domain_gap': result.domain_gap_analysis
            }
            
            generalization_scores.append(result.generalization_score)
            all_recommendations.extend(result.recommendations)
        
        # Calculate overall generalization
        if generalization_scores:
            report_data['overall_generalization_score'] = np.mean(generalization_scores)
        
        # Generate key findings
        avg_map_drop = np.mean([
            result.performance_drop.get('map_50_drop', 0) 
            for result in validation_results.values()
        ])
        
        report_data['key_findings'] = [
            f"Average mAP@0.5 drop across institutions: {avg_map_drop:.1%}",
            f"Overall generalization score: {report_data['overall_generalization_score']:.3f}",
            f"Training institution performance: mAP@0.5 = {training_result.map_50:.3f}"
        ]
        
        # Consolidate recommendations
        unique_recommendations = list(set(all_recommendations))
        report_data['recommendations'] = unique_recommendations
        
        # Save comprehensive report
        report_json = output_path / "institutional_validation_report.json"
        with open(report_json, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        # Create summary CSV
        summary_rows = []
        
        # Training institution row
        summary_rows.append({
            'Institution': self.config.training_institution,
            'Type': 'Training',
            'mAP@0.5': training_result.map_50,
            'mAP@0.75': training_result.map_75,
            'Precision@0.5': training_result.precision_per_iou.get(0.5, 0),
            'Recall@0.5': training_result.recall_per_iou.get(0.5, 0),
            'F1@0.5': training_result.f1_per_iou.get(0.5, 0),
            'Total_Images': training_result.total_images,
            'Total_Detections': training_result.total_detections,
            'Generalization_Score': 1.0  # Reference
        })
        
        # Validation institution rows
        for institution, result in validation_results.items():
            summary_rows.append({
                'Institution': institution,
                'Type': 'Validation',
                'mAP@0.5': result.validation_performance['map_50'],
                'mAP@0.75': result.validation_performance['map_75'],
                'Precision@0.5': result.validation_performance['precision_05'],
                'Recall@0.5': result.validation_performance['recall_05'],
                'F1@0.5': result.validation_performance['f1_05'],
                'Total_Images': 0,  # Would need from original data
                'Total_Detections': 0,  # Would need from original data
                'Generalization_Score': result.generalization_score
            })
        
        summary_df = pd.DataFrame(summary_rows)
        summary_csv = output_path / "institutional_comparison_summary.csv"
        summary_df.to_csv(summary_csv, index=False)
        
        logger.info(f"Institutional comparison report saved to: {output_path}")
        logger.info(f"Overall generalization score: {report_data['overall_generalization_score']:.3f}")


def main():
    """Main function for testing institutional validator"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Institutional Validation Framework")
    parser.add_argument("--model", required=True, help="Path to model weights")
    parser.add_argument("--datasets-config", required=True, help="Path to datasets configuration JSON")
    parser.add_argument("--output", default="./institutional_validation_results", help="Output directory")
    parser.add_argument("--training-institution", default="B_hospital", help="Training institution name")
    
    args = parser.parse_args()
    
    # Load datasets configuration
    with open(args.datasets_config, 'r') as f:
        datasets = json.load(f)
    
    # Create institutional configuration
    config = InstitutionalConfig(
        training_institution=args.training_institution,
        validation_institutions=[inst for inst in datasets.keys() if inst != args.training_institution]
    )
    
    # Initialize validator
    validator = InstitutionalValidator(
        model_path=args.model,
        config=config,
        experiment_name="institutional_validation"
    )
    
    # Run validation
    results = validator.validate_across_institutions(datasets, args.output)
    
    # Print summary
    print("\n🎉 Institutional validation completed!")
    print(f"Training institution: {config.training_institution}")
    
    for institution, result in results.items():
        print(f"\n📊 {institution} Results:")
        print(f"  Generalization score: {result.generalization_score:.3f}")
        print(f"  mAP@0.5: {result.validation_performance['map_50']:.3f}")
        print(f"  Performance drop: {result.performance_drop.get('map_50_drop', 0):.1%}")
        print(f"  Key recommendations: {len(result.recommendations)}")
    
    print(f"\n📁 Results saved to: {args.output}")


if __name__ == "__main__":
    main() 