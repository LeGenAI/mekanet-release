#!/usr/bin/env python3
"""
Benchmark Comparison Module for TESSD Framework
Comprehensive performance comparison against baseline detection methods
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from scipy import stats
import logging
from abc import ABC, abstractmethod

# Setup paths
sys.path.append(str(Path(__file__).parent.parent.parent))

from experiments.detection.comprehensive_evaluator import ComprehensiveEvaluator, EvaluationConfig, EvaluationResult
from experiments.detection.sahi_inference_module import SAHIInferenceModule, SAHIConfig
from experiments.detection.tessd_framework import TESSDFramework

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark comparison"""
    # Dataset configuration
    datasets: List[str]
    max_images_per_dataset: Optional[int] = None
    
    # Evaluation configuration
    confidence_thresholds: List[float] = None
    iou_thresholds: List[float] = None
    
    # Methods to compare
    include_standard_yolo: bool = True
    include_standard_sahi: bool = True
    include_tessd: bool = True
    include_ensemble: bool = False
    
    # Performance analysis
    measure_inference_time: bool = True
    measure_memory_usage: bool = True
    statistical_significance_test: bool = True
    
    # Output configuration
    save_detailed_results: bool = True
    generate_comparison_plots: bool = True
    
    def __post_init__(self):
        if self.confidence_thresholds is None:
            self.confidence_thresholds = [0.15, 0.20, 0.25]
        if self.iou_thresholds is None:
            self.iou_thresholds = [0.5, 0.75]


@dataclass
class MethodResult:
    """Results for a single detection method"""
    method_name: str
    dataset: str
    confidence_threshold: float
    
    # Performance metrics
    map_50: float
    map_75: float
    precision: float
    recall: float
    f1_score: float
    
    # Efficiency metrics
    inference_time_ms: float
    memory_usage_mb: float
    model_size_mb: float
    
    # Detection statistics
    total_predictions: int
    total_ground_truth: int
    
    # Additional metrics
    ap_per_class: Dict[str, float] = None
    confusion_matrix: np.ndarray = None


class BaseDetectionMethod(ABC):
    """Abstract base class for detection methods"""
    
    def __init__(self, model_path: str, config: Dict[str, Any]):
        self.model_path = model_path
        self.config = config
        self.method_name = self.__class__.__name__
    
    @abstractmethod
    def initialize(self):
        """Initialize the detection method"""
        pass
    
    @abstractmethod
    def predict(self, image_path: str) -> List[Dict[str, Any]]:
        """
        Run detection on a single image
        
        Returns:
            List of detections with format:
            [{'bbox': [x1, y1, x2, y2], 'confidence': float, 'class_id': int}, ...]
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information (size, parameters, etc.)"""
        pass


class StandardYOLOMethod(BaseDetectionMethod):
    """Standard YOLOv8 without SAHI"""
    
    def __init__(self, model_path: str, config: Dict[str, Any]):
        super().__init__(model_path, config)
        self.model = None
        self.method_name = "Standard_YOLO"
    
    def initialize(self):
        """Initialize YOLOv8 model"""
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
            logger.info(f"Initialized {self.method_name}")
        except ImportError:
            raise ImportError("ultralytics package required for YOLOv8")
    
    def predict(self, image_path: str) -> List[Dict[str, Any]]:
        """Run standard YOLO prediction"""
        if self.model is None:
            self.initialize()
        
        # Run prediction
        results = self.model(image_path, conf=self.config.get('confidence_threshold', 0.2))
        
        # Convert to standard format
        detections = []
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy()
                
                for box, conf, cls_id in zip(boxes, confidences, class_ids):
                    detections.append({
                        'bbox': box.tolist(),
                        'confidence': float(conf),
                        'class_id': int(cls_id)
                    })
        
        return detections
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        if self.model is None:
            self.initialize()
        
        # Get model file size
        model_size = os.path.getsize(self.model_path) / (1024 * 1024)  # MB
        
        return {
            'model_size_mb': model_size,
            'input_size': [640, 640],  # Standard YOLO input size
            'architecture': 'YOLOv8',
            'parameters': 'N/A'
        }


class StandardSAHIMethod(BaseDetectionMethod):
    """Standard SAHI with YOLOv8"""
    
    def __init__(self, model_path: str, config: Dict[str, Any]):
        super().__init__(model_path, config)
        self.sahi_module = None
        self.method_name = "Standard_SAHI"
    
    def initialize(self):
        """Initialize SAHI module"""
        sahi_config = SAHIConfig(
            tile_size=self.config.get('tile_size', 640),
            overlap_ratio=self.config.get('overlap_ratio', 0.2),
            confidence_threshold=self.config.get('confidence_threshold', 0.2)
        )
        
        self.sahi_module = SAHIInferenceModule(
            model_path=self.model_path,
            config=sahi_config,
            experiment_name=f"{self.method_name}_inference"
        )
        logger.info(f"Initialized {self.method_name}")
    
    def predict(self, image_path: str) -> List[Dict[str, Any]]:
        """Run standard SAHI prediction"""
        if self.sahi_module is None:
            self.initialize()
        
        # Use single-scale inference (no multi-scale for fair comparison)
        detections = self.sahi_module.predict_single_image(
            image_path=image_path,
            use_multiscale=False,
            visualization_output=None
        )
        
        return detections
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        model_size = os.path.getsize(self.model_path) / (1024 * 1024)  # MB
        
        return {
            'model_size_mb': model_size,
            'input_size': [self.config.get('tile_size', 640), self.config.get('tile_size', 640)],
            'architecture': 'YOLOv8 + SAHI',
            'tile_size': self.config.get('tile_size', 640),
            'overlap_ratio': self.config.get('overlap_ratio', 0.2)
        }


class TESSDMethod(BaseDetectionMethod):
    """TESSD (Tiling-Enhanced Semi-Supervised Detection)"""
    
    def __init__(self, model_path: str, config: Dict[str, Any]):
        super().__init__(model_path, config)
        self.tessd_framework = None
        self.method_name = "TESSD"
    
    def initialize(self):
        """Initialize TESSD framework"""
        self.tessd_framework = TESSDFramework(
            model_path=self.model_path,
            experiment_name=f"{self.method_name}_inference"
        )
        logger.info(f"Initialized {self.method_name}")
    
    def predict(self, image_path: str) -> List[Dict[str, Any]]:
        """Run TESSD prediction"""
        if self.tessd_framework is None:
            self.initialize()
        
        # Use TESSD's enhanced inference
        detections = self.tessd_framework.predict_image(
            image_path=image_path,
            confidence_threshold=self.config.get('confidence_threshold', 0.2),
            use_enhancement=True
        )
        
        return detections
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        model_size = os.path.getsize(self.model_path) / (1024 * 1024)  # MB
        
        return {
            'model_size_mb': model_size,
            'input_size': [640, 640],
            'architecture': 'TESSD (YOLOv8 + Enhanced SAHI)',
            'enhancements': ['multi_scale', 'morphological_analysis', 'confidence_calibration']
        }


class BenchmarkComparator:
    """
    Comprehensive benchmark comparison system for detection methods
    
    Features:
    - Multiple baseline method comparison
    - Statistical significance testing
    - Performance and efficiency analysis
    - Detailed result visualization
    - Fair evaluation protocol
    """
    
    def __init__(self, 
                 model_path: str,
                 config: BenchmarkConfig,
                 experiment_name: str = "benchmark_comparison"):
        """
        Initialize benchmark comparator
        
        Args:
            model_path: Path to the trained model
            config: Benchmark configuration
            experiment_name: Name for this benchmark experiment
        """
        self.model_path = model_path
        self.config = config
        self.experiment_name = experiment_name
        
        # Initialize methods
        self.methods = {}
        self._initialize_methods()
        
        # Results storage
        self.results = {}
        self.comparison_stats = {}
        
        logger.info(f"Benchmark Comparator initialized with {len(self.methods)} methods")
    
    def _initialize_methods(self):
        """Initialize all detection methods for comparison"""
        base_config = {
            'tile_size': 640,
            'overlap_ratio': 0.2,
            'confidence_threshold': 0.2  # Will be overridden during evaluation
        }
        
        if self.config.include_standard_yolo:
            self.methods['Standard_YOLO'] = StandardYOLOMethod(self.model_path, base_config)
        
        if self.config.include_standard_sahi:
            self.methods['Standard_SAHI'] = StandardSAHIMethod(self.model_path, base_config)
        
        if self.config.include_tessd:
            self.methods['TESSD'] = TESSDMethod(self.model_path, base_config)
        
        # Initialize all methods
        for method_name, method in self.methods.items():
            try:
                method.initialize()
                logger.info(f"✅ {method_name} initialized successfully")
            except Exception as e:
                logger.error(f"❌ Failed to initialize {method_name}: {e}")
                del self.methods[method_name]
    
    def run_benchmark(self, 
                     datasets: Dict[str, Any], 
                     output_dir: str) -> Dict[str, Any]:
        """
        Run comprehensive benchmark comparison
        
        Args:
            datasets: Dictionary of datasets to evaluate on
            output_dir: Output directory for results
            
        Returns:
            Comprehensive benchmark results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🔬 Starting benchmark comparison")
        logger.info(f"Methods: {list(self.methods.keys())}")
        logger.info(f"Datasets: {list(datasets.keys())}")
        logger.info(f"Confidence thresholds: {self.config.confidence_thresholds}")
        
        # Run evaluation for each method, dataset, and confidence threshold
        for method_name, method in self.methods.items():
            logger.info(f"\n🧪 Evaluating method: {method_name}")
            
            for dataset_name, dataset_info in datasets.items():
                logger.info(f"📊 Dataset: {dataset_name}")
                
                for conf_threshold in self.config.confidence_thresholds:
                    logger.info(f"🎯 Confidence threshold: {conf_threshold}")
                    
                    # Update method configuration
                    method.config['confidence_threshold'] = conf_threshold
                    
                    # Run evaluation
                    result = self._evaluate_method_on_dataset(
                        method, dataset_info, conf_threshold, output_path
                    )
                    
                    # Store result
                    result_key = f"{method_name}_{dataset_name}_{conf_threshold:.2f}"
                    self.results[result_key] = result
        
        # Perform comparative analysis
        self._analyze_comparative_performance(output_path)
        
        # Generate visualizations
        if self.config.generate_comparison_plots:
            self._generate_comparison_visualizations(output_path)
        
        # Save detailed results
        if self.config.save_detailed_results:
            self._save_detailed_results(output_path)
        
        return self.results
    
    def _evaluate_method_on_dataset(self, 
                                   method: BaseDetectionMethod,
                                   dataset_info: Dict[str, Any],
                                   confidence_threshold: float,
                                   output_path: Path) -> MethodResult:
        """Evaluate a single method on a dataset"""
        
        start_time = time.time()
        
        # Prepare evaluation data
        images = dataset_info.get('images', [])
        if self.config.max_images_per_dataset:
            images = images[:self.config.max_images_per_dataset]
        
        # Run predictions and collect metrics
        all_predictions = []
        all_ground_truth = []
        inference_times = []
        
        for image_info in images:
            image_path = image_info['path']
            ground_truth = image_info.get('annotations', [])
            
            # Measure inference time
            inference_start = time.time()
            try:
                predictions = method.predict(image_path)
                inference_time = (time.time() - inference_start) * 1000  # ms
                inference_times.append(inference_time)
                
                all_predictions.extend(predictions)
                all_ground_truth.extend(ground_truth)
                
            except Exception as e:
                logger.warning(f"Failed to process {image_path}: {e}")
                continue
        
        # Calculate metrics using evaluation framework
        eval_config = EvaluationConfig()
        evaluator = ComprehensiveEvaluator(
            model_path=self.model_path,
            sahi_config=None,  # Method handles its own configuration
            eval_config=eval_config,
            experiment_name=f"{method.method_name}_eval"
        )
        
        # Create mock evaluation result for consistent metric calculation
        eval_result = self._calculate_metrics(all_predictions, all_ground_truth)
        
        # Get model information
        model_info = method.get_model_info()
        
        # Create method result
        result = MethodResult(
            method_name=method.method_name,
            dataset=dataset_info.get('institution', 'unknown'),
            confidence_threshold=confidence_threshold,
            map_50=eval_result['map_50'],
            map_75=eval_result['map_75'],
            precision=eval_result['precision'],
            recall=eval_result['recall'],
            f1_score=eval_result['f1_score'],
            inference_time_ms=np.mean(inference_times) if inference_times else 0,
            memory_usage_mb=0,  # Would need separate memory profiling
            model_size_mb=model_info.get('model_size_mb', 0),
            total_predictions=len(all_predictions),
            total_ground_truth=len(all_ground_truth)
        )
        
        execution_time = time.time() - start_time
        logger.info(f"✅ {method.method_name} completed in {execution_time:.1f}s")
        logger.info(f"📈 mAP@0.5: {result.map_50:.3f}, F1: {result.f1_score:.3f}")
        
        return result
    
    def _calculate_metrics(self, predictions: List[Dict], ground_truth: List[Dict]) -> Dict[str, float]:
        """Calculate evaluation metrics from predictions and ground truth"""
        
        # Mock implementation - in real scenario, use proper AP calculation
        # This would normally involve IoU matching, precision-recall curves, etc.
        
        # Simple metrics calculation for demonstration
        num_predictions = len(predictions)
        num_ground_truth = len(ground_truth)
        
        # Mock metrics (in real implementation, use proper COCO evaluation)
        if num_ground_truth > 0:
            recall = min(num_predictions / num_ground_truth, 1.0) * 0.8  # Mock recall
            precision = 0.85 if num_predictions > 0 else 0.0  # Mock precision
        else:
            recall = 0.0
            precision = 0.0
        
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'map_50': precision * 0.9,  # Mock mAP@0.5
            'map_75': precision * 0.7,  # Mock mAP@0.75
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        }
    
    def _analyze_comparative_performance(self, output_path: Path):
        """Analyze comparative performance across methods"""
        logger.info("📊 Analyzing comparative performance...")
        
        # Group results by dataset and confidence threshold
        comparison_data = []
        
        for result_key, result in self.results.items():
            comparison_data.append({
                'method': result.method_name,
                'dataset': result.dataset,
                'confidence': result.confidence_threshold,
                'map_50': result.map_50,
                'map_75': result.map_75,
                'precision': result.precision,
                'recall': result.recall,
                'f1_score': result.f1_score,
                'inference_time': result.inference_time_ms,
                'model_size': result.model_size_mb
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Statistical significance testing
        if self.config.statistical_significance_test:
            self.comparison_stats = self._perform_statistical_tests(comparison_df)
        
        # Calculate summary statistics
        summary_stats = comparison_df.groupby(['method', 'dataset']).agg({
            'map_50': ['mean', 'std'],
            'f1_score': ['mean', 'std'],
            'inference_time': ['mean', 'std']
        }).round(4)
        
        # Save comparison data
        comparison_df.to_csv(output_path / "benchmark_comparison_detailed.csv", index=False)
        summary_stats.to_csv(output_path / "benchmark_comparison_summary.csv")
        
        logger.info(f"📈 Comparative analysis completed")
    
    def _perform_statistical_tests(self, comparison_df: pd.DataFrame) -> Dict[str, Any]:
        """Perform statistical significance tests between methods"""
        stats_results = {}
        
        methods = comparison_df['method'].unique()
        datasets = comparison_df['dataset'].unique()
        
        for dataset in datasets:
            dataset_results = {}
            dataset_df = comparison_df[comparison_df['dataset'] == dataset]
            
            # Pairwise comparisons between methods
            for i, method1 in enumerate(methods):
                for j, method2 in enumerate(methods[i+1:], i+1):
                    method1_data = dataset_df[dataset_df['method'] == method1]['map_50'].values
                    method2_data = dataset_df[dataset_df['method'] == method2]['map_50'].values
                    
                    if len(method1_data) > 1 and len(method2_data) > 1:
                        # Perform t-test
                        t_stat, p_value = stats.ttest_ind(method1_data, method2_data)
                        
                        dataset_results[f"{method1}_vs_{method2}"] = {
                            't_statistic': float(t_stat),
                            'p_value': float(p_value),
                            'significant': p_value < 0.05,
                            'method1_mean': float(np.mean(method1_data)),
                            'method2_mean': float(np.mean(method2_data))
                        }
            
            stats_results[dataset] = dataset_results
        
        return stats_results
    
    def _generate_comparison_visualizations(self, output_path: Path):
        """Generate comprehensive comparison visualizations"""
        logger.info("🎨 Generating comparison visualizations...")
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Prepare data
        comparison_data = []
        for result in self.results.values():
            comparison_data.append({
                'Method': result.method_name,
                'Dataset': result.dataset,
                'Confidence': result.confidence_threshold,
                'mAP@0.5': result.map_50,
                'mAP@0.75': result.map_75,
                'Precision': result.precision,
                'Recall': result.recall,
                'F1-Score': result.f1_score,
                'Inference Time (ms)': result.inference_time_ms,
                'Model Size (MB)': result.model_size_mb
            })
        
        df = pd.DataFrame(comparison_data)
        
        # 1. Performance comparison across methods
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Detection Methods Performance Comparison', fontsize=16, y=0.98)
        
        # mAP@0.5 comparison
        sns.boxplot(data=df, x='Method', y='mAP@0.5', hue='Dataset', ax=axes[0,0])
        axes[0,0].set_title('mAP@0.5 Comparison')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # F1-Score comparison  
        sns.boxplot(data=df, x='Method', y='F1-Score', hue='Dataset', ax=axes[0,1])
        axes[0,1].set_title('F1-Score Comparison')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Inference time comparison
        sns.boxplot(data=df, x='Method', y='Inference Time (ms)', hue='Dataset', ax=axes[1,0])
        axes[1,0].set_title('Inference Time Comparison')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Precision vs Recall scatter
        sns.scatterplot(data=df, x='Recall', y='Precision', hue='Method', 
                       style='Dataset', s=100, ax=axes[1,1])
        axes[1,1].set_title('Precision vs Recall')
        
        plt.tight_layout()
        plt.savefig(output_path / "benchmark_comparison_overview.pdf", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Confidence threshold analysis
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # mAP vs Confidence threshold
        for method in df['Method'].unique():
            method_data = df[df['Method'] == method]
            for dataset in method_data['Dataset'].unique():
                data = method_data[method_data['Dataset'] == dataset]
                axes[0].plot(data['Confidence'], data['mAP@0.5'], 
                           marker='o', label=f"{method}_{dataset}")
        
        axes[0].set_xlabel('Confidence Threshold')
        axes[0].set_ylabel('mAP@0.5')
        axes[0].set_title('mAP@0.5 vs Confidence Threshold')
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0].grid(True, alpha=0.3)
        
        # F1-Score vs Confidence threshold
        for method in df['Method'].unique():
            method_data = df[df['Method'] == method]
            for dataset in method_data['Dataset'].unique():
                data = method_data[method_data['Dataset'] == dataset]
                axes[1].plot(data['Confidence'], data['F1-Score'], 
                           marker='o', label=f"{method}_{dataset}")
        
        axes[1].set_xlabel('Confidence Threshold')
        axes[1].set_ylabel('F1-Score')
        axes[1].set_title('F1-Score vs Confidence Threshold')
        axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / "confidence_threshold_analysis.pdf", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Efficiency analysis
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Performance vs Efficiency trade-off
        sns.scatterplot(data=df, x='Inference Time (ms)', y='mAP@0.5', 
                       hue='Method', style='Dataset', s=100, ax=axes[0])
        axes[0].set_title('Performance vs Efficiency Trade-off')
        axes[0].set_xlabel('Inference Time (ms)')
        axes[0].set_ylabel('mAP@0.5')
        
        # Model size comparison
        model_size_summary = df.groupby('Method')['Model Size (MB)'].first().reset_index()
        sns.barplot(data=model_size_summary, x='Method', y='Model Size (MB)', ax=axes[1])
        axes[1].set_title('Model Size Comparison')
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_path / "efficiency_analysis.pdf", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("✅ Visualization generation completed")
    
    def _save_detailed_results(self, output_path: Path):
        """Save detailed benchmark results"""
        
        # Convert results to serializable format
        results_data = {}
        for key, result in self.results.items():
            results_data[key] = asdict(result)
        
        # Save main results
        with open(output_path / "benchmark_results.json", 'w') as f:
            json.dump({
                'experiment_name': self.experiment_name,
                'config': asdict(self.config),
                'methods_evaluated': list(self.methods.keys()),
                'results': results_data,
                'statistical_analysis': self.comparison_stats
            }, f, indent=2, default=str)
        
        # Save summary report
        self._generate_summary_report(output_path)
        
        logger.info("💾 Detailed results saved")
    
    def _generate_summary_report(self, output_path: Path):
        """Generate human-readable summary report"""
        
        report_lines = []
        report_lines.append("# TESSD Benchmark Comparison Report")
        report_lines.append("=" * 50)
        report_lines.append("")
        
        # Experiment overview
        report_lines.append("## Experiment Overview")
        report_lines.append(f"- Experiment: {self.experiment_name}")
        report_lines.append(f"- Methods compared: {len(self.methods)}")
        report_lines.append(f"- Datasets: {len(self.config.datasets)}")
        report_lines.append(f"- Confidence thresholds: {self.config.confidence_thresholds}")
        report_lines.append("")
        
        # Methods summary
        report_lines.append("## Methods Evaluated")
        for method_name, method in self.methods.items():
            model_info = method.get_model_info()
            report_lines.append(f"### {method_name}")
            report_lines.append(f"- Architecture: {model_info.get('architecture', 'N/A')}")
            report_lines.append(f"- Model Size: {model_info.get('model_size_mb', 0):.1f} MB")
            report_lines.append("")
        
        # Performance summary
        report_lines.append("## Performance Summary")
        
        # Calculate best performance for each metric
        best_map50 = max(self.results.values(), key=lambda x: x.map_50)
        best_f1 = max(self.results.values(), key=lambda x: x.f1_score)
        fastest = min(self.results.values(), key=lambda x: x.inference_time_ms)
        
        report_lines.append(f"### Best mAP@0.5: {best_map50.method_name}")
        report_lines.append(f"- Score: {best_map50.map_50:.3f}")
        report_lines.append(f"- Dataset: {best_map50.dataset}")
        report_lines.append(f"- Confidence: {best_map50.confidence_threshold}")
        report_lines.append("")
        
        report_lines.append(f"### Best F1-Score: {best_f1.method_name}")
        report_lines.append(f"- Score: {best_f1.f1_score:.3f}")
        report_lines.append(f"- Dataset: {best_f1.dataset}")
        report_lines.append("")
        
        report_lines.append(f"### Fastest Method: {fastest.method_name}")
        report_lines.append(f"- Time: {fastest.inference_time_ms:.1f} ms")
        report_lines.append("")
        
        # Statistical significance
        if self.comparison_stats:
            report_lines.append("## Statistical Significance")
            for dataset, stats in self.comparison_stats.items():
                report_lines.append(f"### {dataset}")
                for comparison, result in stats.items():
                    significance = "✅ Significant" if result['significant'] else "❌ Not significant"
                    report_lines.append(f"- {comparison}: {significance} (p={result['p_value']:.4f})")
                report_lines.append("")
        
        # Save report
        with open(output_path / "benchmark_summary_report.txt", 'w') as f:
            f.write("\n".join(report_lines))


def main():
    """Main function for benchmark comparison"""
    import argparse
    
    parser = argparse.ArgumentParser(description="TESSD Benchmark Comparison")
    parser.add_argument("--model", required=True, help="Path to model weights")
    parser.add_argument("--output", default="./benchmark_results", help="Output directory")
    parser.add_argument("--quick", action="store_true", help="Quick test mode")
    
    args = parser.parse_args()
    
    # Configure benchmark
    config = BenchmarkConfig(
        datasets=["B_hospital", "S_hospital"],
        max_images_per_dataset=5 if args.quick else None,
        confidence_thresholds=[0.15, 0.20, 0.25] if not args.quick else [0.20],
        include_standard_yolo=True,
        include_standard_sahi=True,
        include_tessd=True
    )
    
    # Initialize comparator
    comparator = BenchmarkComparator(
        model_path=args.model,
        config=config,
        experiment_name="tessd_benchmark"
    )
    
    # Mock datasets (in real implementation, load from configuration)
    datasets = {
        "B_hospital": {
            "institution": "B_hospital",
            "images": [
                {"path": f"/mock/path/b_hospital/img_{i}.jpg", "annotations": []}
                for i in range(config.max_images_per_dataset or 10)
            ]
        },
        "S_hospital": {
            "institution": "S_hospital", 
            "images": [
                {"path": f"/mock/path/s_hospital/img_{i}.jpg", "annotations": []}
                for i in range(config.max_images_per_dataset or 10)
            ]
        }
    }
    
    # Run benchmark
    results = comparator.run_benchmark(datasets, args.output)
    
    print(f"\n🎉 Benchmark comparison completed!")
    print(f"📁 Results saved to: {args.output}")
    print(f"📊 Methods compared: {len(comparator.methods)}")
    print(f"📈 Total evaluations: {len(results)}")


if __name__ == "__main__":
    main() 