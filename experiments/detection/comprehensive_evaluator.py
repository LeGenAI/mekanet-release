#!/usr/bin/env python3
"""
Comprehensive Evaluation Module for TESSD Framework
Advanced metrics calculation including mAP, precision, recall, F1-score across institutional datasets
"""

import cv2
import numpy as np
import pandas as pd
import torch
import logging
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, average_precision_score
import warnings
warnings.filterwarnings('ignore')

# Setup paths
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from experiments.detection.tessd_framework import TESSDFramework
from experiments.detection.sahi_inference_module import SAHIInferenceModule, SAHIConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """Configuration for evaluation parameters"""
    iou_thresholds: List[float] = None
    confidence_thresholds: List[float] = None
    area_ranges: List[Tuple[int, int]] = None
    max_detections: List[int] = None
    
    # Institutional validation settings
    institutions: List[str] = None
    cross_institutional: bool = True
    
    # Output settings
    save_detailed_results: bool = True
    save_visualizations: bool = True
    save_pr_curves: bool = True
    
    def __post_init__(self):
        if self.iou_thresholds is None:
            # COCO-style IoU thresholds from 0.5 to 0.95 with step 0.05
            self.iou_thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
        
        if self.confidence_thresholds is None:
            # Confidence thresholds for precision-recall analysis
            self.confidence_thresholds = np.arange(0.0, 1.01, 0.01).tolist()
        
        if self.area_ranges is None:
            # Area ranges for different object sizes
            self.area_ranges = [
                (0, 32**2),      # small
                (32**2, 96**2),  # medium  
                (96**2, float('inf'))  # large
            ]
        
        if self.max_detections is None:
            self.max_detections = [1, 10, 100]
        
        if self.institutions is None:
            self.institutions = ['B_hospital', 'S_hospital']


@dataclass
class DetectionMatch:
    """Represents a detection match with ground truth"""
    detection_id: int
    gt_id: int
    iou: float
    confidence: float
    area: float
    matched: bool


@dataclass
class EvaluationResult:
    """Results from comprehensive evaluation"""
    institution: str
    total_images: int
    total_gt_objects: int
    total_detections: int
    
    # Per-IoU threshold metrics
    ap_per_iou: Dict[float, float]
    precision_per_iou: Dict[float, float] 
    recall_per_iou: Dict[float, float]
    f1_per_iou: Dict[float, float]
    
    # Summary metrics
    map_50: float  # mAP@0.5
    map_75: float  # mAP@0.75
    map_50_95: float  # mAP@0.5:0.95
    
    # Per-area metrics
    ap_per_area: Dict[str, float]
    
    # Confidence analysis
    optimal_confidence: float
    precision_recall_data: Dict[str, Any]
    
    # Processing statistics
    total_processing_time: float
    average_processing_time: float


class ComprehensiveEvaluator:
    """
    Comprehensive evaluation module for TESSD framework
    
    Features:
    - COCO-style mAP calculation
    - Precision, Recall, F1-score across IoU thresholds
    - Cross-institutional validation
    - Size-based analysis
    - Confidence threshold optimization
    - Detailed performance breakdowns
    """
    
    def __init__(self, 
                 model_path: str,
                 sahi_config: SAHIConfig = None,
                 eval_config: EvaluationConfig = None,
                 experiment_name: str = "comprehensive_evaluation"):
        """
        Initialize comprehensive evaluator
        
        Args:
            model_path: Path to trained model
            sahi_config: SAHI inference configuration
            eval_config: Evaluation configuration
            experiment_name: Name for experiment tracking
        """
        self.model_path = model_path
        self.sahi_config = sahi_config if sahi_config is not None else SAHIConfig()
        self.eval_config = eval_config if eval_config is not None else EvaluationConfig()
        self.experiment_name = experiment_name
        
        # Initialize inference module
        self.inference_module = SAHIInferenceModule(
            model_path=model_path,
            config=self.sahi_config,
            experiment_name=experiment_name
        )
        
        # Initialize TESSD framework for direct access
        self.tessd = self.inference_module.tessd
        
        logger.info(f"Comprehensive Evaluator initialized: {experiment_name}")
        logger.info(f"IoU thresholds: {self.eval_config.iou_thresholds}")
        logger.info(f"Institutions: {self.eval_config.institutions}")
    
    def evaluate_dataset(self, 
                        dataset_info: Dict[str, Any],
                        output_dir: str = None) -> EvaluationResult:
        """
        Evaluate model on a dataset
        
        Args:
            dataset_info: Dictionary containing image paths and annotations
                         Format: {
                             'institution': 'B_hospital',
                             'images': [{'path': '...', 'annotations': [...]}]
                         }
            output_dir: Output directory for results
            
        Returns:
            EvaluationResult object
        """
        institution = dataset_info['institution']
        images_data = dataset_info['images']
        
        logger.info(f"Evaluating {institution} dataset with {len(images_data)} images")
        
        if output_dir:
            output_path = Path(output_dir) / institution
            output_path.mkdir(parents=True, exist_ok=True)
        
        # Collect all predictions and ground truths
        all_predictions = []
        all_ground_truths = []
        total_processing_time = 0
        
        for i, image_data in enumerate(images_data):
            image_path = image_data['path']
            gt_annotations = image_data['annotations']
            
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                logger.warning(f"Failed to load image: {image_path}")
                continue
            
            # Get predictions
            start_time = time.time()
            inference_result = self.inference_module.predict_single_image(
                image, 
                image_id=f"{institution}_{i}",
                save_visualization=self.eval_config.save_visualizations,
                output_dir=str(output_path) if output_dir else None
            )
            processing_time = time.time() - start_time
            total_processing_time += processing_time
            
            # Process predictions
            predictions = self._format_predictions(inference_result.detections, i)
            ground_truths = self._format_ground_truths(gt_annotations, i, image.shape[:2])
            
            all_predictions.extend(predictions)
            all_ground_truths.extend(ground_truths)
        
        # Calculate comprehensive metrics
        evaluation_result = self._calculate_comprehensive_metrics(
            all_predictions, 
            all_ground_truths,
            institution,
            len(images_data),
            total_processing_time
        )
        
        # Save detailed results
        if output_dir and self.eval_config.save_detailed_results:
            self._save_detailed_results(evaluation_result, output_path)
        
        return evaluation_result
    
    def _format_predictions(self, detections: List[Dict[str, Any]], image_id: int) -> List[Dict[str, Any]]:
        """Format predictions for evaluation"""
        predictions = []
        
        for i, det in enumerate(detections):
            bbox = det['bbox']
            
            predictions.append({
                'image_id': image_id,
                'detection_id': f"{image_id}_{i}",
                'bbox': [bbox['x1'], bbox['y1'], bbox['x2'] - bbox['x1'], bbox['y2'] - bbox['y1']],  # x, y, w, h
                'area': (bbox['x2'] - bbox['x1']) * (bbox['y2'] - bbox['y1']),
                'confidence': det['score'],
                'category_id': 1  # Megakaryocyte class
            })
        
        return predictions
    
    def _format_ground_truths(self, annotations: List[Dict[str, Any]], image_id: int, image_shape: Tuple[int, int]) -> List[Dict[str, Any]]:
        """Format ground truth annotations for evaluation"""
        ground_truths = []
        height, width = image_shape
        
        for i, ann in enumerate(annotations):
            # Convert from YOLO format if needed
            if 'x_center' in ann:
                # YOLO format: x_center, y_center, width, height (normalized)
                x_center = ann['x_center'] * width
                y_center = ann['y_center'] * height
                w = ann['width'] * width
                h = ann['height'] * height
                x = x_center - w / 2
                y = y_center - h / 2
                bbox = [x, y, w, h]
            else:
                # Assume absolute coordinates
                bbox = ann['bbox']  # [x, y, w, h]
            
            ground_truths.append({
                'image_id': image_id,
                'gt_id': f"{image_id}_{i}",
                'bbox': bbox,
                'area': bbox[2] * bbox[3],
                'category_id': ann.get('category_id', 1),
                'iscrowd': ann.get('iscrowd', 0),
                'ignore': ann.get('ignore', False)
            })
        
        return ground_truths
    
    def _calculate_comprehensive_metrics(self, 
                                       predictions: List[Dict[str, Any]],
                                       ground_truths: List[Dict[str, Any]],
                                       institution: str,
                                       total_images: int,
                                       total_processing_time: float) -> EvaluationResult:
        """Calculate comprehensive evaluation metrics"""
        
        logger.info(f"Calculating metrics for {len(predictions)} predictions and {len(ground_truths)} ground truths")
        
        # Initialize results storage
        ap_per_iou = {}
        precision_per_iou = {}
        recall_per_iou = {}
        f1_per_iou = {}
        
        # Calculate metrics for each IoU threshold
        for iou_threshold in self.eval_config.iou_thresholds:
            matches = self._match_predictions_to_gt(predictions, ground_truths, iou_threshold)
            
            # Calculate precision, recall, F1
            tp = sum(1 for match in matches if match.matched)
            fp = len(predictions) - tp
            fn = len(ground_truths) - tp
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            precision_per_iou[iou_threshold] = precision
            recall_per_iou[iou_threshold] = recall
            f1_per_iou[iou_threshold] = f1
            
            # Calculate Average Precision (AP)
            ap = self._calculate_average_precision(matches, len(ground_truths))
            ap_per_iou[iou_threshold] = ap
        
        # Calculate summary metrics
        map_50 = ap_per_iou.get(0.50, 0)
        map_75 = ap_per_iou.get(0.75, 0)
        map_50_95 = np.mean(list(ap_per_iou.values())) if ap_per_iou else 0
        
        # Calculate size-based metrics
        ap_per_area = self._calculate_area_based_metrics(predictions, ground_truths)
        
        # Find optimal confidence threshold
        optimal_confidence, pr_data = self._analyze_confidence_thresholds(predictions, ground_truths)
        
        return EvaluationResult(
            institution=institution,
            total_images=total_images,
            total_gt_objects=len(ground_truths),
            total_detections=len(predictions),
            ap_per_iou=ap_per_iou,
            precision_per_iou=precision_per_iou,
            recall_per_iou=recall_per_iou,
            f1_per_iou=f1_per_iou,
            map_50=map_50,
            map_75=map_75,
            map_50_95=map_50_95,
            ap_per_area=ap_per_area,
            optimal_confidence=optimal_confidence,
            precision_recall_data=pr_data,
            total_processing_time=total_processing_time,
            average_processing_time=total_processing_time / total_images if total_images > 0 else 0
        )
    
    def _match_predictions_to_gt(self, 
                               predictions: List[Dict[str, Any]], 
                               ground_truths: List[Dict[str, Any]], 
                               iou_threshold: float) -> List[DetectionMatch]:
        """
        Match predictions to ground truth using IoU threshold
        
        Args:
            predictions: List of predictions
            ground_truths: List of ground truth annotations
            iou_threshold: IoU threshold for matching
            
        Returns:
            List of DetectionMatch objects
        """
        matches = []
        
        # Group by image
        pred_by_image = {}
        gt_by_image = {}
        
        for pred in predictions:
            img_id = pred['image_id']
            if img_id not in pred_by_image:
                pred_by_image[img_id] = []
            pred_by_image[img_id].append(pred)
        
        for gt in ground_truths:
            img_id = gt['image_id']
            if img_id not in gt_by_image:
                gt_by_image[img_id] = []
            gt_by_image[img_id].append(gt)
        
        # Match predictions to ground truth for each image
        for img_id in pred_by_image.keys():
            img_predictions = pred_by_image.get(img_id, [])
            img_ground_truths = gt_by_image.get(img_id, [])
            
            # Sort predictions by confidence (highest first)
            img_predictions = sorted(img_predictions, key=lambda x: x['confidence'], reverse=True)
            
            # Track which ground truths have been matched
            gt_matched = [False] * len(img_ground_truths)
            
            for pred in img_predictions:
                best_iou = 0
                best_gt_idx = -1
                
                # Find best matching ground truth
                for gt_idx, gt in enumerate(img_ground_truths):
                    if gt_matched[gt_idx] or gt.get('ignore', False):
                        continue
                    
                    iou = self._calculate_bbox_iou(pred['bbox'], gt['bbox'])
                    
                    if iou > best_iou and iou >= iou_threshold:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                # Create match
                if best_gt_idx >= 0:
                    gt_matched[best_gt_idx] = True
                    matched = True
                    gt_id = img_ground_truths[best_gt_idx]['gt_id']
                else:
                    matched = False
                    gt_id = None
                    best_iou = 0
                
                matches.append(DetectionMatch(
                    detection_id=pred['detection_id'],
                    gt_id=gt_id,
                    iou=best_iou,
                    confidence=pred['confidence'],
                    area=pred['area'],
                    matched=matched
                ))
        
        return matches
    
    def _calculate_bbox_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """
        Calculate IoU between two bounding boxes
        
        Args:
            bbox1: [x, y, w, h]
            bbox2: [x, y, w, h]
            
        Returns:
            IoU value
        """
        # Convert to [x1, y1, x2, y2]
        x1_1, y1_1, w1, h1 = bbox1
        x2_1, y2_1 = x1_1 + w1, y1_1 + h1
        
        x1_2, y1_2, w2, h2 = bbox2
        x2_2, y2_2 = x1_2 + w2, y1_2 + h2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_average_precision(self, matches: List[DetectionMatch], num_gt: int) -> float:
        """
        Calculate Average Precision (AP)
        
        Args:
            matches: List of detection matches
            num_gt: Number of ground truth objects
            
        Returns:
            Average Precision value
        """
        if not matches or num_gt == 0:
            return 0.0
        
        # Sort by confidence
        matches = sorted(matches, key=lambda x: x.confidence, reverse=True)
        
        # Calculate precision and recall at each threshold
        tp = 0
        fp = 0
        precisions = []
        recalls = []
        
        for match in matches:
            if match.matched:
                tp += 1
            else:
                fp += 1
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / num_gt if num_gt > 0 else 0
            
            precisions.append(precision)
            recalls.append(recall)
        
        # Calculate AP using precision-recall curve
        if not precisions or not recalls:
            return 0.0
        
        # Add endpoints
        recalls = [0] + recalls + [1]
        precisions = [0] + precisions + [0]
        
        # Make precision monotonically decreasing
        for i in range(len(precisions) - 2, -1, -1):
            precisions[i] = max(precisions[i], precisions[i + 1])
        
        # Calculate area under curve
        ap = 0
        for i in range(1, len(recalls)):
            ap += (recalls[i] - recalls[i - 1]) * precisions[i]
        
        return ap
    
    def _calculate_area_based_metrics(self, 
                                    predictions: List[Dict[str, Any]], 
                                    ground_truths: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate AP for different area ranges"""
        area_names = ['small', 'medium', 'large']
        ap_per_area = {}
        
        for i, (min_area, max_area) in enumerate(self.eval_config.area_ranges):
            area_name = area_names[i]
            
            # Filter predictions and ground truths by area
            area_predictions = [p for p in predictions if min_area <= p['area'] < max_area]
            area_ground_truths = [gt for gt in ground_truths if min_area <= gt['area'] < max_area]
            
            if not area_ground_truths:
                ap_per_area[area_name] = 0.0
                continue
            
            # Calculate AP at IoU=0.5 for this area range
            matches = self._match_predictions_to_gt(area_predictions, area_ground_truths, 0.5)
            ap = self._calculate_average_precision(matches, len(area_ground_truths))
            ap_per_area[area_name] = ap
        
        return ap_per_area
    
    def _analyze_confidence_thresholds(self, 
                                     predictions: List[Dict[str, Any]], 
                                     ground_truths: List[Dict[str, Any]]) -> Tuple[float, Dict[str, Any]]:
        """
        Analyze different confidence thresholds to find optimal
        
        Returns:
            Tuple of (optimal_confidence, precision_recall_data)
        """
        confidences = [p['confidence'] for p in predictions]
        
        if not confidences:
            return 0.0, {'precision': [], 'recall': [], 'thresholds': []}
        
        # Create binary ground truth labels
        # For each prediction, check if it matches any ground truth at IoU=0.5
        matches = self._match_predictions_to_gt(predictions, ground_truths, 0.5)
        y_true = [1 if match.matched else 0 for match in matches]
        y_scores = [match.confidence for match in matches]
        
        if not y_true or all(y == 0 for y in y_true):
            return 0.0, {'precision': [], 'recall': [], 'thresholds': []}
        
        # Calculate precision-recall curve
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
        
        # Find optimal confidence threshold (maximize F1-score)
        f1_scores = []
        for p, r in zip(precisions[:-1], recalls[:-1]):  # Exclude last point
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
            f1_scores.append(f1)
        
        if f1_scores:
            best_f1_idx = np.argmax(f1_scores)
            optimal_confidence = thresholds[best_f1_idx]
        else:
            optimal_confidence = 0.5  # Default
        
        return optimal_confidence, {
            'precision': precisions.tolist(),
            'recall': recalls.tolist(),
            'thresholds': thresholds.tolist(),
            'f1_scores': f1_scores,
            'optimal_threshold': optimal_confidence
        }
    
    def compare_institutions(self, 
                           datasets: List[Dict[str, Any]], 
                           output_dir: str = None) -> Dict[str, EvaluationResult]:
        """
        Compare performance across multiple institutions
        
        Args:
            datasets: List of dataset info dictionaries
            output_dir: Output directory for comparison results
            
        Returns:
            Dictionary of institution -> EvaluationResult
        """
        logger.info(f"Comparing performance across {len(datasets)} institutions")
        
        results = {}
        
        # Evaluate each institution
        for dataset_info in datasets:
            institution = dataset_info['institution']
            logger.info(f"Evaluating {institution}...")
            
            result = self.evaluate_dataset(dataset_info, output_dir)
            results[institution] = result
        
        # Create comparison visualizations
        if output_dir and len(results) > 1:
            self._create_comparison_visualizations(results, output_dir)
        
        return results
    
    def _create_comparison_visualizations(self, 
                                        results: Dict[str, EvaluationResult], 
                                        output_dir: str):
        """Create visualization comparing institutional performance"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        institutions = list(results.keys())
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Cross-Institutional Performance Comparison', fontsize=16)
        
        # mAP comparison
        ax = axes[0, 0]
        map_50_values = [results[inst].map_50 for inst in institutions]
        map_75_values = [results[inst].map_75 for inst in institutions]
        map_50_95_values = [results[inst].map_50_95 for inst in institutions]
        
        x = np.arange(len(institutions))
        width = 0.25
        
        ax.bar(x - width, map_50_values, width, label='mAP@0.5', alpha=0.8)
        ax.bar(x, map_75_values, width, label='mAP@0.75', alpha=0.8)
        ax.bar(x + width, map_50_95_values, width, label='mAP@0.5:0.95', alpha=0.8)
        
        ax.set_xlabel('Institution')
        ax.set_ylabel('mAP')
        ax.set_title('Mean Average Precision Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(institutions)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Precision-Recall at IoU=0.5
        ax = axes[0, 1]
        precision_values = [results[inst].precision_per_iou.get(0.5, 0) for inst in institutions]
        recall_values = [results[inst].recall_per_iou.get(0.5, 0) for inst in institutions]
        
        ax.bar(x - width/2, precision_values, width, label='Precision', alpha=0.8)
        ax.bar(x + width/2, recall_values, width, label='Recall', alpha=0.8)
        
        ax.set_xlabel('Institution')
        ax.set_ylabel('Score')
        ax.set_title('Precision & Recall @ IoU=0.5')
        ax.set_xticks(x)
        ax.set_xticklabels(institutions)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # F1-Score across IoU thresholds
        ax = axes[0, 2]
        iou_thresholds = sorted(results[institutions[0]].f1_per_iou.keys())
        
        for inst in institutions:
            f1_values = [results[inst].f1_per_iou.get(iou, 0) for iou in iou_thresholds]
            ax.plot(iou_thresholds, f1_values, 'o-', label=inst, linewidth=2)
        
        ax.set_xlabel('IoU Threshold')
        ax.set_ylabel('F1-Score')
        ax.set_title('F1-Score vs IoU Threshold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Detection counts
        ax = axes[1, 0]
        total_gt = [results[inst].total_gt_objects for inst in institutions]
        total_det = [results[inst].total_detections for inst in institutions]
        
        ax.bar(x - width/2, total_gt, width, label='Ground Truth', alpha=0.8)
        ax.bar(x + width/2, total_det, width, label='Detections', alpha=0.8)
        
        ax.set_xlabel('Institution')
        ax.set_ylabel('Count')
        ax.set_title('Detection Counts')
        ax.set_xticks(x)
        ax.set_xticklabels(institutions)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Processing time
        ax = axes[1, 1]
        processing_times = [results[inst].average_processing_time for inst in institutions]
        
        ax.bar(institutions, processing_times, alpha=0.8, color='skyblue')
        ax.set_xlabel('Institution')
        ax.set_ylabel('Average Processing Time (s)')
        ax.set_title('Processing Performance')
        ax.grid(True, alpha=0.3)
        
        # Area-based performance
        ax = axes[1, 2]
        area_types = ['small', 'medium', 'large']
        
        for inst in institutions:
            area_aps = [results[inst].ap_per_area.get(area, 0) for area in area_types]
            ax.plot(area_types, area_aps, 'o-', label=inst, linewidth=2)
        
        ax.set_xlabel('Object Size')
        ax.set_ylabel('Average Precision')
        ax.set_title('Performance by Object Size')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save comparison plot
        comparison_path = output_path / "institutional_comparison.png"
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save comparison summary CSV
        summary_data = []
        for inst in institutions:
            result = results[inst]
            summary_data.append({
                'Institution': inst,
                'Total_Images': result.total_images,
                'Total_GT_Objects': result.total_gt_objects,
                'Total_Detections': result.total_detections,
                'mAP@0.5': result.map_50,
                'mAP@0.75': result.map_75,
                'mAP@0.5:0.95': result.map_50_95,
                'Precision@0.5': result.precision_per_iou.get(0.5, 0),
                'Recall@0.5': result.recall_per_iou.get(0.5, 0),
                'F1@0.5': result.f1_per_iou.get(0.5, 0),
                'Optimal_Confidence': result.optimal_confidence,
                'Avg_Processing_Time': result.average_processing_time,
                'AP_Small': result.ap_per_area.get('small', 0),
                'AP_Medium': result.ap_per_area.get('medium', 0),
                'AP_Large': result.ap_per_area.get('large', 0)
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_csv_path = output_path / "institutional_comparison_summary.csv"
        summary_df.to_csv(summary_csv_path, index=False)
        
        logger.info(f"Comparison visualizations saved to: {output_path}")
    
    def _save_detailed_results(self, result: EvaluationResult, output_path: Path):
        """Save detailed evaluation results"""
        
        # Save metrics summary
        summary = {
            'institution': result.institution,
            'total_images': result.total_images,
            'total_gt_objects': result.total_gt_objects,
            'total_detections': result.total_detections,
            'map_50': result.map_50,
            'map_75': result.map_75,
            'map_50_95': result.map_50_95,
            'optimal_confidence': result.optimal_confidence,
            'total_processing_time': result.total_processing_time,
            'average_processing_time': result.average_processing_time,
            'ap_per_iou': result.ap_per_iou,
            'precision_per_iou': result.precision_per_iou,
            'recall_per_iou': result.recall_per_iou,
            'f1_per_iou': result.f1_per_iou,
            'ap_per_area': result.ap_per_area
        }
        
        # Save as JSON
        json_path = output_path / "evaluation_results.json"
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Save precision-recall curve data
        if result.precision_recall_data and self.eval_config.save_pr_curves:
            pr_df = pd.DataFrame(result.precision_recall_data)
            pr_csv_path = output_path / "precision_recall_curve.csv"
            pr_df.to_csv(pr_csv_path, index=False)
            
            # Create PR curve plot
            self._plot_precision_recall_curve(result.precision_recall_data, output_path)
        
        logger.info(f"Detailed results saved to: {output_path}")
    
    def _plot_precision_recall_curve(self, pr_data: Dict[str, Any], output_path: Path):
        """Plot precision-recall curve"""
        if not pr_data.get('precision') or not pr_data.get('recall'):
            return
        
        plt.figure(figsize=(8, 6))
        
        precisions = pr_data['precision']
        recalls = pr_data['recall']
        
        plt.plot(recalls, precisions, 'b-', linewidth=2, label='PR Curve')
        plt.fill_between(recalls, precisions, alpha=0.2)
        
        # Mark optimal threshold
        if 'optimal_threshold' in pr_data and 'f1_scores' in pr_data:
            opt_idx = pr_data['thresholds'].index(pr_data['optimal_threshold'])
            if opt_idx < len(precisions) - 1:  # Exclude last point
                plt.plot(recalls[opt_idx], precisions[opt_idx], 'ro', markersize=8, 
                        label=f'Optimal (conf={pr_data["optimal_threshold"]:.3f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        
        # Save plot
        pr_plot_path = output_path / "precision_recall_curve.png"
        plt.savefig(pr_plot_path, dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Main function for testing comprehensive evaluator"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive Evaluation Module")
    parser.add_argument("--model", required=True, help="Path to model weights")
    parser.add_argument("--dataset-config", required=True, help="Path to dataset configuration JSON")
    parser.add_argument("--output", default="./evaluation_results", help="Output directory")
    parser.add_argument("--tile-size", type=int, default=640, help="SAHI tile size")
    parser.add_argument("--overlap", type=float, default=0.2, help="SAHI overlap ratio")
    parser.add_argument("--confidence", type=float, default=0.20, help="Initial confidence threshold")
    
    args = parser.parse_args()
    
    # Load dataset configuration
    with open(args.dataset_config, 'r') as f:
        dataset_configs = json.load(f)
    
    # Create SAHI configuration
    sahi_config = SAHIConfig(
        tile_size=args.tile_size,
        overlap_ratio=args.overlap,
        confidence_threshold=args.confidence
    )
    
    # Create evaluation configuration
    eval_config = EvaluationConfig()
    
    # Initialize evaluator
    evaluator = ComprehensiveEvaluator(
        model_path=args.model,
        sahi_config=sahi_config,
        eval_config=eval_config,
        experiment_name="comprehensive_evaluation"
    )
    
    # Run evaluation
    results = evaluator.compare_institutions(dataset_configs, args.output)
    
    # Print summary
    print("\n🎉 Comprehensive evaluation completed!")
    for institution, result in results.items():
        print(f"\n📊 {institution} Results:")
        print(f"  mAP@0.5: {result.map_50:.3f}")
        print(f"  mAP@0.75: {result.map_75:.3f}")
        print(f"  mAP@0.5:0.95: {result.map_50_95:.3f}")
        print(f"  Optimal confidence: {result.optimal_confidence:.3f}")
        print(f"  Processing time: {result.average_processing_time:.2f}s/image")
    
    print(f"\n📁 Results saved to: {args.output}")


if __name__ == "__main__":
    main() 