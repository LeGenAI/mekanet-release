"""
Detection Evaluator for TESSD Framework
Comprehensive evaluation metrics for megakaryocyte detection performance
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import logging
from sklearn.metrics import precision_recall_curve, average_precision_score
import cv2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DetectionEvaluator:
    """
    Comprehensive evaluation for megakaryocyte detection
    
    Implements metrics from SAHI1.ipynb and paper requirements:
    - IoU-based precision/recall
    - mAP calculation
    - Confidence threshold analysis
    - Cross-institutional validation
    """
    
    def __init__(self, iou_thresholds: List[float] = None):
        """
        Initialize evaluator
        
        Args:
            iou_thresholds: List of IoU thresholds for evaluation (default: [0.5, 0.75])
        """
        self.iou_thresholds = iou_thresholds or [0.5, 0.75]
        self.results_history = []
        
    def compute_iou(self, boxA: List[float], boxB: List[float]) -> float:
        """
        Compute Intersection over Union (IoU) between two bounding boxes
        Implementation from SAHI1.ipynb
        
        Args:
            boxA: [x1, y1, x2, y2] format
            boxB: [x1, y1, x2, y2] format
            
        Returns:
            IoU score
        """
        # Determine the coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        
        # Compute the area of intersection
        interArea = max(0, xB - xA) * max(0, yB - yA)
        
        # Compute the area of both bounding boxes
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        
        # Compute the intersection over union
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou
    
    def evaluate_single_image(self, 
                            ground_truth_boxes: List[Dict],
                            predicted_boxes: List[Dict],
                            iou_threshold: float = 0.5) -> Dict[str, Any]:
        """
        Evaluate predictions for a single image
        Based on evaluation logic from SAHI1.ipynb
        
        Args:
            ground_truth_boxes: List of ground truth boxes with 'bbox' key
            predicted_boxes: List of predictions with 'bbox' and 'score' keys
            iou_threshold: IoU threshold for considering a match
            
        Returns:
            Dictionary with precision, recall, F1, and IoU metrics
        """
        if not ground_truth_boxes and not predicted_boxes:
            return {
                'precision': 1.0,
                'recall': 1.0,
                'f1_score': 1.0,
                'mean_iou': 0.0,
                'num_matches': 0,
                'num_gt': 0,
                'num_pred': 0
            }
        
        if not predicted_boxes:
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'mean_iou': 0.0,
                'num_matches': 0,
                'num_gt': len(ground_truth_boxes),
                'num_pred': 0
            }
        
        if not ground_truth_boxes:
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'mean_iou': 0.0,
                'num_matches': 0,
                'num_gt': 0,
                'num_pred': len(predicted_boxes)
            }
        
        # Convert bounding boxes to standard format
        gt_boxes = []
        for gt in ground_truth_boxes:
            if 'bbox' in gt:
                gt_boxes.append(gt['bbox'])
            else:
                gt_boxes.append(gt)
        
        pred_boxes = []
        pred_scores = []
        for pred in predicted_boxes:
            if isinstance(pred, dict):
                if 'bbox' in pred:
                    bbox = pred['bbox']
                    if isinstance(bbox, dict):
                        # Convert from dict format {x1, y1, x2, y2}
                        pred_boxes.append([bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']])
                    else:
                        pred_boxes.append(bbox)
                    pred_scores.append(pred.get('score', 1.0))
                else:
                    pred_boxes.append(pred)
                    pred_scores.append(1.0)
            else:
                pred_boxes.append(pred)
                pred_scores.append(1.0)
        
        # Track matches and IoU scores
        matches = 0
        iou_scores = []
        matched_gt = set()
        matched_pred = set()
        
        # For each ground truth box, find best matching prediction
        for i, gt_box in enumerate(gt_boxes):
            best_iou = 0
            best_pred_idx = -1
            
            for j, pred_box in enumerate(pred_boxes):
                if j in matched_pred:
                    continue
                    
                iou = self.compute_iou(gt_box, pred_box)
                if iou > best_iou:
                    best_iou = iou
                    best_pred_idx = j
            
            if best_iou >= iou_threshold and best_pred_idx != -1:
                matches += 1
                matched_gt.add(i)
                matched_pred.add(best_pred_idx)
                iou_scores.append(best_iou)
        
        # Calculate metrics
        precision = matches / len(pred_boxes) if pred_boxes else 0
        recall = matches / len(gt_boxes) if gt_boxes else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        mean_iou = np.mean(iou_scores) if iou_scores else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'mean_iou': mean_iou,
            'num_matches': matches,
            'num_gt': len(gt_boxes),
            'num_pred': len(pred_boxes),
            'iou_scores': iou_scores
        }
    
    def evaluate_dataset(self, 
                        predictions: List[Dict],
                        ground_truth: List[Dict] = None) -> Dict[str, Any]:
        """
        Evaluate predictions across a dataset
        
        Args:
            predictions: List of prediction results with image_id and detections
            ground_truth: List of ground truth annotations (optional)
            
        Returns:
            Comprehensive evaluation metrics
        """
        results_by_threshold = {}
        
        for iou_threshold in self.iou_thresholds:
            image_results = []
            
            for pred_data in predictions:
                image_id = pred_data['image_id']
                pred_boxes = pred_data['detections']
                
                # Find corresponding ground truth
                gt_boxes = []
                if ground_truth:
                    for gt_data in ground_truth:
                        if gt_data['image_id'] == image_id:
                            gt_boxes = gt_data.get('annotations', [])
                            break
                
                # Evaluate single image
                result = self.evaluate_single_image(gt_boxes, pred_boxes, iou_threshold)
                result['image_id'] = image_id
                result['institution'] = pred_data.get('institution', 'unknown')
                image_results.append(result)
            
            # Aggregate results
            aggregated = self._aggregate_results(image_results)
            results_by_threshold[f'iou_{iou_threshold}'] = aggregated
        
        return results_by_threshold
    
    def _aggregate_results(self, image_results: List[Dict]) -> Dict[str, Any]:
        """Aggregate results across multiple images"""
        if not image_results:
            return {}
        
        # Calculate weighted averages and totals
        total_matches = sum(r['num_matches'] for r in image_results)
        total_gt = sum(r['num_gt'] for r in image_results)
        total_pred = sum(r['num_pred'] for r in image_results)
        
        # Overall precision and recall
        overall_precision = total_matches / total_pred if total_pred > 0 else 0
        overall_recall = total_matches / total_gt if total_gt > 0 else 0
        overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
        
        # Per-image averages
        precisions = [r['precision'] for r in image_results]
        recalls = [r['recall'] for r in image_results]
        f1_scores = [r['f1_score'] for r in image_results]
        ious = [r['mean_iou'] for r in image_results]
        
        # Collect all IoU scores for distribution analysis
        all_ious = []
        for r in image_results:
            all_ious.extend(r.get('iou_scores', []))
        
        return {
            'overall_precision': overall_precision,
            'overall_recall': overall_recall,
            'overall_f1': overall_f1,
            'mean_precision': np.mean(precisions),
            'std_precision': np.std(precisions),
            'mean_recall': np.mean(recalls),
            'std_recall': np.std(recalls),
            'mean_f1': np.mean(f1_scores),
            'std_f1': np.std(f1_scores),
            'mean_iou': np.mean(ious),
            'std_iou': np.std(ious),
            'total_matches': total_matches,
            'total_gt': total_gt,
            'total_pred': total_pred,
            'num_images': len(image_results),
            'iou_distribution': all_ious,
            'image_results': image_results
        }
    
    def calculate_map(self, 
                     predictions: List[Dict],
                     ground_truth: List[Dict],
                     iou_threshold: float = 0.5) -> float:
        """
        Calculate mean Average Precision (mAP)
        
        Args:
            predictions: Predictions with confidence scores
            ground_truth: Ground truth annotations
            iou_threshold: IoU threshold for matches
            
        Returns:
            mAP score
        """
        # Collect all predictions and ground truth across images
        all_predictions = []
        all_ground_truth = []
        
        for pred_data in predictions:
            image_id = pred_data['image_id']
            
            # Find corresponding ground truth
            gt_boxes = []
            for gt_data in ground_truth:
                if gt_data['image_id'] == image_id:
                    gt_boxes = gt_data.get('annotations', [])
                    break
            
            # Add image-specific identifiers
            for pred in pred_data['detections']:
                pred_copy = pred.copy()
                pred_copy['image_id'] = image_id
                all_predictions.append(pred_copy)
            
            for gt in gt_boxes:
                gt_copy = gt.copy()
                gt_copy['image_id'] = image_id
                all_ground_truth.append(gt_copy)
        
        if not all_predictions or not all_ground_truth:
            return 0.0
        
        # Sort predictions by confidence score (descending)
        all_predictions.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        # Match predictions to ground truth
        tp = np.zeros(len(all_predictions))
        fp = np.zeros(len(all_predictions))
        
        gt_matched = set()
        
        for i, pred in enumerate(all_predictions):
            pred_box = pred['bbox']
            pred_image = pred['image_id']
            
            # Convert bbox format if needed
            if isinstance(pred_box, dict):
                pred_box = [pred_box['x1'], pred_box['y1'], pred_box['x2'], pred_box['y2']]
            
            best_iou = 0
            best_gt_idx = -1
            
            for j, gt in enumerate(all_ground_truth):
                if gt['image_id'] != pred_image:
                    continue
                    
                gt_key = f"{pred_image}_{j}"
                if gt_key in gt_matched:
                    continue
                
                gt_box = gt['bbox']
                if isinstance(gt_box, dict):
                    gt_box = [gt_box['x1'], gt_box['y1'], gt_box['x2'], gt_box['y2']]
                
                iou = self.compute_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
            
            if best_iou >= iou_threshold and best_gt_idx != -1:
                gt_key = f"{pred_image}_{best_gt_idx}"
                if gt_key not in gt_matched:
                    tp[i] = 1
                    gt_matched.add(gt_key)
                else:
                    fp[i] = 1
            else:
                fp[i] = 1
        
        # Calculate precision and recall
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        recalls = tp_cumsum / len(all_ground_truth)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
        
        # Calculate Average Precision
        ap = average_precision_score([1] * len(recalls), precisions) if len(recalls) > 0 else 0
        return ap
    
    def analyze_confidence_thresholds(self, 
                                    predictions_by_threshold: Dict[str, List[Dict]],
                                    ground_truth: List[Dict] = None) -> pd.DataFrame:
        """
        Analyze performance across different confidence thresholds
        Based on threshold analysis from SAHI1.ipynb
        """
        results = []
        
        for threshold_name, predictions in predictions_by_threshold.items():
            threshold_value = float(threshold_name.split('_')[1])
            
            # Calculate metrics
            if ground_truth:
                eval_results = self.evaluate_dataset(predictions, ground_truth)
                map_score = self.calculate_map(predictions, ground_truth)
            else:
                # Basic statistics when no ground truth available
                eval_results = {'iou_0.5': {'overall_precision': 0, 'overall_recall': 0, 'overall_f1': 0}}
                map_score = 0
            
            # Extract key metrics
            metrics = eval_results.get('iou_0.5', {})
            
            results.append({
                'confidence_threshold': threshold_value,
                'num_detections': sum(len(pred['detections']) for pred in predictions),
                'avg_detections_per_image': np.mean([len(pred['detections']) for pred in predictions]),
                'precision': metrics.get('overall_precision', 0),
                'recall': metrics.get('overall_recall', 0),
                'f1_score': metrics.get('overall_f1', 0),
                'map': map_score
            })
        
        return pd.DataFrame(results)
    
    def create_evaluation_report(self, 
                               results: Dict[str, Any],
                               output_dir: str = None) -> str:
        """Create comprehensive evaluation report"""
        report_lines = []
        report_lines.append("TESSD DETECTION EVALUATION REPORT")
        report_lines.append("=" * 50)
        report_lines.append("")
        
        for threshold_key, metrics in results.items():
            iou_value = threshold_key.split('_')[1]
            report_lines.append(f"IoU Threshold: {iou_value}")
            report_lines.append("-" * 30)
            
            report_lines.append(f"Overall Metrics:")
            report_lines.append(f"  Precision: {metrics['overall_precision']:.3f}")
            report_lines.append(f"  Recall: {metrics['overall_recall']:.3f}")
            report_lines.append(f"  F1-Score: {metrics['overall_f1']:.3f}")
            report_lines.append(f"  Mean IoU: {metrics['mean_iou']:.3f}")
            report_lines.append("")
            
            report_lines.append(f"Dataset Statistics:")
            report_lines.append(f"  Total Images: {metrics['num_images']}")
            report_lines.append(f"  Total Ground Truth: {metrics['total_gt']}")
            report_lines.append(f"  Total Predictions: {metrics['total_pred']}")
            report_lines.append(f"  Total Matches: {metrics['total_matches']}")
            report_lines.append("")
        
        report_text = "\n".join(report_lines)
        
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            report_file = output_path / "evaluation_report.txt"
            
            with open(report_file, 'w') as f:
                f.write(report_text)
            
            logger.info(f"Evaluation report saved to: {report_file}")
        
        return report_text
    
    def plot_performance_analysis(self, 
                                 results: Dict[str, Any],
                                 output_dir: str = None):
        """Create performance analysis plots"""
        if not results:
            return
        
        # Setup plot style
        plt.style.use('default')
        sns.set_palette("husl")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('TESSD Detection Performance Analysis', fontsize=16, fontweight='bold')
        
        # Extract data for plotting
        iou_thresholds = []
        precisions = []
        recalls = []
        f1_scores = []
        
        for threshold_key, metrics in results.items():
            iou_value = float(threshold_key.split('_')[1])
            iou_thresholds.append(iou_value)
            precisions.append(metrics['overall_precision'])
            recalls.append(metrics['overall_recall'])
            f1_scores.append(metrics['overall_f1'])
        
        # Plot 1: Precision-Recall by IoU threshold
        axes[0, 0].plot(iou_thresholds, precisions, 'o-', label='Precision', linewidth=2, markersize=8)
        axes[0, 0].plot(iou_thresholds, recalls, 's-', label='Recall', linewidth=2, markersize=8)
        axes[0, 0].plot(iou_thresholds, f1_scores, '^-', label='F1-Score', linewidth=2, markersize=8)
        axes[0, 0].set_xlabel('IoU Threshold')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Performance vs IoU Threshold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: IoU distribution (if available)
        if 'iou_0.5' in results and 'iou_distribution' in results['iou_0.5']:
            iou_dist = results['iou_0.5']['iou_distribution']
            if iou_dist:
                axes[0, 1].hist(iou_dist, bins=20, alpha=0.7, edgecolor='black')
                axes[0, 1].axvline(np.mean(iou_dist), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(iou_dist):.3f}')
                axes[0, 1].set_xlabel('IoU Score')
                axes[0, 1].set_ylabel('Frequency')
                axes[0, 1].set_title('IoU Score Distribution')
                axes[0, 1].legend()
        
        # Plot 3: Detection count analysis
        if 'iou_0.5' in results and 'image_results' in results['iou_0.5']:
            image_results = results['iou_0.5']['image_results']
            institutions = [r.get('institution', 'unknown') for r in image_results]
            num_detections = [r['num_pred'] for r in image_results]
            
            # Group by institution
            inst_data = {}
            for inst, count in zip(institutions, num_detections):
                if inst not in inst_data:
                    inst_data[inst] = []
                inst_data[inst].append(count)
            
            if len(inst_data) > 1:
                inst_names = list(inst_data.keys())
                inst_counts = [inst_data[name] for name in inst_names]
                axes[1, 0].boxplot(inst_counts, labels=inst_names)
                axes[1, 0].set_ylabel('Number of Detections')
                axes[1, 0].set_title('Detection Count by Institution')
            else:
                axes[1, 0].hist(num_detections, bins=10, alpha=0.7, edgecolor='black')
                axes[1, 0].set_xlabel('Number of Detections')
                axes[1, 0].set_ylabel('Frequency')
                axes[1, 0].set_title('Detection Count Distribution')
        
        # Plot 4: Performance by institution
        if 'iou_0.5' in results and 'image_results' in results['iou_0.5']:
            image_results = results['iou_0.5']['image_results']
            inst_performance = {}
            
            for result in image_results:
                inst = result.get('institution', 'unknown')
                if inst not in inst_performance:
                    inst_performance[inst] = {'precision': [], 'recall': [], 'f1': []}
                
                inst_performance[inst]['precision'].append(result['precision'])
                inst_performance[inst]['recall'].append(result['recall'])
                inst_performance[inst]['f1'].append(result['f1_score'])
            
            if len(inst_performance) > 1:
                institutions = list(inst_performance.keys())
                metrics = ['precision', 'recall', 'f1']
                x_pos = np.arange(len(institutions))
                width = 0.25
                
                for i, metric in enumerate(metrics):
                    means = [np.mean(inst_performance[inst][metric]) for inst in institutions]
                    axes[1, 1].bar(x_pos + i*width, means, width, label=metric.capitalize())
                
                axes[1, 1].set_xlabel('Institution')
                axes[1, 1].set_ylabel('Score')
                axes[1, 1].set_title('Performance by Institution')
                axes[1, 1].set_xticks(x_pos + width)
                axes[1, 1].set_xticklabels(institutions)
                axes[1, 1].legend()
        
        plt.tight_layout()
        
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            plot_file = output_path / "performance_analysis.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            logger.info(f"Performance plots saved to: {plot_file}")
        
        plt.show() 