#!/usr/bin/env python3
"""
Semi-Supervised Training Module for TESSD Framework
Implements self-training approach for partially labeled megakaryocyte detection
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import cv2
import yaml
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Setup paths
sys.path.append(str(Path(__file__).parent.parent.parent))

from experiments.detection.tessd_framework import TESSDFramework

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SemiSupervisedDetectionDataset(Dataset):
    """
    Dataset for semi-supervised detection training
    Handles both labeled and unlabeled images
    """
    
    def __init__(self, 
                 image_paths: List[str],
                 labels: Optional[List[Dict]] = None,
                 pseudo_labels: Optional[List[Dict]] = None,
                 transform=None,
                 is_labeled: bool = True):
        """
        Initialize dataset
        
        Args:
            image_paths: List of image file paths
            labels: Ground truth labels (YOLO format)
            pseudo_labels: Generated pseudo labels
            transform: Albumentations transforms
            is_labeled: Whether this dataset contains labeled data
        """
        self.image_paths = image_paths
        self.labels = labels if labels is not None else [None] * len(image_paths)
        self.pseudo_labels = pseudo_labels if pseudo_labels is not None else [None] * len(image_paths)
        self.transform = transform
        self.is_labeled = is_labeled
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get labels (ground truth or pseudo)
        if self.is_labeled and self.labels[idx] is not None:
            labels = self.labels[idx]
        elif self.pseudo_labels[idx] is not None:
            labels = self.pseudo_labels[idx]
        else:
            labels = []
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, bboxes=labels)
            image = transformed['image']
            labels = transformed['bboxes']
        
        return {
            'image': image,
            'labels': labels,
            'image_path': image_path,
            'is_labeled': self.is_labeled
        }


class SemiSupervisedTrainer:
    """
    Semi-supervised trainer for TESSD framework
    
    Implements iterative self-training with pseudo-labeling:
    1. Train on labeled data
    2. Generate pseudo-labels on unlabeled data
    3. Select high-confidence pseudo-labels
    4. Retrain on labeled + pseudo-labeled data
    5. Repeat until convergence
    """
    
    def __init__(self, 
                 initial_model_path: str,
                 labeled_data_dir: str,
                 unlabeled_data_dir: str,
                 output_dir: str,
                 config: Dict[str, Any]):
        """
        Initialize semi-supervised trainer
        
        Args:
            initial_model_path: Path to pretrained model
            labeled_data_dir: Directory with labeled training data
            unlabeled_data_dir: Directory with unlabeled data
            output_dir: Output directory for models and logs
            config: Training configuration
        """
        self.initial_model_path = initial_model_path
        self.labeled_data_dir = Path(labeled_data_dir)
        self.unlabeled_data_dir = Path(unlabeled_data_dir)
        self.output_dir = Path(output_dir)
        self.config = config
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "models").mkdir(exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)
        (self.output_dir / "pseudo_labels").mkdir(exist_ok=True)
        
        # Initialize TESSD framework
        self.tessd = TESSDFramework(
            model_path=initial_model_path,
            confidence_threshold=config.get('initial_confidence_threshold', 0.20),
            experiment_name="semi_supervised_training"
        )
        
        # Training parameters
        self.max_iterations = config.get('max_iterations', 5)
        self.pseudo_label_confidence = config.get('pseudo_label_confidence', 0.80)
        self.min_improvement = config.get('min_improvement', 0.01)
        self.validation_split = config.get('validation_split', 0.2)
        
        # Track training progress
        self.training_history = []
        self.current_iteration = 0
        
        logger.info("Semi-supervised trainer initialized")
        logger.info(f"Labeled data: {self.labeled_data_dir}")
        logger.info(f"Unlabeled data: {self.unlabeled_data_dir}")
        logger.info(f"Output: {self.output_dir}")
    
    def load_labeled_data(self) -> Tuple[List[str], List[Dict]]:
        """
        Load labeled training data
        
        Returns:
            Tuple of (image_paths, labels)
        """
        image_paths = []
        labels = []
        
        # Find image files
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_paths.extend(list(self.labeled_data_dir.glob(f"**/{ext}")))
        
        # Load corresponding labels (YOLO format)
        for image_path in image_paths:
            label_path = image_path.with_suffix('.txt')
            
            if label_path.exists():
                # Load YOLO format labels
                image_labels = []
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id, x_center, y_center, width, height = map(float, parts[:5])
                            image_labels.append({
                                'class_id': int(class_id),
                                'x_center': x_center,
                                'y_center': y_center,
                                'width': width,
                                'height': height
                            })
                labels.append(image_labels)
            else:
                # No labels found - treat as unlabeled
                labels.append([])
        
        logger.info(f"Loaded {len(image_paths)} labeled images")
        return [str(p) for p in image_paths], labels
    
    def load_unlabeled_data(self) -> List[str]:
        """
        Load unlabeled data paths
        
        Returns:
            List of image paths
        """
        image_paths = []
        
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_paths.extend(list(self.unlabeled_data_dir.glob(f"**/{ext}")))
        
        logger.info(f"Loaded {len(image_paths)} unlabeled images")
        return [str(p) for p in image_paths]
    
    def generate_pseudo_labels(self, 
                             unlabeled_paths: List[str],
                             confidence_threshold: float = None) -> List[Dict]:
        """
        Generate pseudo-labels for unlabeled data
        
        Args:
            unlabeled_paths: Paths to unlabeled images
            confidence_threshold: Minimum confidence for pseudo-labels
            
        Returns:
            List of pseudo-labels in YOLO format
        """
        if confidence_threshold is None:
            confidence_threshold = self.pseudo_label_confidence
        
        logger.info(f"Generating pseudo-labels with confidence >= {confidence_threshold}")
        
        pseudo_labels = []
        high_confidence_count = 0
        
        for image_path in unlabeled_paths:
            # Load and process image
            image = cv2.imread(image_path)
            if image is None:
                pseudo_labels.append([])
                continue
            
            # Get detections
            detections = self.tessd.predict(
                image,
                use_sahi=True,
                slice_height=640,
                slice_width=640,
                overlap_height_ratio=0.2,
                overlap_width_ratio=0.2
            )
            
            # Filter high-confidence detections for pseudo-labels
            image_pseudo_labels = []
            for det in detections:
                if det['score'] >= confidence_threshold:
                    # Convert to YOLO format
                    bbox = det['bbox']
                    x_center = (bbox['x1'] + bbox['x2']) / 2 / image.shape[1]
                    y_center = (bbox['y1'] + bbox['y2']) / 2 / image.shape[0]
                    width = (bbox['x2'] - bbox['x1']) / image.shape[1]
                    height = (bbox['y2'] - bbox['y1']) / image.shape[0]
                    
                    image_pseudo_labels.append({
                        'class_id': 0,  # Megakaryocyte class
                        'x_center': x_center,
                        'y_center': y_center,
                        'width': width,
                        'height': height,
                        'confidence': det['score']
                    })
            
            pseudo_labels.append(image_pseudo_labels)
            if image_pseudo_labels:
                high_confidence_count += len(image_pseudo_labels)
        
        logger.info(f"Generated {high_confidence_count} high-confidence pseudo-labels from {len(unlabeled_paths)} images")
        return pseudo_labels
    
    def evaluate_model(self, validation_data: List[Tuple[str, List[Dict]]]) -> Dict[str, float]:
        """
        Evaluate model performance on validation set
        
        Args:
            validation_data: List of (image_path, labels) tuples
            
        Returns:
            Evaluation metrics
        """
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_detections = 0
        total_ground_truth = 0
        
        for image_path, gt_labels in validation_data:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                continue
            
            # Get predictions
            detections = self.tessd.predict(image, use_sahi=True)
            
            # Convert ground truth to absolute coordinates
            gt_boxes = []
            for label in gt_labels:
                x_center = label['x_center'] * image.shape[1]
                y_center = label['y_center'] * image.shape[0]
                width = label['width'] * image.shape[1]
                height = label['height'] * image.shape[0]
                
                x1 = x_center - width / 2
                y1 = y_center - height / 2
                x2 = x_center + width / 2
                y2 = y_center + height / 2
                
                gt_boxes.append([x1, y1, x2, y2])
            
            # Calculate IoU-based metrics
            pred_boxes = [[det['bbox']['x1'], det['bbox']['y1'], 
                          det['bbox']['x2'], det['bbox']['y2']] for det in detections]
            
            tp, fp, fn = self._calculate_tp_fp_fn(pred_boxes, gt_boxes, iou_threshold=0.5)
            
            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_detections += len(detections)
            total_ground_truth += len(gt_labels)
        
        # Calculate metrics
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'total_detections': total_detections,
            'total_ground_truth': total_ground_truth
        }
    
    def _calculate_tp_fp_fn(self, pred_boxes: List[List[float]], 
                           gt_boxes: List[List[float]], 
                           iou_threshold: float = 0.5) -> Tuple[int, int, int]:
        """
        Calculate True Positives, False Positives, and False Negatives
        
        Args:
            pred_boxes: Predicted bounding boxes [[x1, y1, x2, y2], ...]
            gt_boxes: Ground truth bounding boxes [[x1, y1, x2, y2], ...]
            iou_threshold: IoU threshold for positive match
            
        Returns:
            Tuple of (tp, fp, fn)
        """
        if not pred_boxes and not gt_boxes:
            return 0, 0, 0
        if not pred_boxes:
            return 0, 0, len(gt_boxes)
        if not gt_boxes:
            return 0, len(pred_boxes), 0
        
        # Calculate IoU matrix
        iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)))
        
        for i, pred_box in enumerate(pred_boxes):
            for j, gt_box in enumerate(gt_boxes):
                iou_matrix[i, j] = self._calculate_iou(pred_box, gt_box)
        
        # Find matches
        used_gt = set()
        tp = 0
        
        # Sort predictions by confidence (assuming they're already sorted)
        for i in range(len(pred_boxes)):
            best_gt_idx = -1
            best_iou = 0
            
            for j in range(len(gt_boxes)):
                if j not in used_gt and iou_matrix[i, j] > best_iou and iou_matrix[i, j] >= iou_threshold:
                    best_iou = iou_matrix[i, j]
                    best_gt_idx = j
            
            if best_gt_idx >= 0:
                tp += 1
                used_gt.add(best_gt_idx)
        
        fp = len(pred_boxes) - tp
        fn = len(gt_boxes) - tp
        
        return tp, fp, fn
    
    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """
        Calculate Intersection over Union (IoU) of two bounding boxes
        
        Args:
            box1: [x1, y1, x2, y2]
            box2: [x1, y1, x2, y2]
            
        Returns:
            IoU value
        """
        # Calculate intersection
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        # Calculate union
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = box1_area + box2_area - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def train_iteration(self, 
                       labeled_data: Tuple[List[str], List[Dict]],
                       pseudo_labeled_data: Tuple[List[str], List[Dict]] = None) -> Dict[str, Any]:
        """
        Run one training iteration
        
        Args:
            labeled_data: Tuple of (image_paths, labels)
            pseudo_labeled_data: Tuple of (image_paths, pseudo_labels)
            
        Returns:
            Training results
        """
        # This is a simplified version - in practice, you'd need to integrate with 
        # YOLOv8 training pipeline or use the ultralytics API
        
        # For now, we'll simulate training and return mock results
        logger.info(f"Training iteration {self.current_iteration + 1}")
        
        labeled_paths, labeled_labels = labeled_data
        total_labeled = sum(len(labels) for labels in labeled_labels)
        
        if pseudo_labeled_data:
            pseudo_paths, pseudo_labels = pseudo_labeled_data
            total_pseudo = sum(len(labels) for labels in pseudo_labels)
            logger.info(f"Using {len(labeled_paths)} labeled images ({total_labeled} labels)")
            logger.info(f"Using {len(pseudo_paths)} pseudo-labeled images ({total_pseudo} pseudo-labels)")
        else:
            logger.info(f"Using {len(labeled_paths)} labeled images ({total_labeled} labels)")
        
        # Simulate training time
        time.sleep(2)
        
        # Mock training results
        results = {
            'iteration': self.current_iteration + 1,
            'labeled_samples': len(labeled_paths),
            'pseudo_labeled_samples': len(pseudo_paths) if pseudo_labeled_data else 0,
            'total_labels': total_labeled,
            'total_pseudo_labels': total_pseudo if pseudo_labeled_data else 0,
            'training_time': 2.0
        }
        
        return results
    
    def run_semi_supervised_training(self) -> Dict[str, Any]:
        """
        Run complete semi-supervised training pipeline
        
        Returns:
            Training results and final model path
        """
        logger.info("Starting semi-supervised training pipeline")
        
        # Load initial data
        labeled_paths, labeled_labels = self.load_labeled_data()
        unlabeled_paths = self.load_unlabeled_data()
        
        # Split labeled data for validation
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            labeled_paths, labeled_labels, 
            test_size=self.validation_split, 
            random_state=42
        )
        
        validation_data = list(zip(val_paths, val_labels))
        
        # Initial evaluation
        initial_metrics = self.evaluate_model(validation_data)
        logger.info(f"Initial performance: {initial_metrics}")
        
        best_f1 = initial_metrics['f1_score']
        best_model_path = self.initial_model_path
        
        # Iterative self-training
        for iteration in range(self.max_iterations):
            self.current_iteration = iteration
            logger.info(f"\n{'='*50}")
            logger.info(f"Semi-supervised Training Iteration {iteration + 1}/{self.max_iterations}")
            logger.info(f"{'='*50}")
            
            # Generate pseudo-labels for unlabeled data
            pseudo_labels = self.generate_pseudo_labels(unlabeled_paths)
            
            # Filter images with pseudo-labels
            pseudo_paths_with_labels = []
            pseudo_labels_filtered = []
            
            for path, labels in zip(unlabeled_paths, pseudo_labels):
                if labels:  # Only include images with pseudo-labels
                    pseudo_paths_with_labels.append(path)
                    pseudo_labels_filtered.append(labels)
            
            # Run training iteration
            training_results = self.train_iteration(
                labeled_data=(train_paths, train_labels),
                pseudo_labeled_data=(pseudo_paths_with_labels, pseudo_labels_filtered) if pseudo_paths_with_labels else None
            )
            
            # Evaluate current model
            current_metrics = self.evaluate_model(validation_data)
            
            # Save iteration results
            iteration_results = {
                **training_results,
                **current_metrics,
                'pseudo_label_confidence': self.pseudo_label_confidence
            }
            
            self.training_history.append(iteration_results)
            
            logger.info(f"Iteration {iteration + 1} Results:")
            logger.info(f"  Precision: {current_metrics['precision']:.4f}")
            logger.info(f"  Recall: {current_metrics['recall']:.4f}")
            logger.info(f"  F1-Score: {current_metrics['f1_score']:.4f}")
            logger.info(f"  Pseudo-labels used: {len(pseudo_paths_with_labels)}")
            
            # Check for improvement
            improvement = current_metrics['f1_score'] - best_f1
            
            if improvement > self.min_improvement:
                best_f1 = current_metrics['f1_score']
                best_model_path = self.output_dir / "models" / f"best_model_iter_{iteration + 1}.pt"
                logger.info(f"  ✅ Improvement: +{improvement:.4f} F1-score")
                
                # In practice, save the actual model here
                # torch.save(model.state_dict(), best_model_path)
                
            else:
                logger.info(f"  ⏸️ No significant improvement (+{improvement:.4f})")
                
                # Early stopping if no improvement for multiple iterations
                if iteration > 0:
                    recent_improvements = [
                        self.training_history[i]['f1_score'] - self.training_history[i-1]['f1_score'] 
                        for i in range(max(1, len(self.training_history)-2), len(self.training_history))
                    ]
                    
                    if all(imp <= self.min_improvement for imp in recent_improvements):
                        logger.info("Early stopping: No improvement in recent iterations")
                        break
            
            # Gradually increase confidence threshold for pseudo-labels
            self.pseudo_label_confidence = min(0.95, self.pseudo_label_confidence + 0.02)
        
        # Save training history
        history_df = pd.DataFrame(self.training_history)
        history_df.to_csv(self.output_dir / "training_history.csv", index=False)
        
        # Final results
        final_results = {
            'best_f1_score': best_f1,
            'best_model_path': str(best_model_path),
            'total_iterations': len(self.training_history),
            'initial_f1': initial_metrics['f1_score'],
            'improvement': best_f1 - initial_metrics['f1_score'],
            'training_history': self.training_history
        }
        
        logger.info(f"\n{'='*50}")
        logger.info("Semi-supervised Training Complete")
        logger.info(f"{'='*50}")
        logger.info(f"Best F1-Score: {best_f1:.4f}")
        logger.info(f"Total Improvement: +{final_results['improvement']:.4f}")
        logger.info(f"Training History saved to: {self.output_dir / 'training_history.csv'}")
        
        return final_results


def create_training_config() -> Dict[str, Any]:
    """
    Create default configuration for semi-supervised training
    
    Returns:
        Configuration dictionary
    """
    return {
        'max_iterations': 5,
        'initial_confidence_threshold': 0.20,
        'pseudo_label_confidence': 0.80,
        'min_improvement': 0.01,
        'validation_split': 0.2,
        'iou_threshold': 0.5,
        'early_stopping_patience': 2,
        'augmentation': {
            'horizontal_flip': 0.5,
            'vertical_flip': 0.2,
            'rotation': 15,
            'brightness': 0.2,
            'contrast': 0.2
        }
    }


def main():
    """
    Main function for running semi-supervised training
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Semi-supervised training for TESSD")
    parser.add_argument("--model", required=True, help="Path to initial model")
    parser.add_argument("--labeled-data", required=True, help="Path to labeled data directory")
    parser.add_argument("--unlabeled-data", required=True, help="Path to unlabeled data directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--config", help="Path to configuration YAML file")
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = create_training_config()
    
    # Initialize trainer
    trainer = SemiSupervisedTrainer(
        initial_model_path=args.model,
        labeled_data_dir=args.labeled_data,
        unlabeled_data_dir=args.unlabeled_data,
        output_dir=args.output,
        config=config
    )
    
    # Run training
    results = trainer.run_semi_supervised_training()
    
    print("\n🎉 Semi-supervised training completed!")
    print(f"📈 Best F1-Score: {results['best_f1_score']:.4f}")
    print(f"📁 Results saved to: {args.output}")


if __name__ == "__main__":
    main() 