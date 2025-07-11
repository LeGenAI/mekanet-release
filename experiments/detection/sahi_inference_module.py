#!/usr/bin/env python3
"""
SAHI-Enhanced Inference Module for TESSD Framework
Advanced tiling-based detection with configurable parameters and optimization features
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
from concurrent.futures import ThreadPoolExecutor, as_completed

# Setup paths
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from experiments.detection.tessd_framework import TESSDFramework

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SAHIConfig:
    """Configuration for SAHI inference parameters"""
    tile_size: int = 640
    overlap_ratio: float = 0.2
    confidence_threshold: float = 0.20
    iou_threshold: float = 0.5
    
    # Advanced SAHI parameters
    postprocess_type: str = "NMS"  # NMS, GREEDYNMM
    postprocess_match_metric: str = "IOS"  # IOU, IOS
    postprocess_match_threshold: float = 0.5
    postprocess_class_agnostic: bool = False
    
    # Performance optimization
    device: str = "auto"
    batch_size: int = 1
    use_mixed_precision: bool = False
    
    # Quality control
    min_detection_area: int = 100
    max_detections_per_image: int = 1000
    
    # Multi-scale inference
    use_multiscale: bool = False
    scale_factors: List[float] = None
    
    def __post_init__(self):
        if self.scale_factors is None:
            self.scale_factors = [0.8, 1.0, 1.2]


@dataclass
class InferenceResult:
    """Results from SAHI inference"""
    image_id: str
    detections: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    processing_time: float
    tile_info: Dict[str, Any]
    quality_metrics: Dict[str, Any]


class SAHIInferenceModule:
    """
    Enhanced SAHI inference module for TESSD framework
    
    Features:
    - Configurable tiling parameters
    - Multi-scale inference
    - Parallel processing
    - Quality assessment
    - Performance optimization
    - Confidence threshold analysis
    """
    
    def __init__(self, 
                 model_path: str,
                 config: SAHIConfig = None,
                 experiment_name: str = "sahi_inference"):
        """
        Initialize SAHI inference module
        
        Args:
            model_path: Path to trained model
            config: SAHI configuration
            experiment_name: Name for experiment tracking
        """
        self.model_path = model_path
        self.config = config if config is not None else SAHIConfig()
        self.experiment_name = experiment_name
        
        # Initialize TESSD framework
        self.tessd = TESSDFramework(
            model_path=model_path,
            confidence_threshold=self.config.confidence_threshold,
            device=self.config.device,
            tile_size=self.config.tile_size,
            overlap_ratio=self.config.overlap_ratio,
            experiment_name=experiment_name
        )
        
        # Performance tracking
        self.inference_stats = {
            'total_images': 0,
            'total_detections': 0,
            'total_processing_time': 0,
            'average_processing_time': 0,
            'tiles_processed': 0
        }
        
        logger.info(f"SAHI Inference Module initialized: {experiment_name}")
        logger.info(f"Config: {self.config}")
    
    def predict_single_image(self, 
                           image: np.ndarray,
                           image_id: str = None,
                           save_visualization: bool = False,
                           output_dir: str = None) -> InferenceResult:
        """
        Perform SAHI inference on a single image
        
        Args:
            image: Input image in BGR format
            image_id: Unique identifier for the image
            save_visualization: Whether to save detection visualization
            output_dir: Directory to save outputs
            
        Returns:
            InferenceResult with detections and metadata
        """
        start_time = time.time()
        
        if image_id is None:
            image_id = f"image_{int(time.time())}"
        
        # Multi-scale inference if enabled
        if self.config.use_multiscale:
            detections = self._multiscale_inference(image)
        else:
            detections = self._single_scale_inference(image)
        
        processing_time = time.time() - start_time
        
        # Calculate tile information
        tile_info = self._calculate_tile_info(image.shape[:2])
        
        # Quality assessment
        quality_metrics = self._assess_detection_quality(detections, image.shape[:2])
        
        # Create metadata
        metadata = {
            'image_shape': image.shape,
            'config': self.config.__dict__,
            'num_detections': len(detections),
            'processing_time': processing_time,
            'model_path': self.model_path
        }
        
        # Save visualization if requested
        if save_visualization and output_dir:
            self._save_visualization(image, detections, image_id, output_dir)
        
        # Update statistics
        self._update_stats(processing_time, len(detections), tile_info['total_tiles'])
        
        return InferenceResult(
            image_id=image_id,
            detections=detections,
            metadata=metadata,
            processing_time=processing_time,
            tile_info=tile_info,
            quality_metrics=quality_metrics
        )
    
    def _single_scale_inference(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Perform single-scale SAHI inference"""
        return self.tessd.predict(
            image,
            use_sahi=True,
            slice_height=self.config.tile_size,
            slice_width=self.config.tile_size,
            overlap_height_ratio=self.config.overlap_ratio,
            overlap_width_ratio=self.config.overlap_ratio
        )
    
    def _multiscale_inference(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Perform multi-scale SAHI inference
        
        Args:
            image: Input image
            
        Returns:
            Combined detections from all scales
        """
        all_detections = []
        
        for scale_factor in self.config.scale_factors:
            # Resize image
            new_height = int(image.shape[0] * scale_factor)
            new_width = int(image.shape[1] * scale_factor)
            resized_image = cv2.resize(image, (new_width, new_height))
            
            # Perform detection
            scale_detections = self.tessd.predict(
                resized_image,
                use_sahi=True,
                slice_height=self.config.tile_size,
                slice_width=self.config.tile_size,
                overlap_height_ratio=self.config.overlap_ratio,
                overlap_width_ratio=self.config.overlap_ratio
            )
            
            # Scale detections back to original size
            for det in scale_detections:
                bbox = det['bbox']
                det['bbox'] = {
                    'x1': bbox['x1'] / scale_factor,
                    'y1': bbox['y1'] / scale_factor,
                    'x2': bbox['x2'] / scale_factor,
                    'y2': bbox['y2'] / scale_factor
                }
                det['scale_factor'] = scale_factor
            
            all_detections.extend(scale_detections)
        
        # Apply NMS across scales
        if all_detections:
            all_detections = self._apply_multiscale_nms(all_detections)
        
        return all_detections
    
    def _apply_multiscale_nms(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply Non-Maximum Suppression across multiple scales
        
        Args:
            detections: List of detections from different scales
            
        Returns:
            Filtered detections after NMS
        """
        if not detections:
            return []
        
        # Convert to format suitable for NMS
        boxes = []
        scores = []
        
        for det in detections:
            bbox = det['bbox']
            boxes.append([bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']])
            scores.append(det['score'])
        
        boxes = np.array(boxes)
        scores = np.array(scores)
        
        # Apply NMS
        indices = self._nms(boxes, scores, self.config.iou_threshold)
        
        return [detections[i] for i in indices]
    
    def _nms(self, boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
        """
        Non-Maximum Suppression implementation
        
        Args:
            boxes: Array of bounding boxes [x1, y1, x2, y2]
            scores: Array of confidence scores
            iou_threshold: IoU threshold for suppression
            
        Returns:
            Indices of boxes to keep
        """
        if len(boxes) == 0:
            return []
        
        # Calculate areas
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        
        # Sort by scores
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            # Keep the box with highest score
            i = order[0]
            keep.append(i)
            
            if order.size == 1:
                break
            
            # Calculate IoU with remaining boxes
            xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
            yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
            xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
            yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
            
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            
            intersection = w * h
            iou = intersection / (areas[i] + areas[order[1:]] - intersection)
            
            # Keep boxes with IoU below threshold
            indices = np.where(iou <= iou_threshold)[0]
            order = order[indices + 1]
        
        return keep
    
    def _calculate_tile_info(self, image_shape: Tuple[int, int]) -> Dict[str, Any]:
        """
        Calculate tiling information for the image
        
        Args:
            image_shape: (height, width) of the image
            
        Returns:
            Tiling information dictionary
        """
        height, width = image_shape
        
        # Calculate overlap in pixels
        overlap_pixels = int(self.config.tile_size * self.config.overlap_ratio)
        step_size = self.config.tile_size - overlap_pixels
        
        # Calculate number of tiles
        tiles_x = int(np.ceil((width - overlap_pixels) / step_size))
        tiles_y = int(np.ceil((height - overlap_pixels) / step_size))
        total_tiles = tiles_x * tiles_y
        
        # Calculate coverage
        effective_width = (tiles_x - 1) * step_size + self.config.tile_size
        effective_height = (tiles_y - 1) * step_size + self.config.tile_size
        
        coverage_x = min(1.0, effective_width / width)
        coverage_y = min(1.0, effective_height / height)
        
        return {
            'tiles_x': tiles_x,
            'tiles_y': tiles_y,
            'total_tiles': total_tiles,
            'tile_size': self.config.tile_size,
            'overlap_ratio': self.config.overlap_ratio,
            'overlap_pixels': overlap_pixels,
            'step_size': step_size,
            'coverage_x': coverage_x,
            'coverage_y': coverage_y,
            'effective_resolution': (effective_width, effective_height)
        }
    
    def _assess_detection_quality(self, 
                                detections: List[Dict[str, Any]], 
                                image_shape: Tuple[int, int]) -> Dict[str, Any]:
        """
        Assess quality of detections
        
        Args:
            detections: List of detections
            image_shape: Image dimensions
            
        Returns:
            Quality metrics dictionary
        """
        if not detections:
            return {
                'num_detections': 0,
                'avg_confidence': 0,
                'confidence_std': 0,
                'detection_density': 0,
                'size_distribution': {},
                'spatial_distribution': {}
            }
        
        # Basic statistics
        confidences = [det['score'] for det in detections]
        avg_confidence = np.mean(confidences)
        confidence_std = np.std(confidences)
        
        # Detection density (detections per 1000 pixels²)
        image_area = image_shape[0] * image_shape[1]
        detection_density = len(detections) / (image_area / 1000000)  # per megapixel
        
        # Size distribution
        sizes = []
        for det in detections:
            bbox = det['bbox']
            width = bbox['x2'] - bbox['x1']
            height = bbox['y2'] - bbox['y1']
            area = width * height
            sizes.append(area)
        
        size_distribution = {
            'mean_size': np.mean(sizes) if sizes else 0,
            'std_size': np.std(sizes) if sizes else 0,
            'min_size': np.min(sizes) if sizes else 0,
            'max_size': np.max(sizes) if sizes else 0,
            'median_size': np.median(sizes) if sizes else 0
        }
        
        # Spatial distribution
        centers = []
        for det in detections:
            bbox = det['bbox']
            center_x = (bbox['x1'] + bbox['x2']) / 2
            center_y = (bbox['y1'] + bbox['y2']) / 2
            centers.append([center_x, center_y])
        
        spatial_distribution = {}
        if centers:
            centers = np.array(centers)
            spatial_distribution = {
                'spatial_coverage_x': (np.max(centers[:, 0]) - np.min(centers[:, 0])) / image_shape[1],
                'spatial_coverage_y': (np.max(centers[:, 1]) - np.min(centers[:, 1])) / image_shape[0],
                'center_of_mass_x': np.mean(centers[:, 0]) / image_shape[1],
                'center_of_mass_y': np.mean(centers[:, 1]) / image_shape[0]
            }
        
        return {
            'num_detections': len(detections),
            'avg_confidence': avg_confidence,
            'confidence_std': confidence_std,
            'detection_density': detection_density,
            'size_distribution': size_distribution,
            'spatial_distribution': spatial_distribution,
            'confidence_distribution': {
                'q25': np.percentile(confidences, 25),
                'q50': np.percentile(confidences, 50),
                'q75': np.percentile(confidences, 75),
                'q90': np.percentile(confidences, 90),
                'q95': np.percentile(confidences, 95)
            }
        }
    
    def _save_visualization(self, 
                          image: np.ndarray,
                          detections: List[Dict[str, Any]],
                          image_id: str,
                          output_dir: str):
        """
        Save detection visualization
        
        Args:
            image: Original image
            detections: Detection results
            image_id: Image identifier
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create visualization
        vis_image = image.copy()
        
        for det in detections:
            bbox = det['bbox']
            confidence = det['score']
            
            # Draw bounding box
            x1, y1, x2, y2 = int(bbox['x1']), int(bbox['y1']), int(bbox['x2']), int(bbox['y2'])
            
            # Color based on confidence
            if confidence >= 0.8:
                color = (0, 255, 0)  # Green for high confidence
            elif confidence >= 0.5:
                color = (0, 255, 255)  # Yellow for medium confidence
            else:
                color = (0, 0, 255)  # Red for low confidence
            
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            
            # Add confidence text
            text = f"{confidence:.2f}"
            cv2.putText(vis_image, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Save visualization
        output_file = output_path / f"{image_id}_detections.jpg"
        cv2.imwrite(str(output_file), vis_image)
    
    def batch_inference(self, 
                       image_paths: List[str],
                       output_dir: str = None,
                       save_visualizations: bool = True,
                       parallel_processing: bool = True,
                       max_workers: int = None) -> pd.DataFrame:
        """
        Perform batch inference on multiple images
        
        Args:
            image_paths: List of image file paths
            output_dir: Output directory for results
            save_visualizations: Whether to save detection visualizations
            parallel_processing: Whether to use parallel processing
            max_workers: Maximum number of worker threads
            
        Returns:
            DataFrame with inference results
        """
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting batch inference on {len(image_paths)} images")
        
        results = []
        
        if parallel_processing:
            # Parallel processing
            if max_workers is None:
                max_workers = min(4, len(image_paths))
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_path = {
                    executor.submit(
                        self._process_single_image, 
                        path, 
                        output_dir if save_visualizations else None
                    ): path for path in image_paths
                }
                
                for future in as_completed(future_to_path):
                    path = future_to_path[future]
                    try:
                        result = future.result()
                        if result:
                            results.append(result)
                    except Exception as e:
                        logger.error(f"Error processing {path}: {e}")
        else:
            # Sequential processing
            for path in image_paths:
                try:
                    result = self._process_single_image(
                        path, 
                        output_dir if save_visualizations else None
                    )
                    if result:
                        results.append(result)
                except Exception as e:
                    logger.error(f"Error processing {path}: {e}")
        
        # Convert results to DataFrame
        if results:
            df = self._results_to_dataframe(results)
            
            # Save results CSV
            if output_dir:
                csv_path = output_path / "batch_inference_results.csv"
                df.to_csv(csv_path, index=False)
                logger.info(f"Results saved to: {csv_path}")
            
            return df
        else:
            logger.warning("No successful inference results")
            return pd.DataFrame()
    
    def _process_single_image(self, image_path: str, output_dir: str = None) -> Optional[Dict[str, Any]]:
        """Process a single image for batch inference"""
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return None
        
        image_id = Path(image_path).stem
        
        result = self.predict_single_image(
            image, 
            image_id=image_id, 
            save_visualization=output_dir is not None,
            output_dir=output_dir
        )
        
        # Convert result to dictionary format
        return {
            'image_path': image_path,
            'image_id': result.image_id,
            'num_detections': len(result.detections),
            'processing_time': result.processing_time,
            'avg_confidence': result.quality_metrics['avg_confidence'],
            'detection_density': result.quality_metrics['detection_density'],
            'total_tiles': result.tile_info['total_tiles'],
            'coverage_x': result.tile_info['coverage_x'],
            'coverage_y': result.tile_info['coverage_y'],
            **result.quality_metrics['size_distribution'],
            **result.quality_metrics['spatial_distribution']
        }
    
    def _results_to_dataframe(self, results: List[Dict[str, Any]]) -> pd.DataFrame:
        """Convert results list to pandas DataFrame"""
        df = pd.DataFrame(results)
        
        # Add summary statistics
        df['detection_rate'] = df['num_detections'] > 0
        df['processing_speed'] = 1 / df['processing_time']  # images per second
        
        return df
    
    def _update_stats(self, processing_time: float, num_detections: int, num_tiles: int):
        """Update internal statistics"""
        self.inference_stats['total_images'] += 1
        self.inference_stats['total_detections'] += num_detections
        self.inference_stats['total_processing_time'] += processing_time
        self.inference_stats['tiles_processed'] += num_tiles
        
        # Calculate averages
        total_images = self.inference_stats['total_images']
        self.inference_stats['average_processing_time'] = (
            self.inference_stats['total_processing_time'] / total_images
        )
        self.inference_stats['average_detections_per_image'] = (
            self.inference_stats['total_detections'] / total_images
        )
        self.inference_stats['average_tiles_per_image'] = (
            self.inference_stats['tiles_processed'] / total_images
        )
    
    def analyze_confidence_thresholds(self, 
                                    image: np.ndarray,
                                    thresholds: List[float] = None,
                                    save_analysis: bool = True,
                                    output_dir: str = None) -> Dict[str, Any]:
        """
        Analyze performance across different confidence thresholds
        
        Args:
            image: Input image
            thresholds: List of confidence thresholds to test
            save_analysis: Whether to save analysis results
            output_dir: Output directory
            
        Returns:
            Threshold analysis results
        """
        if thresholds is None:
            thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
        logger.info(f"Analyzing confidence thresholds: {thresholds}")
        
        analysis_results = {}
        
        for threshold in thresholds:
            # Update confidence threshold
            original_threshold = self.config.confidence_threshold
            self.config.confidence_threshold = threshold
            self.tessd.confidence_threshold = threshold
            
            # Perform inference
            detections = self._single_scale_inference(image)
            
            # Calculate metrics
            confidences = [det['score'] for det in detections] if detections else []
            
            analysis_results[f"threshold_{threshold:.2f}"] = {
                'threshold': threshold,
                'num_detections': len(detections),
                'avg_confidence': np.mean(confidences) if confidences else 0,
                'min_confidence': np.min(confidences) if confidences else 0,
                'max_confidence': np.max(confidences) if confidences else 0,
                'confidence_std': np.std(confidences) if confidences else 0
            }
        
        # Restore original threshold
        self.config.confidence_threshold = original_threshold
        self.tessd.confidence_threshold = original_threshold
        
        # Save analysis if requested
        if save_analysis and output_dir:
            self._save_threshold_analysis(analysis_results, output_dir)
        
        return analysis_results
    
    def _save_threshold_analysis(self, 
                               analysis_results: Dict[str, Any], 
                               output_dir: str):
        """Save confidence threshold analysis results"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Convert to DataFrame
        df_data = []
        for key, result in analysis_results.items():
            df_data.append(result)
        
        df = pd.DataFrame(df_data)
        
        # Save CSV
        csv_path = output_path / "confidence_threshold_analysis.csv"
        df.to_csv(csv_path, index=False)
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # Plot number of detections vs threshold
        plt.subplot(2, 2, 1)
        plt.plot(df['threshold'], df['num_detections'], 'b-o')
        plt.xlabel('Confidence Threshold')
        plt.ylabel('Number of Detections')
        plt.title('Detections vs Confidence Threshold')
        plt.grid(True)
        
        # Plot average confidence vs threshold
        plt.subplot(2, 2, 2)
        plt.plot(df['threshold'], df['avg_confidence'], 'r-o')
        plt.xlabel('Confidence Threshold')
        plt.ylabel('Average Confidence')
        plt.title('Average Confidence vs Threshold')
        plt.grid(True)
        
        # Plot confidence range
        plt.subplot(2, 2, 3)
        plt.fill_between(df['threshold'], df['min_confidence'], df['max_confidence'], alpha=0.3)
        plt.plot(df['threshold'], df['avg_confidence'], 'g-o', label='Average')
        plt.xlabel('Confidence Threshold')
        plt.ylabel('Confidence')
        plt.title('Confidence Range vs Threshold')
        plt.legend()
        plt.grid(True)
        
        # Plot detection efficiency (detections per confidence unit)
        plt.subplot(2, 2, 4)
        efficiency = df['num_detections'] / (df['avg_confidence'] + 1e-6)
        plt.plot(df['threshold'], efficiency, 'm-o')
        plt.xlabel('Confidence Threshold')
        plt.ylabel('Detection Efficiency')
        plt.title('Detection Efficiency vs Threshold')
        plt.grid(True)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = output_path / "confidence_threshold_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Threshold analysis saved to: {output_path}")
    
    def get_inference_statistics(self) -> Dict[str, Any]:
        """Get current inference statistics"""
        return self.inference_stats.copy()
    
    def reset_statistics(self):
        """Reset inference statistics"""
        self.inference_stats = {
            'total_images': 0,
            'total_detections': 0,
            'total_processing_time': 0,
            'average_processing_time': 0,
            'tiles_processed': 0
        }


def create_sahi_config(tile_size: int = 640,
                      overlap_ratio: float = 0.2,
                      confidence_threshold: float = 0.20,
                      use_multiscale: bool = False) -> SAHIConfig:
    """
    Create SAHI configuration with common settings
    
    Args:
        tile_size: Size of tiles for SAHI
        overlap_ratio: Overlap ratio between tiles
        confidence_threshold: Minimum confidence threshold
        use_multiscale: Whether to use multi-scale inference
        
    Returns:
        SAHIConfig object
    """
    return SAHIConfig(
        tile_size=tile_size,
        overlap_ratio=overlap_ratio,
        confidence_threshold=confidence_threshold,
        use_multiscale=use_multiscale
    )


def main():
    """Main function for testing SAHI inference module"""
    import argparse
    
    parser = argparse.ArgumentParser(description="SAHI Enhanced Inference Module")
    parser.add_argument("--model", required=True, help="Path to model weights")
    parser.add_argument("--image", help="Path to single image for inference")
    parser.add_argument("--image-dir", help="Path to directory of images for batch inference")
    parser.add_argument("--output", default="./sahi_inference_results", help="Output directory")
    parser.add_argument("--tile-size", type=int, default=640, help="Tile size for SAHI")
    parser.add_argument("--overlap", type=float, default=0.2, help="Overlap ratio")
    parser.add_argument("--confidence", type=float, default=0.20, help="Confidence threshold")
    parser.add_argument("--multiscale", action="store_true", help="Use multi-scale inference")
    parser.add_argument("--analyze-thresholds", action="store_true", help="Analyze confidence thresholds")
    
    args = parser.parse_args()
    
    # Create SAHI configuration
    config = create_sahi_config(
        tile_size=args.tile_size,
        overlap_ratio=args.overlap,
        confidence_threshold=args.confidence,
        use_multiscale=args.multiscale
    )
    
    # Initialize inference module
    inference_module = SAHIInferenceModule(
        model_path=args.model,
        config=config,
        experiment_name="sahi_inference_test"
    )
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.image:
        # Single image inference
        image = cv2.imread(args.image)
        if image is None:
            logger.error(f"Failed to load image: {args.image}")
            return
        
        result = inference_module.predict_single_image(
            image,
            image_id=Path(args.image).stem,
            save_visualization=True,
            output_dir=str(output_dir)
        )
        
        print(f"✅ Inference completed!")
        print(f"📊 Detections: {len(result.detections)}")
        print(f"⏱️ Processing time: {result.processing_time:.2f}s")
        print(f"🎯 Average confidence: {result.quality_metrics['avg_confidence']:.3f}")
        
        # Analyze thresholds if requested
        if args.analyze_thresholds:
            analysis = inference_module.analyze_confidence_thresholds(
                image,
                save_analysis=True,
                output_dir=str(output_dir)
            )
            print(f"🔍 Threshold analysis completed")
    
    elif args.image_dir:
        # Batch inference
        image_dir = Path(args.image_dir)
        image_paths = []
        
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_paths.extend(list(image_dir.glob(ext)))
        
        if not image_paths:
            logger.error(f"No images found in: {image_dir}")
            return
        
        results_df = inference_module.batch_inference(
            [str(p) for p in image_paths],
            output_dir=str(output_dir),
            save_visualizations=True
        )
        
        print(f"✅ Batch inference completed!")
        print(f"📊 Total images: {len(results_df)}")
        print(f"🎯 Average detections per image: {results_df['num_detections'].mean():.1f}")
        print(f"⏱️ Average processing time: {results_df['processing_time'].mean():.2f}s")
        print(f"📁 Results saved to: {output_dir}")
    
    else:
        print("❌ Please specify either --image or --image-dir")


if __name__ == "__main__":
    main() 