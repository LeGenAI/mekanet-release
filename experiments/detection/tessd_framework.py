"""
TESSD (Tiling-Enhanced Semi-Supervised Detection) Framework
Core implementation for MekaNet megakaryocyte detection with cross-institutional validation
"""

import cv2
import numpy as np
import pandas as pd
import torch
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import seaborn as sns

# Import base detector
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from mekanet.models.yolo_sahi import YoloSahiDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TESSDFramework(YoloSahiDetector):
    """
    TESSD Framework: Tiling-Enhanced Semi-Supervised Detection
    
    Extends YoloSahiDetector with:
    - Enhanced morphological feature extraction
    - Cross-institutional validation capabilities
    - Comprehensive evaluation metrics
    - Reproducible experiment configurations
    """
    
    def __init__(self, 
                 model_path: str,
                 confidence_threshold: float = 0.20,
                 device: str = 'cpu',
                 tile_size: int = 640,
                 overlap_ratio: float = 0.2,
                 experiment_name: str = "tessd_experiment"):
        """
        Initialize TESSD Framework
        
        Args:
            model_path: Path to trained YOLO model (e.g., epoch60.pt)
            confidence_threshold: Detection confidence threshold
            device: Computation device ('cpu' or 'cuda')
            tile_size: Size of image tiles for SAHI
            overlap_ratio: Overlap ratio between tiles
            experiment_name: Name for experiment tracking
        """
        super().__init__(model_path, confidence_threshold, device)
        
        self.tile_size = tile_size
        self.overlap_ratio = overlap_ratio
        self.experiment_name = experiment_name
        
        # Morphological feature extraction parameters
        self.feature_config = {
            'dbscan_eps': 50,  # DBSCAN clustering epsilon
            'dbscan_min_samples': 2,  # Minimum samples for cluster
            'size_bins': 10,  # Number of bins for size distribution
            'spatial_grid_size': 5  # Grid size for spatial analysis
        }
        
        logger.info(f"TESSD Framework initialized: {experiment_name}")
        logger.info(f"Tile size: {tile_size}x{tile_size}, Overlap: {overlap_ratio}")
    
    def detect_with_metadata(self, 
                           image: np.ndarray,
                           image_id: str = None,
                           institution: str = None,
                           use_sahi: bool = True) -> Dict[str, Any]:
        """
        Enhanced detection with metadata tracking for experiments
        
        Args:
            image: Input image in BGR format
            image_id: Unique identifier for the image
            institution: Institution identifier (e.g., 'B_hospital', 'S_hospital')
            use_sahi: Whether to use SAHI tiling
            
        Returns:
            Dict containing detections, metadata, and timing information
        """
        start_time = time.time()
        
        # Perform detection
        detections = self.predict(
            image, 
            use_sahi=use_sahi,
            slice_height=self.tile_size,
            slice_width=self.tile_size,
            overlap_height_ratio=self.overlap_ratio,
            overlap_width_ratio=self.overlap_ratio
        )
        
        detection_time = time.time() - start_time
        
        # Extract morphological features
        features = self.extract_morphological_features(detections, image.shape[:2])
        
        return {
            'image_id': image_id,
            'institution': institution,
            'detections': detections,
            'num_detections': len(detections),
            'morphological_features': features,
            'detection_time': detection_time,
            'image_shape': image.shape,
            'confidence_threshold': self.confidence_threshold,
            'used_sahi': use_sahi,
            'tile_size': self.tile_size if use_sahi else None
        }
    
    def extract_morphological_features(self, 
                                     detections: List[Dict[str, Any]], 
                                     image_shape: Tuple[int, int]) -> Dict[str, float]:
        """
        Extract comprehensive morphological features from detections
        
        Args:
            detections: List of detection results
            image_shape: Image dimensions (height, width)
            
        Returns:
            Dictionary of morphological features
        """
        if not detections:
            return self._get_empty_features()
        
        # Extract basic measurements
        sizes = []
        centers = []
        confidences = []
        
        for det in detections:
            bbox = det['bbox']
            # Calculate size (area of bounding box)
            width = bbox['x2'] - bbox['x1']
            height = bbox['y2'] - bbox['y1']
            size = width * height
            sizes.append(size)
            
            # Calculate center coordinates
            center_x = (bbox['x1'] + bbox['x2']) / 2
            center_y = (bbox['y1'] + bbox['y2']) / 2
            centers.append([center_x, center_y])
            
            confidences.append(det['score'])
        
        sizes = np.array(sizes)
        centers = np.array(centers)
        confidences = np.array(confidences)
        
        # Basic size statistics
        features = {
            'Num_Megakaryocytes': len(detections),
            'Avg_Size': np.mean(sizes),
            'Std_Size': np.std(sizes),
            'Max_Size': np.max(sizes),
            'Min_Size': np.min(sizes),
            'Median_Size': np.median(sizes),
            'Size_Range': np.max(sizes) - np.min(sizes),
        }
        
        # Confidence statistics
        features.update({
            'Avg_Confidence': np.mean(confidences),
            'Std_Confidence': np.std(confidences),
            'Min_Confidence': np.min(confidences)
        })
        
        # Spatial analysis
        if len(centers) > 1:
            # Nearest neighbor distances
            distances = pdist(centers)
            distance_matrix = squareform(distances)
            
            # For each point, find distance to nearest neighbor
            nnd = []
            for i in range(len(centers)):
                row = distance_matrix[i]
                row = row[row > 0]  # Exclude distance to self
                if len(row) > 0:
                    nnd.append(np.min(row))
            
            if nnd:
                features.update({
                    'Avg_NND': np.mean(nnd),
                    'Std_NND': np.std(nnd),
                    'Median_NND': np.median(nnd)
                })
            else:
                features.update({
                    'Avg_NND': 0,
                    'Std_NND': 0,
                    'Median_NND': 0
                })
            
            # Local density calculation
            local_densities = []
            for i, center in enumerate(centers):
                # Count neighbors within radius
                radius = features['Avg_NND'] * 2  # Use 2x average NND as radius
                distances_from_point = distance_matrix[i]
                neighbors_in_radius = np.sum(distances_from_point <= radius) - 1  # Exclude self
                local_densities.append(neighbors_in_radius)
            
            features.update({
                'Avg_Local_Density': np.mean(local_densities),
                'Std_Local_Density': np.std(local_densities)
            })
        else:
            features.update({
                'Avg_NND': 0,
                'Std_NND': 0,
                'Median_NND': 0,
                'Avg_Local_Density': 0,
                'Std_Local_Density': 0
            })
        
        # Clustering analysis
        cluster_features = self._analyze_clustering(centers)
        features.update(cluster_features)
        
        # Spatial distribution
        spatial_features = self._analyze_spatial_distribution(centers, image_shape)
        features.update(spatial_features)
        
        return features
    
    def _analyze_clustering(self, centers: np.ndarray) -> Dict[str, float]:
        """Analyze megakaryocyte clustering patterns"""
        if len(centers) < 2:
            return {
                'Num_Clusters': 0,
                'Avg_Cluster_Size': 0,
                'Std_Cluster_Size': 0,
                'Max_Cluster_Size': 0,
                'Min_Cluster_Size': 0,
                'Clustering_Ratio': 0
            }
        
        # DBSCAN clustering
        clustering = DBSCAN(
            eps=self.feature_config['dbscan_eps'],
            min_samples=self.feature_config['dbscan_min_samples']
        ).fit(centers)
        
        labels = clustering.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        if n_clusters == 0:
            cluster_sizes = []
        else:
            cluster_sizes = [np.sum(labels == i) for i in range(n_clusters)]
        
        features = {
            'Num_Clusters': n_clusters,
            'Clustering_Ratio': n_clusters / len(centers) if len(centers) > 0 else 0
        }
        
        if cluster_sizes:
            features.update({
                'Avg_Cluster_Size': np.mean(cluster_sizes),
                'Std_Cluster_Size': np.std(cluster_sizes),
                'Max_Cluster_Size': np.max(cluster_sizes),
                'Min_Cluster_Size': np.min(cluster_sizes)
            })
        else:
            features.update({
                'Avg_Cluster_Size': 0,
                'Std_Cluster_Size': 0,
                'Max_Cluster_Size': 0,
                'Min_Cluster_Size': 0
            })
        
        return features
    
    def _analyze_spatial_distribution(self, 
                                    centers: np.ndarray, 
                                    image_shape: Tuple[int, int]) -> Dict[str, float]:
        """Analyze spatial distribution of megakaryocytes"""
        if len(centers) == 0:
            return {
                'Spatial_Spread_X': 0,
                'Spatial_Spread_Y': 0,
                'Spatial_Coverage': 0,
                'Central_Tendency': 0
            }
        
        height, width = image_shape
        
        # Spatial spread
        x_coords = centers[:, 0]
        y_coords = centers[:, 1]
        
        features = {
            'Spatial_Spread_X': np.std(x_coords) / width,  # Normalized by image width
            'Spatial_Spread_Y': np.std(y_coords) / height,  # Normalized by image height
        }
        
        # Coverage area (convex hull area normalized by image area)
        if len(centers) >= 3:
            from scipy.spatial import ConvexHull
            try:
                hull = ConvexHull(centers)
                coverage = hull.volume / (width * height)
                features['Spatial_Coverage'] = coverage
            except:
                features['Spatial_Coverage'] = 0
        else:
            features['Spatial_Coverage'] = 0
        
        # Central tendency (how clustered around image center)
        image_center = np.array([width/2, height/2])
        distances_to_center = np.linalg.norm(centers - image_center, axis=1)
        max_distance = np.sqrt((width/2)**2 + (height/2)**2)
        central_tendency = 1 - np.mean(distances_to_center) / max_distance
        features['Central_Tendency'] = central_tendency
        
        return features
    
    def _get_empty_features(self) -> Dict[str, float]:
        """Return zero-filled feature dictionary for images with no detections"""
        return {
            'Num_Megakaryocytes': 0,
            'Avg_Size': 0, 'Std_Size': 0, 'Max_Size': 0, 'Min_Size': 0,
            'Median_Size': 0, 'Size_Range': 0,
            'Avg_Confidence': 0, 'Std_Confidence': 0, 'Min_Confidence': 0,
            'Avg_NND': 0, 'Std_NND': 0, 'Median_NND': 0,
            'Avg_Local_Density': 0, 'Std_Local_Density': 0,
            'Num_Clusters': 0, 'Avg_Cluster_Size': 0, 'Std_Cluster_Size': 0,
            'Max_Cluster_Size': 0, 'Min_Cluster_Size': 0, 'Clustering_Ratio': 0,
            'Spatial_Spread_X': 0, 'Spatial_Spread_Y': 0,
            'Spatial_Coverage': 0, 'Central_Tendency': 0
        }
    
    def batch_process_images(self, 
                           image_paths: List[str],
                           output_dir: str = None,
                           save_visualizations: bool = True) -> pd.DataFrame:
        """
        Process multiple images and extract features for analysis
        
        Args:
            image_paths: List of paths to images
            output_dir: Directory to save results
            save_visualizations: Whether to save detection visualizations
            
        Returns:
            DataFrame with detection results and features
        """
        results = []
        
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            if save_visualizations:
                viz_path = output_path / "visualizations"
                viz_path.mkdir(exist_ok=True)
        
        for img_path in image_paths:
            try:
                # Load image
                image = cv2.imread(str(img_path))
                if image is None:
                    logger.warning(f"Failed to load image: {img_path}")
                    continue
                
                # Extract image metadata
                img_name = Path(img_path).stem
                institution = "S_hospital" if img_name.startswith('S') else "B_hospital"
                
                # Perform detection
                result = self.detect_with_metadata(
                    image, 
                    image_id=img_name,
                    institution=institution
                )
                
                # Save visualization if requested
                if save_visualizations and output_dir:
                    viz_image = self.visualize_predictions(
                        image, 
                        result['detections'],
                        show_confidence=True
                    )
                    viz_file = viz_path / f"{img_name}_detection.jpg"
                    cv2.imwrite(str(viz_file), viz_image)
                
                # Flatten result for DataFrame
                flat_result = {
                    'image_id': result['image_id'],
                    'institution': result['institution'],
                    'num_detections': result['num_detections'],
                    'detection_time': result['detection_time'],
                    'confidence_threshold': result['confidence_threshold']
                }
                flat_result.update(result['morphological_features'])
                
                results.append(flat_result)
                
                logger.info(f"Processed {img_name}: {result['num_detections']} detections")
                
            except Exception as e:
                logger.error(f"Error processing {img_path}: {str(e)}")
                continue
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Save results if output directory specified
        if output_dir:
            results_file = output_path / f"{self.experiment_name}_results.csv"
            df.to_csv(results_file, index=False)
            logger.info(f"Results saved to: {results_file}")
        
        return df
    
    def compare_confidence_thresholds(self, 
                                    image: np.ndarray,
                                    thresholds: List[float] = [0.15, 0.20, 0.25]) -> Dict[str, Any]:
        """
        Compare detection performance across different confidence thresholds
        Based on SAHI1.ipynb experiments
        """
        results = {}
        original_threshold = self.confidence_threshold
        
        for threshold in thresholds:
            self.confidence_threshold = threshold
            # Reload models with new threshold
            self._load_models()
            
            # Perform detection
            detections = self.predict(image, use_sahi=True)
            features = self.extract_morphological_features(detections, image.shape[:2])
            
            results[f"threshold_{threshold}"] = {
                'num_detections': len(detections),
                'avg_confidence': np.mean([d['score'] for d in detections]) if detections else 0,
                'features': features
            }
        
        # Restore original threshold
        self.confidence_threshold = original_threshold
        self._load_models()
        
        return results 