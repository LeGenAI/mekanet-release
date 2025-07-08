"""
Feature Extraction Module for Megakaryocyte Analysis

This module extracts comprehensive morphological features from detected megakaryocytes
including size statistics, spatial distribution, density metrics, and shape characteristics.
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN
from typing import List, Dict, Any, Tuple, Optional
import cv2
import logging

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Comprehensive feature extractor for megakaryocyte analysis
    
    This class extracts various morphological and spatial features from
    detected megakaryocytes for downstream classification tasks.
    """
    
    def __init__(self):
        """Initialize the feature extractor"""
        self.feature_names = [
            # Size features
            'Avg_Size', 'Std_Size', 'Max_Size', 'Min_Size', 'Median_Size', 'Size_Range',
            # Spatial features  
            'Avg_NND', 'Std_NND', 'Median_NND',
            # Density features
            'Avg_Local_Density', 'Std_Local_Density',
            # Shape features
            'Avg_Ellipticity', 'Std_Ellipticity', 'Max_Ellipticity', 'Min_Ellipticity',
            # Clustering features
            'Num_Clusters', 'Avg_Cluster_Size', 'Std_Cluster_Size',
            # Count features
            'Num_Megakaryocytes'
        ]
    
    def extract_features(self, 
                        detections: List[Dict[str, Any]], 
                        image_shape: Optional[Tuple[int, int]] = None) -> Dict[str, float]:
        """
        Extract comprehensive features from megakaryocyte detections
        
        Args:
            detections: List of detection results with bbox coordinates
            image_shape: Optional image shape (height, width) for density calculations
            
        Returns:
            Dict containing extracted features
        """
        if not detections:
            return self._get_zero_features()
        
        try:
            # Extract basic measurements
            centers = self._extract_centers(detections)
            sizes = self._extract_sizes(detections)
            ellipticities = self._extract_ellipticities(detections)
            
            # Calculate size features
            size_features = self._calculate_size_features(sizes)
            
            # Calculate spatial features
            spatial_features = self._calculate_spatial_features(centers)
            
            # Calculate density features
            density_features = self._calculate_density_features(centers, image_shape)
            
            # Calculate shape features
            shape_features = self._calculate_shape_features(ellipticities)
            
            # Calculate clustering features
            clustering_features = self._calculate_clustering_features(centers)
            
            # Combine all features
            features = {
                **size_features,
                **spatial_features,
                **density_features,
                **shape_features,
                **clustering_features,
                'Num_Megakaryocytes': len(detections)
            }
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            return self._get_zero_features()
    
    def _extract_centers(self, detections: List[Dict[str, Any]]) -> np.ndarray:
        """Extract center coordinates from detections"""
        centers = []
        for det in detections:
            bbox = det['bbox']
            center_x = (bbox['x1'] + bbox['x2']) / 2
            center_y = (bbox['y1'] + bbox['y2']) / 2
            centers.append([center_x, center_y])
        return np.array(centers)
    
    def _extract_sizes(self, detections: List[Dict[str, Any]]) -> np.ndarray:
        """Extract size measurements from detections"""
        sizes = []
        for det in detections:
            bbox = det['bbox']
            width = bbox['x2'] - bbox['x1']
            height = bbox['y2'] - bbox['y1']
            # Use area as size metric
            size = width * height
            sizes.append(size)
        return np.array(sizes)
    
    def _extract_ellipticities(self, detections: List[Dict[str, Any]]) -> np.ndarray:
        """Extract ellipticity measurements from detections"""
        ellipticities = []
        for det in detections:
            bbox = det['bbox']
            width = bbox['x2'] - bbox['x1']
            height = bbox['y2'] - bbox['y1']
            
            # Calculate ellipticity as aspect ratio
            if height > 0:
                ellipticity = width / height
            else:
                ellipticity = 1.0
                
            ellipticities.append(ellipticity)
        return np.array(ellipticities)
    
    def _calculate_size_features(self, sizes: np.ndarray) -> Dict[str, float]:
        """Calculate size-related features"""
        if len(sizes) == 0:
            return {
                'Avg_Size': 0.0, 'Std_Size': 0.0, 'Max_Size': 0.0,
                'Min_Size': 0.0, 'Median_Size': 0.0, 'Size_Range': 0.0
            }
        
        return {
            'Avg_Size': float(np.mean(sizes)),
            'Std_Size': float(np.std(sizes)),
            'Max_Size': float(np.max(sizes)),
            'Min_Size': float(np.min(sizes)),
            'Median_Size': float(np.median(sizes)),
            'Size_Range': float(np.max(sizes) - np.min(sizes))
        }
    
    def _calculate_spatial_features(self, centers: np.ndarray) -> Dict[str, float]:
        """Calculate spatial distribution features"""
        if len(centers) < 2:
            return {
                'Avg_NND': 0.0, 'Std_NND': 0.0, 'Median_NND': 0.0
            }
        
        # Calculate nearest neighbor distances
        distances = pdist(centers)
        distance_matrix = squareform(distances)
        
        # Set diagonal to infinity to exclude self-distances
        np.fill_diagonal(distance_matrix, np.inf)
        
        # Find nearest neighbor distance for each point
        nearest_distances = np.min(distance_matrix, axis=1)
        
        return {
            'Avg_NND': float(np.mean(nearest_distances)),
            'Std_NND': float(np.std(nearest_distances)),
            'Median_NND': float(np.median(nearest_distances))
        }
    
    def _calculate_density_features(self, 
                                   centers: np.ndarray, 
                                   image_shape: Optional[Tuple[int, int]] = None) -> Dict[str, float]:
        """Calculate local density features"""
        if len(centers) < 2:
            return {
                'Avg_Local_Density': 0.0, 'Std_Local_Density': 0.0
            }
        
        # Calculate local density using k-nearest neighbors (k=5)
        k = min(5, len(centers) - 1)
        distances = pdist(centers)
        distance_matrix = squareform(distances)
        np.fill_diagonal(distance_matrix, np.inf)
        
        local_densities = []
        for i in range(len(centers)):
            # Get k nearest distances
            nearest_k_distances = np.sort(distance_matrix[i])[:k]
            # Calculate average distance to k nearest neighbors
            avg_distance = np.mean(nearest_k_distances)
            # Density is inverse of average distance
            local_density = 1.0 / (avg_distance + 1e-6)
            local_densities.append(local_density)
        
        local_densities = np.array(local_densities)
        
        return {
            'Avg_Local_Density': float(np.mean(local_densities)),
            'Std_Local_Density': float(np.std(local_densities))
        }
    
    def _calculate_shape_features(self, ellipticities: np.ndarray) -> Dict[str, float]:
        """Calculate shape-related features"""
        if len(ellipticities) == 0:
            return {
                'Avg_Ellipticity': 0.0, 'Std_Ellipticity': 0.0,
                'Max_Ellipticity': 0.0, 'Min_Ellipticity': 0.0
            }
        
        return {
            'Avg_Ellipticity': float(np.mean(ellipticities)),
            'Std_Ellipticity': float(np.std(ellipticities)),
            'Max_Ellipticity': float(np.max(ellipticities)),
            'Min_Ellipticity': float(np.min(ellipticities))
        }
    
    def _calculate_clustering_features(self, centers: np.ndarray) -> Dict[str, float]:
        """Calculate clustering-based features using DBSCAN"""
        if len(centers) < 3:
            return {
                'Num_Clusters': 0.0, 'Avg_Cluster_Size': 0.0, 'Std_Cluster_Size': 0.0
            }
        
        try:
            # Use DBSCAN for clustering
            # eps is set based on typical megakaryocyte spacing
            eps = np.mean(pdist(centers)) * 0.5
            min_samples = max(2, len(centers) // 10)
            
            clustering = DBSCAN(eps=eps, min_samples=min_samples)
            cluster_labels = clustering.fit_predict(centers)
            
            # Count clusters (excluding noise points labeled as -1)
            unique_labels = set(cluster_labels)
            num_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
            
            # Calculate cluster sizes
            cluster_sizes = []
            for label in unique_labels:
                if label != -1:  # Exclude noise
                    cluster_size = np.sum(cluster_labels == label)
                    cluster_sizes.append(cluster_size)
            
            if cluster_sizes:
                avg_cluster_size = np.mean(cluster_sizes)
                std_cluster_size = np.std(cluster_sizes)
            else:
                avg_cluster_size = 0.0
                std_cluster_size = 0.0
            
            return {
                'Num_Clusters': float(num_clusters),
                'Avg_Cluster_Size': float(avg_cluster_size),
                'Std_Cluster_Size': float(std_cluster_size)
            }
            
        except Exception as e:
            logger.warning(f"Error in clustering analysis: {str(e)}")
            return {
                'Num_Clusters': 0.0, 'Avg_Cluster_Size': 0.0, 'Std_Cluster_Size': 0.0
            }
    
    def _get_zero_features(self) -> Dict[str, float]:
        """Return dictionary with all features set to zero"""
        return {name: 0.0 for name in self.feature_names}
    
    def extract_batch_features(self, 
                              detection_list: List[List[Dict[str, Any]]], 
                              image_shapes: Optional[List[Tuple[int, int]]] = None) -> pd.DataFrame:
        """
        Extract features for a batch of detection results
        
        Args:
            detection_list: List of detection results for each image
            image_shapes: Optional list of image shapes
            
        Returns:
            DataFrame with extracted features for each image
        """
        if image_shapes is None:
            image_shapes = [None] * len(detection_list)
        
        features_list = []
        for i, (detections, img_shape) in enumerate(zip(detection_list, image_shapes)):
            features = self.extract_features(detections, img_shape)
            features['Image_Index'] = i
            features_list.append(features)
        
        return pd.DataFrame(features_list)


def extract_morphological_features(detections: List[Dict[str, Any]], 
                                  image_shape: Optional[Tuple[int, int]] = None) -> Dict[str, float]:
    """
    Convenience function to extract morphological features
    
    Args:
        detections: List of detection results
        image_shape: Optional image shape
        
    Returns:
        Dict containing extracted features
    """
    extractor = FeatureExtractor()
    return extractor.extract_features(detections, image_shape)