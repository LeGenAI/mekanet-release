"""
YOLO + SAHI Detection Model for Megakaryocyte Detection

This module implements the YoloSahiDetector class which combines YOLO object detection
with SAHI (Slicing Aided Hyper Inference) for detecting megakaryocytes in bone marrow images.
"""

import cv2
import numpy as np
import os
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from ultralytics import YOLO
from typing import List, Dict, Any, Optional, Tuple
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class YoloSahiDetector:
    """
    YOLO + SAHI based megakaryocyte detector
    
    This class provides both standard YOLO detection and SAHI-enhanced detection
    for megakaryocytes in bone marrow histopathology images.
    """
    
    def __init__(self, 
                 model_path: str,
                 confidence_threshold: float = 0.20,
                 device: str = 'cpu'):
        """
        Initialize the YOLO+SAHI detector
        
        Args:
            model_path (str): Path to the trained YOLO model weights
            confidence_threshold (float): Confidence threshold for detections
            device (str): Device to run inference on ('cuda' or 'cpu')
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.direct_model = None  # Standard YOLO model
        self.sahi_model = None    # SAHI wrapper model
        
        self._load_models()
        
    def _load_models(self):
        """Load YOLO and SAHI models"""
        try:
            start_time = time.time()
            
            # Load standard YOLO model
            self.direct_model = YOLO(self.model_path)
            
            # Move model to specified device
            try:
                self.direct_model.to(self.device)
                logger.info(f"YOLO model moved to {self.device}")
            except Exception as e:
                logger.warning(f"Failed to move YOLO model to {self.device}: {e}")
            
            # Load SAHI model wrapper
            self.sahi_model = AutoDetectionModel.from_pretrained(
                model_type="yolov8",
                model_path=self.model_path,
                confidence_threshold=self.confidence_threshold,
                device=self.device
            )
            
            logger.info(f"Models loaded successfully in {time.time() - start_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Failed to load models: {str(e)}")
            raise
    
    def predict_with_sahi(self, 
                         image: np.ndarray, 
                         slice_height: int = 640,
                         slice_width: int = 640,
                         overlap_height_ratio: float = 0.2,
                         overlap_width_ratio: float = 0.2) -> List[Dict[str, Any]]:
        """
        Perform detection using SAHI for large images
        
        Args:
            image (np.ndarray): Input image in BGR format
            slice_height (int): Height of each slice
            slice_width (int): Width of each slice
            overlap_height_ratio (float): Height overlap ratio between slices
            overlap_width_ratio (float): Width overlap ratio between slices
            
        Returns:
            List[Dict]: List of detection results with bbox coordinates, scores, and categories
        """
        try:
            # Convert BGR to RGB (SAHI uses RGB images)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            logger.info(f"Image size: {image.shape[1]}x{image.shape[0]}, "
                       f"Slice size: {slice_width}x{slice_height}")
            
            # Perform sliced prediction
            result = get_sliced_prediction(
                image_rgb,
                self.sahi_model,
                slice_height=slice_height,
                slice_width=slice_width,
                overlap_height_ratio=overlap_height_ratio,
                overlap_width_ratio=overlap_width_ratio,
                perform_standard_pred=False  # Skip standard prediction on full image
            )
            
            # Convert results to standardized format
            predictions = []
            for pred in result.object_prediction_list:
                predictions.append({
                    'bbox': {
                        'x1': int(pred.bbox.minx),
                        'y1': int(pred.bbox.miny),
                        'x2': int(pred.bbox.maxx),
                        'y2': int(pred.bbox.maxy)
                    },
                    'score': float(pred.score.value),
                    'category_id': int(pred.category.id) if hasattr(pred.category, 'id') else 0,
                    'category_name': pred.category.name if hasattr(pred.category, 'name') else 'megakaryocyte'
                })
            
            logger.info(f"SAHI model detected {len(predictions)} objects")
            return predictions
        
        except Exception as e:
            logger.error(f"Error during SAHI prediction: {str(e)}")
            raise
    
    def predict_without_sahi(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Perform detection using standard YOLO (without SAHI)
        
        Args:
            image (np.ndarray): Input image in BGR format
            
        Returns:
            List[Dict]: List of detection results
        """
        try:
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Perform prediction
            results = self.direct_model.predict(
                source=image_rgb, 
                conf=self.confidence_threshold,
                device=self.device
            )
            
            # Convert results to standardized format
            predictions = []
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()  # Bounding box coordinates (x1, y1, x2, y2)
                    confidences = result.boxes.conf.cpu().numpy()  # Confidence scores
                    classes = result.boxes.cls.cpu().numpy()  # Class labels
                    
                    for box, conf, cls in zip(boxes, confidences, classes):
                        x1, y1, x2, y2 = map(int, box)
                        predictions.append({
                            'bbox': {
                                'x1': x1,
                                'y1': y1,
                                'x2': x2,
                                'y2': y2
                            },
                            'score': float(conf),
                            'category_id': int(cls),
                            'category_name': 'megakaryocyte'
                        })
            
            logger.info(f"Standard YOLO detected {len(predictions)} objects")
            return predictions
        
        except Exception as e:
            logger.error(f"Error during YOLO prediction: {str(e)}")
            raise
    
    def predict(self, 
                image: np.ndarray, 
                use_sahi: bool = True,
                **sahi_kwargs) -> List[Dict[str, Any]]:
        """
        Perform detection with automatic method selection
        
        Args:
            image (np.ndarray): Input image in BGR format
            use_sahi (bool): Whether to use SAHI for detection
            **sahi_kwargs: Additional arguments for SAHI prediction
            
        Returns:
            List[Dict]: List of detection results
        """
        if use_sahi:
            return self.predict_with_sahi(image, **sahi_kwargs)
        else:
            return self.predict_without_sahi(image)
    
    def visualize_predictions(self, 
                             image: np.ndarray, 
                             predictions: List[Dict[str, Any]],
                             color: Tuple[int, int, int] = (0, 255, 0),
                             thickness: int = 2,
                             show_confidence: bool = True) -> np.ndarray:
        """
        Visualize detection results on the image
        
        Args:
            image (np.ndarray): Input image in BGR format
            predictions (List[Dict]): Detection results
            color (Tuple[int, int, int]): BGR color for bounding boxes
            thickness (int): Line thickness for bounding boxes
            show_confidence (bool): Whether to show confidence scores
            
        Returns:
            np.ndarray: Image with visualized detections
        """
        viz_image = image.copy()
        
        for pred in predictions:
            bbox = pred['bbox']
            score = pred['score']
            category_name = pred.get('category_name', 'object')
            
            # Draw bounding box
            cv2.rectangle(
                viz_image, 
                (bbox['x1'], bbox['y1']), 
                (bbox['x2'], bbox['y2']), 
                color, 
                thickness
            )
            
            # Draw label
            if show_confidence:
                label = f"{category_name}: {score:.2f}"
            else:
                label = category_name
                
            # Calculate text size for background
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, thickness
            )
            
            # Draw background rectangle for text
            cv2.rectangle(
                viz_image,
                (bbox['x1'], bbox['y1'] - text_height - 5),
                (bbox['x1'] + text_width, bbox['y1']),
                color,
                -1
            )
            
            # Draw text
            cv2.putText(
                viz_image, 
                label, 
                (bbox['x1'], bbox['y1'] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (255, 255, 255),  # White text
                thickness
            )
            
        return viz_image