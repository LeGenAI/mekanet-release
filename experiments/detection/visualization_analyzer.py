#!/usr/bin/env python3
"""
Visualization and Analysis Tools for TESSD Framework
Comprehensive visualization for detection results, IoU analysis, and performance comparison
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import List, Dict, Any, Optional, Tuple
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DetectionVisualizer:
    """
    Comprehensive visualization and analysis tools for TESSD detection results
    
    Features:
    - Detection result visualization
    - IoU distribution analysis  
    - Performance comparison charts
    - Confidence threshold analysis
    - Spatial distribution analysis
    """
    
    def __init__(self, output_dir: str = "./visualization_results"):
        """Initialize visualization module"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set visualization style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        logger.info(f"Visualization module initialized: {self.output_dir}")
    
    def visualize_detections(self, 
                           image: np.ndarray,
                           detections: List[Dict[str, Any]],
                           ground_truths: List[Dict[str, Any]] = None,
                           image_id: str = "image",
                           save_path: str = None) -> np.ndarray:
        """
        Visualize detection results on image
        
        Args:
            image: Input image
            detections: Detection results
            ground_truths: Ground truth annotations (optional)
            image_id: Image identifier
            save_path: Path to save visualization
            
        Returns:
            Annotated image
        """
        vis_image = image.copy()
        
        # Draw ground truths in green
        if ground_truths:
            for gt in ground_truths:
                bbox = gt['bbox']
                if 'x_center' in gt:  # YOLO format
                    h, w = image.shape[:2]
                    x_center, y_center = gt['x_center'] * w, gt['y_center'] * h
                    width, height = gt['width'] * w, gt['height'] * h
                    x1, y1 = int(x_center - width/2), int(y_center - height/2)
                    x2, y2 = int(x_center + width/2), int(y_center + height/2)
                else:  # Absolute coordinates
                    x1, y1, w, h = bbox
                    x2, y2 = x1 + w, y1 + h
                
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(vis_image, "GT", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw detections with confidence-based colors
        for i, det in enumerate(detections):
            bbox = det['bbox']
            confidence = det['score']
            
            x1, y1, x2, y2 = int(bbox['x1']), int(bbox['y1']), int(bbox['x2']), int(bbox['y2'])
            
            # Color based on confidence
            if confidence >= 0.8:
                color = (255, 0, 0)  # Red for high confidence
            elif confidence >= 0.5:
                color = (0, 255, 255)  # Yellow for medium confidence
            else:
                color = (0, 0, 255)  # Blue for low confidence
            
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis_image, f"{confidence:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Add legend
        legend_y = 30
        cv2.putText(vis_image, "Green=GT, Red=High, Yellow=Med, Blue=Low", (10, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Save if path provided
        if save_path:
            cv2.imwrite(save_path, vis_image)
        
        return vis_image
    
    def analyze_iou_distribution(self, 
                               evaluation_results: List[Dict[str, Any]],
                               save_plots: bool = True) -> Dict[str, Any]:
        """
        Analyze IoU distribution across detections
        
        Args:
            evaluation_results: List of evaluation results with IoU data
            save_plots: Whether to save analysis plots
            
        Returns:
            IoU analysis results
        """
        iou_values = []
        confidence_values = []
        institutions = []
        
        for result in evaluation_results:
            institution = result.get('institution', 'Unknown')
            matches = result.get('matches', [])
            
            for match in matches:
                iou_values.append(match.get('iou', 0))
                confidence_values.append(match.get('confidence', 0))
                institutions.append(institution)
        
        if not iou_values:
            logger.warning("No IoU data found for analysis")
            return {}
        
        # Create analysis plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('IoU Distribution Analysis', fontsize=16)
        
        # IoU histogram
        ax = axes[0, 0]
        ax.hist(iou_values, bins=50, alpha=0.7, density=True)
        ax.axvline(np.mean(iou_values), color='red', linestyle='--', label=f'Mean: {np.mean(iou_values):.3f}')
        ax.set_xlabel('IoU')
        ax.set_ylabel('Density')
        ax.set_title('IoU Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # IoU vs Confidence scatter
        ax = axes[0, 1]
        scatter = ax.scatter(confidence_values, iou_values, alpha=0.6, s=20)
        ax.set_xlabel('Confidence')
        ax.set_ylabel('IoU')
        ax.set_title('IoU vs Confidence')
        ax.grid(True, alpha=0.3)
        
        # IoU by institution
        if len(set(institutions)) > 1:
            ax = axes[1, 0]
            df = pd.DataFrame({'IoU': iou_values, 'Institution': institutions})
            sns.boxplot(data=df, x='Institution', y='IoU', ax=ax)
            ax.set_title('IoU Distribution by Institution')
            ax.grid(True, alpha=0.3)
        
        # IoU threshold analysis
        ax = axes[1, 1]
        thresholds = np.arange(0.1, 1.0, 0.05)
        detection_rates = [np.mean(np.array(iou_values) >= t) for t in thresholds]
        ax.plot(thresholds, detection_rates, 'b-o')
        ax.set_xlabel('IoU Threshold')
        ax.set_ylabel('Detection Rate')
        ax.set_title('Detection Rate vs IoU Threshold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plot_path = self.output_dir / "iou_analysis.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"IoU analysis saved to: {plot_path}")
        
        plt.close()
        
        # Return summary statistics
        return {
            'mean_iou': np.mean(iou_values),
            'std_iou': np.std(iou_values),
            'median_iou': np.median(iou_values),
            'iou_at_0_5': np.mean(np.array(iou_values) >= 0.5),
            'iou_at_0_7': np.mean(np.array(iou_values) >= 0.7),
            'correlation_iou_confidence': np.corrcoef(iou_values, confidence_values)[0, 1]
        }
    
    def create_performance_dashboard(self, 
                                   institutional_results: Dict[str, Any],
                                   save_dashboard: bool = True) -> str:
        """
        Create comprehensive performance dashboard
        
        Args:
            institutional_results: Results from multiple institutions
            save_dashboard: Whether to save dashboard
            
        Returns:
            Path to saved dashboard
        """
        institutions = list(institutional_results.keys())
        
        # Create dashboard with multiple subplots
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # Main title
        fig.suptitle('TESSD Detection Performance Dashboard', fontsize=20, fontweight='bold')
        
        # 1. mAP Comparison (top-left)
        ax1 = fig.add_subplot(gs[0, 0])
        map_50_values = [institutional_results[inst]['map_50'] for inst in institutions]
        ax1.bar(institutions, map_50_values, color='skyblue', alpha=0.8)
        ax1.set_title('mAP@0.5 by Institution', fontweight='bold')
        ax1.set_ylabel('mAP@0.5')
        for i, v in enumerate(map_50_values):
            ax1.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        ax1.grid(True, alpha=0.3)
        
        # 2. Precision-Recall (top-center)
        ax2 = fig.add_subplot(gs[0, 1])
        precision_values = [institutional_results[inst]['precision_at_05'] for inst in institutions]
        recall_values = [institutional_results[inst]['recall_at_05'] for inst in institutions]
        x = np.arange(len(institutions))
        width = 0.35
        ax2.bar(x - width/2, precision_values, width, label='Precision', alpha=0.8)
        ax2.bar(x + width/2, recall_values, width, label='Recall', alpha=0.8)
        ax2.set_title('Precision & Recall @IoU=0.5', fontweight='bold')
        ax2.set_ylabel('Score')
        ax2.set_xticks(x)
        ax2.set_xticklabels(institutions)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Detection Counts (top-right)
        ax3 = fig.add_subplot(gs[0, 2])
        gt_counts = [institutional_results[inst]['total_gt'] for inst in institutions]
        det_counts = [institutional_results[inst]['total_detections'] for inst in institutions]
        x = np.arange(len(institutions))
        ax3.bar(x - width/2, gt_counts, width, label='Ground Truth', alpha=0.8)
        ax3.bar(x + width/2, det_counts, width, label='Detections', alpha=0.8)
        ax3.set_title('Detection Counts', fontweight='bold')
        ax3.set_ylabel('Count')
        ax3.set_xticks(x)
        ax3.set_xticklabels(institutions)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Processing Time (top-far-right)
        ax4 = fig.add_subplot(gs[0, 3])
        proc_times = [institutional_results[inst]['avg_processing_time'] for inst in institutions]
        ax4.bar(institutions, proc_times, color='orange', alpha=0.8)
        ax4.set_title('Processing Time', fontweight='bold')
        ax4.set_ylabel('Time (seconds)')
        for i, v in enumerate(proc_times):
            ax4.text(i, v + 0.01, f'{v:.2f}s', ha='center', va='bottom')
        ax4.grid(True, alpha=0.3)
        
        # 5. F1-Score vs IoU (middle-left span)
        ax5 = fig.add_subplot(gs[1, :2])
        iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        for inst in institutions:
            f1_scores = [institutional_results[inst].get(f'f1_at_{t:.2f}', 0) for t in iou_thresholds]
            ax5.plot(iou_thresholds, f1_scores, 'o-', label=inst, linewidth=2, markersize=6)
        ax5.set_title('F1-Score vs IoU Threshold', fontweight='bold')
        ax5.set_xlabel('IoU Threshold')
        ax5.set_ylabel('F1-Score')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Confidence Distribution (middle-right span)
        ax6 = fig.add_subplot(gs[1, 2:])
        for inst in institutions:
            confidences = institutional_results[inst].get('confidence_values', [])
            if confidences:
                ax6.hist(confidences, bins=30, alpha=0.6, label=inst, density=True)
        ax6.set_title('Confidence Score Distribution', fontweight='bold')
        ax6.set_xlabel('Confidence Score')
        ax6.set_ylabel('Density')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # 7. Size-based Performance (bottom-left span)
        ax7 = fig.add_subplot(gs[2, :2])
        size_categories = ['Small', 'Medium', 'Large']
        for inst in institutions:
            size_aps = [
                institutional_results[inst].get('ap_small', 0),
                institutional_results[inst].get('ap_medium', 0), 
                institutional_results[inst].get('ap_large', 0)
            ]
            ax7.plot(size_categories, size_aps, 'o-', label=inst, linewidth=2, markersize=6)
        ax7.set_title('Average Precision by Object Size', fontweight='bold')
        ax7.set_xlabel('Object Size Category')
        ax7.set_ylabel('Average Precision')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        
        # 8. Detection Efficiency (bottom-right span)
        ax8 = fig.add_subplot(gs[2, 2:])
        for inst in institutions:
            precision = institutional_results[inst]['precision_at_05']
            recall = institutional_results[inst]['recall_at_05']
            processing_time = institutional_results[inst]['avg_processing_time']
            efficiency = (precision * recall) / processing_time if processing_time > 0 else 0
            ax8.bar(inst, efficiency, alpha=0.8)
            ax8.text(institutions.index(inst), efficiency + 0.001, f'{efficiency:.3f}', 
                    ha='center', va='bottom')
        ax8.set_title('Detection Efficiency (F1/Time)', fontweight='bold')
        ax8.set_ylabel('Efficiency Score')
        ax8.grid(True, alpha=0.3)
        
        # 9. Summary Statistics Table (bottom row)
        ax9 = fig.add_subplot(gs[3, :])
        ax9.axis('tight')
        ax9.axis('off')
        
        # Create summary table
        table_data = []
        for inst in institutions:
            result = institutional_results[inst]
            table_data.append([
                inst,
                f"{result['map_50']:.3f}",
                f"{result['precision_at_05']:.3f}",
                f"{result['recall_at_05']:.3f}",
                f"{result['total_detections']}",
                f"{result['avg_processing_time']:.2f}s"
            ])
        
        headers = ['Institution', 'mAP@0.5', 'Precision', 'Recall', 'Detections', 'Proc.Time']
        table = ax9.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax9.set_title('Performance Summary', fontweight='bold', pad=20)
        
        if save_dashboard:
            dashboard_path = self.output_dir / "performance_dashboard.png"
            plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
            logger.info(f"Performance dashboard saved to: {dashboard_path}")
            plt.close()
            return str(dashboard_path)
        
        return ""
    
    def plot_confidence_analysis(self, 
                               confidence_data: Dict[str, Any],
                               institution: str = "Unknown") -> str:
        """
        Create confidence threshold analysis visualization
        
        Args:
            confidence_data: Confidence analysis results
            institution: Institution name
            
        Returns:
            Path to saved plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Confidence Threshold Analysis - {institution}', fontsize=16)
        
        thresholds = confidence_data.get('thresholds', [])
        precisions = confidence_data.get('precisions', [])
        recalls = confidence_data.get('recalls', [])
        
        if not thresholds:
            logger.warning("No confidence data available for plotting")
            plt.close()
            return ""
        
        # Precision-Recall Curve
        ax = axes[0, 0]
        ax.plot(recalls, precisions, 'b-', linewidth=2)
        ax.fill_between(recalls, precisions, alpha=0.2)
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.grid(True, alpha=0.3)
        
        # F1-Score vs Threshold
        ax = axes[0, 1]
        f1_scores = [2*p*r/(p+r) if (p+r) > 0 else 0 for p, r in zip(precisions[:-1], recalls[:-1])]
        ax.plot(thresholds, f1_scores, 'g-', linewidth=2)
        optimal_idx = np.argmax(f1_scores) if f1_scores else 0
        if f1_scores:
            ax.axvline(thresholds[optimal_idx], color='red', linestyle='--', 
                      label=f'Optimal: {thresholds[optimal_idx]:.3f}')
        ax.set_xlabel('Confidence Threshold')
        ax.set_ylabel('F1-Score')
        ax.set_title('F1-Score vs Confidence Threshold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Precision vs Threshold
        ax = axes[1, 0]
        ax.plot(thresholds, precisions[:-1], 'r-', linewidth=2, label='Precision')
        ax.plot(thresholds, recalls[:-1], 'b-', linewidth=2, label='Recall')
        ax.set_xlabel('Confidence Threshold')
        ax.set_ylabel('Score')
        ax.set_title('Precision & Recall vs Threshold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Detection Count vs Threshold
        ax = axes[1, 1]
        detection_counts = [sum(1 for conf in confidence_data.get('all_confidences', []) if conf >= t) 
                          for t in thresholds]
        ax.plot(thresholds, detection_counts, 'purple', linewidth=2)
        ax.set_xlabel('Confidence Threshold')
        ax.set_ylabel('Number of Detections')
        ax.set_title('Detection Count vs Threshold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = self.output_dir / f"confidence_analysis_{institution.lower()}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Confidence analysis saved to: {plot_path}")
        return str(plot_path)
    
    def create_detection_heatmap(self, 
                               detections_data: List[Dict[str, Any]],
                               image_shape: Tuple[int, int],
                               grid_size: int = 20) -> str:
        """
        Create spatial heatmap of detection locations
        
        Args:
            detections_data: List of detection results
            image_shape: Shape of reference image (height, width)
            grid_size: Size of spatial grid
            
        Returns:
            Path to saved heatmap
        """
        height, width = image_shape
        
        # Create spatial grid
        heatmap = np.zeros((grid_size, grid_size))
        
        # Accumulate detections in grid
        for det_data in detections_data:
            detections = det_data.get('detections', [])
            for det in detections:
                bbox = det['bbox']
                center_x = (bbox['x1'] + bbox['x2']) / 2
                center_y = (bbox['y1'] + bbox['y2']) / 2
                
                # Convert to grid coordinates
                grid_x = int((center_x / width) * grid_size)
                grid_y = int((center_y / height) * grid_size)
                
                # Ensure within bounds
                grid_x = max(0, min(grid_size - 1, grid_x))
                grid_y = max(0, min(grid_size - 1, grid_y))
                
                heatmap[grid_y, grid_x] += 1
        
        # Create heatmap visualization
        plt.figure(figsize=(10, 8))
        sns.heatmap(heatmap, annot=False, cmap='YlOrRd', cbar_kws={'label': 'Detection Count'})
        plt.title('Spatial Distribution of Detections')
        plt.xlabel('X Position (normalized)')
        plt.ylabel('Y Position (normalized)')
        
        heatmap_path = self.output_dir / "detection_heatmap.png"
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Detection heatmap saved to: {heatmap_path}")
        return str(heatmap_path)


def main():
    """Main function for testing visualization module"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Detection Visualization and Analysis")
    parser.add_argument("--results-dir", required=True, help="Directory containing evaluation results")
    parser.add_argument("--output-dir", default="./visualization_results", help="Output directory")
    
    args = parser.parse_args()
    
    # Initialize visualizer
    visualizer = DetectionVisualizer(args.output_dir)
    
    # Load results (mock data for demo)
    results_dir = Path(args.results_dir)
    
    # Create sample institutional results for demo
    institutional_results = {
        'B_hospital': {
            'map_50': 0.85, 'precision_at_05': 0.88, 'recall_at_05': 0.82,
            'total_gt': 150, 'total_detections': 145, 'avg_processing_time': 2.3
        },
        'S_hospital': {
            'map_50': 0.78, 'precision_at_05': 0.81, 'recall_at_05': 0.76, 
            'total_gt': 75, 'total_detections': 72, 'avg_processing_time': 2.1
        }
    }
    
    # Create performance dashboard
    dashboard_path = visualizer.create_performance_dashboard(institutional_results)
    print(f"✅ Performance dashboard created: {dashboard_path}")
    
    print(f"📁 All visualizations saved to: {args.output_dir}")


if __name__ == "__main__":
    main() 