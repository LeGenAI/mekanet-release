#!/usr/bin/env python3
"""
TESSD Inference Demo
Simple demo script for single image megakaryocyte detection
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
import sys
import time

# Add paths for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from experiments.detection.tessd_framework import TESSDFramework

def main():
    parser = argparse.ArgumentParser(description='TESSD Detection Demo')
    parser.add_argument('--image_path', type=str, required=True,
                      help='Path to input image')
    parser.add_argument('--model_path', type=str, 
                      default='../../weights/epoch60.pt',
                      help='Path to YOLO model weights')
    parser.add_argument('--confidence', type=float, default=0.20,
                      help='Confidence threshold')
    parser.add_argument('--use_sahi', action='store_true', default=True,
                      help='Use SAHI tiling')
    parser.add_argument('--tile_size', type=int, default=640,
                      help='Tile size for SAHI')
    parser.add_argument('--overlap_ratio', type=float, default=0.2,
                      help='Overlap ratio for SAHI')
    parser.add_argument('--output_dir', type=str, default='./demo_output',
                      help='Output directory for results')
    parser.add_argument('--device', type=str, default='cpu',
                      help='Device for inference (cpu/cuda)')
    
    args = parser.parse_args()
    
    # Validate inputs
    image_path = Path(args.image_path)
    if not image_path.exists():
        print(f"Error: Image file not found: {image_path}")
        return 1
    
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        print("Please download the model weights using:")
        print("cd ../../weights && python download_weights.py")
        return 1
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("TESSD Detection Demo")
    print("=" * 50)
    print(f"Image: {image_path}")
    print(f"Model: {model_path}")
    print(f"Confidence: {args.confidence}")
    print(f"SAHI: {'Enabled' if args.use_sahi else 'Disabled'}")
    print(f"Device: {args.device}")
    print()
    
    try:
        # Initialize TESSD framework
        print("Loading TESSD framework...")
        tessd = TESSDFramework(
            model_path=str(model_path),
            confidence_threshold=args.confidence,
            device=args.device,
            tile_size=args.tile_size,
            overlap_ratio=args.overlap_ratio,
            experiment_name="demo"
        )
        
        # Load image
        print("Loading image...")
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Error: Failed to load image: {image_path}")
            return 1
        
        print(f"Image shape: {image.shape}")
        
        # Perform detection
        print("Performing detection...")
        start_time = time.time()
        
        image_id = image_path.stem
        institution = "S_hospital" if image_id.startswith('S') else "B_hospital"
        
        result = tessd.detect_with_metadata(
            image,
            image_id=image_id,
            institution=institution,
            use_sahi=args.use_sahi
        )
        
        detection_time = time.time() - start_time
        
        # Print results
        print("\nDetection Results:")
        print("-" * 30)
        print(f"Number of detections: {result['num_detections']}")
        print(f"Detection time: {detection_time:.2f}s")
        print(f"Institution: {result['institution']}")
        
        if result['detections']:
            confidences = [d['score'] for d in result['detections']]
            print(f"Confidence range: {min(confidences):.3f} - {max(confidences):.3f}")
            print(f"Mean confidence: {np.mean(confidences):.3f}")
        
        # Print morphological features
        print("\nMorphological Features:")
        print("-" * 30)
        features = result['morphological_features']
        for feature_name, value in features.items():
            print(f"{feature_name}: {value:.3f}")
        
        # Save visualization
        print("\nSaving results...")
        viz_image = tessd.visualize_predictions(
            image, 
            result['detections'],
            show_confidence=True
        )
        
        # Save original and visualization
        output_image = output_dir / f"{image_id}_detection.jpg"
        original_image = output_dir / f"{image_id}_original.jpg"
        
        cv2.imwrite(str(output_image), viz_image)
        cv2.imwrite(str(original_image), image)
        
        # Save detailed results
        results_file = output_dir / f"{image_id}_results.txt"
        with open(results_file, 'w') as f:
            f.write("TESSD Detection Results\n")
            f.write("=" * 30 + "\n")
            f.write(f"Image: {image_path}\n")
            f.write(f"Model: {model_path}\n")
            f.write(f"Confidence threshold: {args.confidence}\n")
            f.write(f"SAHI enabled: {args.use_sahi}\n")
            f.write(f"Detection time: {detection_time:.2f}s\n")
            f.write(f"Number of detections: {result['num_detections']}\n\n")
            
            f.write("Morphological Features:\n")
            f.write("-" * 20 + "\n")
            for feature_name, value in features.items():
                f.write(f"{feature_name}: {value:.6f}\n")
            
            f.write("\nDetailed Detections:\n")
            f.write("-" * 20 + "\n")
            for i, detection in enumerate(result['detections']):
                f.write(f"Detection {i+1}:\n")
                f.write(f"  Confidence: {detection['score']:.3f}\n")
                f.write(f"  Bbox: {detection['bbox']}\n")
        
        print(f"Results saved to: {output_dir}")
        print(f"Visualization: {output_image}")
        print(f"Detailed results: {results_file}")
        
        # Optional: Display image (if display is available)
        try:
            import matplotlib.pyplot as plt
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
            
            # Original image
            ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            ax1.set_title('Original Image')
            ax1.axis('off')
            
            # Detection results
            ax2.imshow(cv2.cvtColor(viz_image, cv2.COLOR_BGR2RGB))
            ax2.set_title(f'Detections: {result["num_detections"]} megakaryocytes')
            ax2.axis('off')
            
            plt.tight_layout()
            plt.savefig(output_dir / f"{image_id}_comparison.png", dpi=300, bbox_inches='tight')
            plt.show()
            
        except ImportError:
            print("Matplotlib not available for display")
        except Exception as e:
            print(f"Display error: {e}")
        
        print("\nDemo completed successfully!")
        return 0
        
    except Exception as e:
        print(f"Error during detection: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main()) 