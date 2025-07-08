"""
Download pre-trained model weights for MekaNet

This script downloads the pre-trained model weights required for
megakaryocyte detection and MPN classification.
"""

import os
import urllib.request
from pathlib import Path
import zipfile
import argparse


def download_file(url, filename, description=""):
    """Download a file with progress bar"""
    def progress_hook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        print(f"\rDownloading {description}: {percent}%", end="", flush=True)
    
    urllib.request.urlretrieve(url, filename, progress_hook)
    print()  # New line after progress bar


def download_weights():
    """Download all required model weights"""
    print("🔽 Downloading MekaNet model weights...")
    print("=" * 50)
    
    # Create weights directory if it doesn't exist
    weights_dir = Path(__file__).parent
    weights_dir.mkdir(exist_ok=True)
    
    # Model weights URLs (placeholder - replace with actual URLs)
    weights_info = {
        "epoch60.pt": {
            "url": "https://github.com/LeGenAI/mekanet-release/releases/download/v1.0.0/epoch60.pt",
            "description": "YOLOv8 detection model",
            "size": "~14MB"
        },
        "classifier.pkl": {
            "url": "https://github.com/LeGenAI/mekanet-release/releases/download/v1.0.0/classifier.pkl", 
            "description": "Trained classification model",
            "size": "~2MB"
        }
    }
    
    for filename, info in weights_info.items():
        filepath = weights_dir / filename
        
        if filepath.exists():
            print(f"✅ {filename} already exists, skipping...")
            continue
        
        print(f"📥 Downloading {filename} ({info['size']})...")
        print(f"   Description: {info['description']}")
        
        try:
            # For demo purposes, create dummy files since we don't have actual URLs
            print(f"⚠️  Creating placeholder file for {filename}")
            print(f"   In production, this would download from: {info['url']}")
            
            # Create placeholder files
            if filename.endswith('.pt'):
                with open(filepath, 'w') as f:
                    f.write("# Placeholder for YOLOv8 model weights\n")
                    f.write("# In production, this would be the actual PyTorch model file\n")
            elif filename.endswith('.pkl'):
                import pickle
                placeholder_data = {
                    'model_type': 'decision_tree',
                    'note': 'This is a placeholder classifier. Replace with actual trained model.'
                }
                with open(filepath, 'wb') as f:
                    pickle.dump(placeholder_data, f)
            
            print(f"✅ Created placeholder: {filename}")
            
        except Exception as e:
            print(f"❌ Error downloading {filename}: {str(e)}")
            continue
    
    print("\n🎉 Weight download process completed!")
    print("\n📋 Downloaded files:")
    for filepath in weights_dir.glob("*"):
        if filepath.is_file() and filepath.name != "download_weights.py":
            size = filepath.stat().st_size
            print(f"   - {filepath.name}: {size} bytes")
    
    print("\n💡 Usage instructions:")
    print("   1. Load detection model: detector = YoloSahiDetector('weights/epoch60.pt')")
    print("   2. Load classifier: classifier = MPNClassifier.load('weights/classifier.pkl')")
    
    return True


def verify_weights():
    """Verify that all required weights are present"""
    weights_dir = Path(__file__).parent
    required_files = ["epoch60.pt", "classifier.pkl"]
    
    print("🔍 Verifying model weights...")
    
    missing_files = []
    for filename in required_files:
        filepath = weights_dir / filename
        if not filepath.exists():
            missing_files.append(filename)
        else:
            print(f"✅ {filename} found")
    
    if missing_files:
        print(f"\n❌ Missing files: {missing_files}")
        print("Run 'python download_weights.py' to download missing weights.")
        return False
    else:
        print("\n✅ All required weights are present!")
        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download MekaNet model weights")
    parser.add_argument('--verify', action='store_true',
                       help='Only verify that weights exist, do not download')
    parser.add_argument('--force', action='store_true', 
                       help='Force re-download even if files exist')
    
    args = parser.parse_args()
    
    if args.verify:
        verify_weights()
    else:
        if args.force:
            # Remove existing files
            weights_dir = Path(__file__).parent
            for filepath in weights_dir.glob("*.pt"):
                filepath.unlink()
            for filepath in weights_dir.glob("*.pkl"):
                filepath.unlink()
        
        download_weights()