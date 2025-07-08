# Model Weights

This directory contains the pre-trained model weights for MekaNet.

## Available Models

### Detection Model
- **File**: `epoch60.pt`
- **Type**: YOLOv8 detection model
- **Purpose**: Megakaryocyte detection in bone marrow images
- **Size**: ~14MB
- **Training**: Trained on 100 partially labeled images from B hospital

### Classification Model  
- **File**: `classifier.pkl`
- **Type**: Decision Tree classifier
- **Purpose**: MPN subtype classification (ET, PV, PMF)
- **Size**: ~2MB
- **Features**: Clinical + morphological features

## Download Instructions

### Automatic Download
```bash
cd weights
python download_weights.py
```

### Manual Download
If automatic download fails, you can manually download the weights:

1. **Detection Model**: Download `epoch60.pt` from the releases page
2. **Classification Model**: Download `classifier.pkl` from the releases page

### Verification
To verify all weights are present:
```bash
python download_weights.py --verify
```

## Usage

### Detection Model
```python
from mekanet import YoloSahiDetector

# Load detector
detector = YoloSahiDetector('weights/epoch60.pt')

# Use for detection
detections = detector.predict(image, use_sahi=True)
```

### Classification Model
```python
from mekanet.models import MPNClassifier

# Load classifier
classifier = MPNClassifier.load('weights/classifier.pkl')

# Make predictions
result = classifier.predict_single(features)
```

## Model Performance

### Detection Model
- **Architecture**: YOLOv8 with SAHI
- **Training Data**: 100 partially labeled MPN images  
- **Validation**: External validation on 5 images from S hospital
- **Performance**: High precision detection across institutional sources

### Classification Model
- **Algorithm**: Decision Tree with grid search optimization
- **Features**: 13 clinical + morphological features
- **Performance**: 
  - Binary classification (MPN vs Control): Up to 100% accuracy
  - Multi-class classification (ET/PV/PMF): Up to 88% accuracy

## File Integrity

After downloading, verify file integrity:

```bash
# Check file sizes
ls -lh *.pt *.pkl

# Verify loading (optional)
python -c "
from mekanet import YoloSahiDetector
from mekanet.models import MPNClassifier
detector = YoloSahiDetector('epoch60.pt')
classifier = MPNClassifier.load('classifier.pkl')
print('✅ All models loaded successfully!')
"
```

## Troubleshooting

### Download Issues
- Check internet connection
- Verify GitHub releases are accessible
- Try manual download if automatic fails

### Loading Issues
- Ensure correct PyTorch version (≥1.9.0)
- Check CUDA compatibility for GPU usage
- Verify file integrity (re-download if needed)

## License

These model weights are provided under the same license as the MekaNet framework.
For research and educational use only.