<div align="center">

# 🚀 MekaNet Quick Start Guide

<img src="https://readme-typing-svg.herokuapp.com?font=Orbitron&size=25&duration=3000&pause=1000&color=4ECDC4&center=true&vCenter=true&width=600&height=50&lines=Get+Started+in+5+Minutes;From+Zero+to+MPN+Classification;AI-Powered+Pathology" alt="Quick Start Typing SVG" />

</div>

---

## 🎯 **What You'll Achieve**

<div align="center">

| ⏱️ **Time** | 🎯 **Goal** | 📊 **Result** |
|:---:|:---|:---:|
| **2 minutes** | Installation & Setup | ✅ Ready Environment |
| **1 minute** | Download Models | 🤖 Pre-trained Weights |
| **2 minutes** | Run Demo | 🏥 MPN Classification |

</div>

---

## 🏁 **Prerequisites**

### 🖥️ **System Check**

```bash
# ✅ Check Python version (3.8+ required)
python --version

# ✅ Check available space (5GB+ required)
df -h

# ✅ Check GPU availability (optional)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## 📦 **Step 1: Installation (2 minutes)**

### 🚀 **Quick Install**

```bash
# 🔄 Clone the repository
git clone https://github.com/LeGenAI/mekanet-release.git
cd mekanet-release

# 🐍 Create virtual environment (recommended)
python -m venv mekanet_env

# 🔧 Activate environment
# On Windows:
mekanet_env\Scripts\activate
# On macOS/Linux:
source mekanet_env/bin/activate

# 📦 Install all dependencies
pip install -r requirements.txt

# 🎯 Install MekaNet package
pip install -e .
```

### 🔧 **Alternative: Manual Install**

<details>
<summary>Click for manual dependency installation</summary>

```bash
# Core ML libraries
pip install torch>=1.9.0 torchvision>=0.10.0
pip install ultralytics>=8.0.0 sahi>=0.11.0

# Data processing
pip install pandas>=1.3.0 numpy>=1.21.0
pip install scikit-learn>=1.0.0 xgboost>=1.5.0

# Computer vision
pip install opencv-python>=4.5.0 Pillow>=8.3.0

# Visualization
pip install matplotlib>=3.4.0 seaborn>=0.11.0

# Scientific computing
pip install scipy>=1.7.0 tqdm>=4.62.0
```

</details>

---

## 🔽 **Step 2: Download Models (1 minute)**

```bash
# 📁 Navigate to weights directory
cd weights

# 🤖 Download pre-trained models
python download_weights.py

# ✅ Verify models are downloaded
ls -la *.pt *.pkl
```

**Expected output:**
```
epoch60.pt      # YOLOv8 detection model (~14MB)
classifier.pkl  # Trained classification model (~2MB)
```

---

## 🧪 **Step 3: Run Your First Demo (2 minutes)**

### 🩺 **Binary Classification Demo**

```bash
# 📁 Navigate to experiments
cd ../experiments/classification

# 🚀 Run binary classification
python binary_classification.py --data ../../data/demo_data/classification_demo.csv
```

**Expected Output:**
```
🔬 Binary Classification Experiment: MPN vs Control
================================================================
📊 Dataset Info:
   - Total samples: 20
   - Features: 13 (clinical + morphological)
   - Controls: 10 (Lymphoma cases)
   - MPN cases: 10 (ET, PV, PMF)

🤖 Training Models:
   ✅ Logistic Regression: 100.0% accuracy
   ✅ Random Forest: 85.0% accuracy
   ✅ Decision Tree: 93.0% accuracy
   ✅ XGBoost: 93.0% accuracy

🏆 Best Model: Logistic Regression (100% accuracy)
📊 Perfect distinction between MPN patients and controls
```

### 🔬 **MPN Subtype Classification Demo**

```bash
# 🎯 Run multi-class classification
python mpn_classification.py --data ../../data/demo_data/classification_demo.csv
```

**Expected Output:**
```
🔬 Multi-class MPN Classification Experiment
================================================================
📊 MPN Subtype Distribution:
   ET (Essential Thrombocythemia): 6 cases (60.0%)
   PV (Polycythemia Vera): 2 cases (20.0%)
   PMF (Primary Myelofibrosis): 2 cases (20.0%)

🤖 Model Performance:
   ✅ Decision Tree: High accuracy with interpretable rules
   ✅ Random Forest: Robust ensemble performance
   ✅ XGBoost: Advanced gradient boosting results

🏆 Best Model: Decision Tree
📊 Average recall: 0.90 across all MPN subtypes
```

---

## 🎯 **Quick API Usage**

### 🔍 **Basic Detection & Classification**

```python
from mekanet import YoloSahiDetector, FeatureExtractor, MPNClassifier
import cv2

# 🏥 Load your bone marrow image
image = cv2.imread('your_BM_image.jpg')

# 🤖 Initialize detector
detector = YoloSahiDetector('weights/epoch60.pt')

# 🔍 Detect megakaryocytes
detections = detector.predict(image, use_sahi=True)
print(f"Found {len(detections)} megakaryocytes")

# 📊 Extract features
extractor = FeatureExtractor()
features = extractor.extract_features(detections)

# 🎯 Classify MPN subtype
classifier = MPNClassifier.load('weights/classifier.pkl')
result = classifier.predict_single(list(features.values()))

print(f"🏥 Diagnosis: {result['predicted_label']}")
print(f"📊 Confidence: {result['probability']:.1%}")
```

### 🧬 **With Clinical Data**

```python
# 🩸 Patient clinical data
clinical_data = {
    'Age': 65,
    'Hb': 18.7,     # Hemoglobin
    'PLT': 764,     # Platelet count
    'JAK2': 1,      # JAK2 positive
    'CALR': 0,      # CALR negative
    'MPL': 0        # MPL negative
}

# 🔗 Combine with morphological features
all_features = list(clinical_data.values()) + list(features.values())

# 🎯 Enhanced classification
result = classifier.predict_single(all_features)
print(f"🏥 Enhanced Diagnosis: {result['predicted_label']}")
```

---

## 📓 **Interactive Demos**

### 🌐 **Jupyter Notebook Demo**

```bash
# 📁 Navigate to notebooks
cd ../notebooks

# 🚀 Launch Jupyter
jupyter notebook demo_classification.ipynb
```

**Notebook Features:**
- 📊 Interactive data exploration
- 🔍 Feature visualization
- 🎯 Model comparison
- 📈 Performance analysis

---

## 🛠️ **Customization Options**

### ⚙️ **Detection Parameters**

```python
# 🎯 Customize detection settings
detector = YoloSahiDetector(
    model_path='weights/epoch60.pt',
    confidence_threshold=0.15,  # Lower = more detections
    device='cuda'               # Use GPU if available
)

# 🔬 SAHI parameters for large images
detections = detector.predict_with_sahi(
    image,
    slice_height=640,           # Tile size
    slice_width=640,
    overlap_height_ratio=0.2,   # Overlap between tiles
    overlap_width_ratio=0.2
)
```

### 🧮 **Feature Extraction Options**

```python
# 📊 Advanced feature extraction
features = extractor.extract_features(
    detections, 
    image_shape=image.shape[:2],  # For density calculations
)

# 🔍 Access specific feature groups
print("Size features:", features['Avg_Size'], features['Std_Size'])
print("Spatial features:", features['Avg_NND'], features['Num_Clusters'])
```

---

## 🚨 **Troubleshooting**

### ❌ **Common Issues & Solutions**

<details>
<summary>🐛 Import errors</summary>

```bash
# 💡 Solution: Reinstall dependencies
pip uninstall -y torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

</details>

<details>
<summary>🔽 Model download fails</summary>

```bash
# 💡 Solution: Manual download
cd weights
# Check download_weights.py for placeholder files
# In production, models would be downloaded from releases
```

</details>

<details>
<summary>💾 Out of memory</summary>

```python
# 💡 Solution: Reduce image size or use CPU
detector = YoloSahiDetector(
    model_path='weights/epoch60.pt',
    device='cpu'  # Use CPU instead of GPU
)

# Or reduce tile size
detections = detector.predict_with_sahi(
    image,
    slice_height=320,  # Smaller tiles
    slice_width=320
)
```

</details>

<details>
<summary>🖼️ Image format issues</summary>

```python
# 💡 Solution: Ensure correct image format
import cv2
image = cv2.imread('image.jpg')
if image is None:
    print("❌ Could not load image")
else:
    print(f"✅ Image loaded: {image.shape}")
```

</details>

---

## 🎯 **Next Steps**

<div align="center">

| 🚀 **Action** | 📝 **Description** | 🔗 **Link** |
|:---:|:---|:---:|
| **📖 Documentation** | Read full documentation | [README.md](../README.md) |
| **🧪 Advanced Experiments** | Try more complex demos | [experiments/](../experiments/) |
| **🏥 Clinical Integration** | Learn about clinical use | [Clinical Apps](../README.md#-clinical-applications) |
| **🤝 Contributing** | Join the community | [Contributing](../README.md#-contributing) |

</div>

---

## 📞 **Get Help**

<div align="center">

### 🆘 **Need Assistance?**

[![GitHub Issues](https://img.shields.io/badge/GitHub-Issues-red?style=for-the-badge&logo=github)](https://github.com/LeGenAI/mekanet-release/issues)
[![Email Support](https://img.shields.io/badge/Email-Support-blue?style=for-the-badge&logo=gmail)](mailto:jhbaek@sogang.ac.kr)
[![Documentation](https://img.shields.io/badge/Full-Documentation-green?style=for-the-badge&logo=gitbook)](../README.md)

</div>

### 🔍 **Before Asking for Help**

1. ✅ Check the [troubleshooting section](#-troubleshooting)
2. 📖 Read the [full documentation](../README.md)
3. 🔍 Search [existing issues](https://github.com/LeGenAI/mekanet-release/issues)
4. 📋 Provide system info and error messages

---

<div align="center">

**🎉 Congratulations! You're now ready to use MekaNet for MPN classification! 🎉**

<img src="https://readme-typing-svg.herokuapp.com?font=Orbitron&size=18&duration=2000&pause=1000&color=4ECDC4&center=true&vCenter=true&width=500&height=40&lines=Ready+to+Transform+Pathology!;Start+Your+AI+Journey+Now!" alt="Success Message" />

</div>