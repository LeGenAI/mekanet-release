"""
Binary Classification Experiment for MPN Detection

This script demonstrates binary classification to distinguish between
MPN patients and controls using clinical and morphological features.
"""

import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from mekanet.models import MPNClassifier
from mekanet.utils import calculate_metrics, plot_confusion_matrix


def load_and_prepare_data(data_path):
    """Load and prepare data for binary classification"""
    df = pd.read_csv(data_path)
    
    # Create binary labels: 0 for Lymphoma (Control), 1 for MPN (ET, PV, PMF)
    df['Binary_Label'] = df['Label'].apply(lambda x: 0 if x == 'Lymphoma' else 1)
    
    # Select features for classification
    clinical_features = ['Age', 'Hb', 'WBC', 'PLT', 'JAK2', 'CALR', 'MPL']
    morphological_features = ['Avg_Size', 'Std_Size', 'Num_Megakaryocytes', 
                             'Avg_NND', 'Avg_Local_Density', 'Num_Clusters']
    
    # Combine features
    feature_columns = clinical_features + morphological_features
    
    # Handle missing values
    for col in feature_columns:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    
    X = df[feature_columns].values
    y = df['Binary_Label'].values
    
    return X, y, feature_columns, df


def run_binary_classification_experiment(data_path, model_type='logistic_regression'):
    """Run binary classification experiment"""
    print("🔬 Binary Classification Experiment: MPN vs Control")
    print("=" * 60)
    
    # Load data
    X, y, feature_names, df = load_and_prepare_data(data_path)
    
    print(f"📊 Dataset Info:")
    print(f"   - Total samples: {len(X)}")
    print(f"   - Features: {len(feature_names)}")
    print(f"   - Controls: {sum(y == 0)}")
    print(f"   - MPN cases: {sum(y == 1)}")
    print()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Test different models
    models = {
        'Logistic Regression': 'logistic_regression',
        'Random Forest': 'random_forest',
        'Decision Tree': 'decision_tree',
        'XGBoost': 'xgboost'
    }
    
    results = {}
    
    for model_name, model_type in models.items():
        print(f"🤖 Training {model_name}...")
        
        # Initialize and train classifier
        classifier = MPNClassifier(model_type=model_type, binary_mode=True)
        training_results = classifier.train(
            X_train_scaled, y_train, 
            feature_names=feature_names,
            test_size=0.0,  # We already split
            use_grid_search=True
        )
        
        # Make predictions on test set
        predictions = classifier.predict(X_test_scaled)
        y_pred = predictions['predictions']
        y_prob = np.array(predictions['probabilities'])[:, 1]  # Probability of MPN
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        
        results[model_name] = {
            'accuracy': accuracy,
            'auc': auc,
            'y_pred': y_pred,
            'y_prob': y_prob,
            'classifier': classifier
        }
        
        print(f"   ✅ Accuracy: {accuracy:.3f}")
        print(f"   📈 AUC: {auc:.3f}")
        print()
    
    # Display results summary
    print("📋 Results Summary:")
    print("-" * 40)
    for model_name, result in results.items():
        print(f"{model_name:20s}: Accuracy={result['accuracy']:.3f}, AUC={result['auc']:.3f}")
    
    # Get best model
    best_model_name = max(results.keys(), key=lambda k: results[k]['accuracy'])
    best_classifier = results[best_model_name]['classifier']
    
    print(f"\n🏆 Best Model: {best_model_name}")
    
    # Feature importance
    feature_importance = best_classifier.get_feature_importance()
    if feature_importance:
        print("\n🔍 Feature Importance (Top 10):")
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        for i, (feature, importance) in enumerate(sorted_features[:10]):
            print(f"   {i+1:2d}. {feature:20s}: {importance:.3f}")
    
    # Detailed classification report
    y_pred_best = results[best_model_name]['y_pred']
    print(f"\n📊 Detailed Classification Report ({best_model_name}):")
    print(classification_report(y_test, y_pred_best, 
                              target_names=['Control', 'MPN']))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_best)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Control', 'MPN'],
                yticklabels=['Control', 'MPN'])
    plt.title(f'Confusion Matrix - {best_model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('binary_classification_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # ROC Curves
    plt.figure(figsize=(10, 8))
    for model_name, result in results.items():
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_test, result['y_prob'])
        plt.plot(fpr, tpr, label=f"{model_name} (AUC={result['auc']:.3f})")
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - Binary Classification')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('binary_classification_roc_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return results, best_classifier


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Binary Classification for MPN Detection")
    parser.add_argument('--data', type=str, required=True,
                       help='Path to the CSV data file')
    parser.add_argument('--model', type=str, default='logistic_regression',
                       choices=['logistic_regression', 'random_forest', 'decision_tree', 'xgboost'],
                       help='Model type to use')
    
    args = parser.parse_args()
    
    # Run experiment
    results, best_classifier = run_binary_classification_experiment(
        args.data, args.model
    )
    
    print("✅ Binary classification experiment completed!")
    print("📁 Results saved as PNG files in current directory.")