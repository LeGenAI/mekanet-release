"""
MekaNet Enhanced RFECV Feature Selection Analyzer
Advanced RFECV analysis with feature stability and correlation analysis
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.feature_selection import RFECV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class RFECVFeatureSelector:
    """
    Enhanced RFECV analyzer addressing feature instability issues
    - Detection feature correlation analysis
    - PCA-based feature grouping
    - Stability metrics monitoring
    """
    
    def __init__(self, random_seeds=None, cv_folds=5):
        self.random_seeds = random_seeds or [42, 123, 456, 789, 1011]
        self.cv_folds = cv_folds
        self.results = {}
        self.feature_correlation_analysis = {}
        self.pca_transformations = {}
        
    def analyze_feature_correlations(self, df, threshold=0.7, save_dir=None):
        """
        Analyze correlations between detection features
        
        Args:
            df: Complete dataset
            threshold: Correlation threshold for high correlation pairs
            save_dir: Directory to save visualizations
        """
        # Extract detection features
        detection_features = [col for col in df.columns if col not in 
                            ['Label', 'Image_ID', 'sex', 'age', 'Hb', 'WBC', 'PLT', 'Reti%', 
                             'Driver_gene', 'VAF', 'data_source', 'Hospital', 'Binary_Label']]
        
        if len(detection_features) == 0:
            print("Warning: No detection features found for correlation analysis")
            return {}
        
        # Calculate correlation matrix
        detection_df = df[detection_features].select_dtypes(include=[np.number])
        correlation_matrix = detection_df.corr()
        
        # Find high correlation pairs
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > threshold:
                    high_corr_pairs.append({
                        'feature1': correlation_matrix.columns[i],
                        'feature2': correlation_matrix.columns[j],
                        'correlation': corr_value
                    })
        
        # Group correlated features using PCA
        feature_groups = self._create_feature_groups(correlation_matrix, threshold)
        
        # Save correlation analysis results
        self.feature_correlation_analysis = {
            'correlation_matrix': correlation_matrix,
            'high_correlation_pairs': high_corr_pairs,
            'feature_groups': feature_groups,
            'threshold': threshold
        }
        
        # Create visualizations if save directory provided
        if save_dir:
            self._save_correlation_visualizations(save_dir)
        
        return self.feature_correlation_analysis
    
    def _create_feature_groups(self, correlation_matrix, threshold):
        """Create feature groups based on correlation patterns"""
        features = correlation_matrix.columns.tolist()
        feature_groups = []
        used_features = set()
        
        for i, feature1 in enumerate(features):
            if feature1 in used_features:
                continue
                
            group = [feature1]
            used_features.add(feature1)
            
            for j, feature2 in enumerate(features):
                if j != i and feature2 not in used_features:
                    if abs(correlation_matrix.iloc[i, j]) > threshold:
                        group.append(feature2)
                        used_features.add(feature2)
            
            if len(group) > 1:
                feature_groups.append(group)
        
        return feature_groups
    
    def run_enhanced_rfecv(self, X, y, feature_set_name, task_type='binary'):
        """
        Run enhanced RFECV with stability analysis
        
        Args:
            X: Feature matrix
            y: Target labels
            feature_set_name: Name of feature set being analyzed
            task_type: 'binary' or 'multiclass'
        """
        print(f"Running Enhanced RFECV for {feature_set_name} ({task_type})")
        
        # Initialize storage for results
        optimal_counts = []
        accuracies = []
        selected_features_list = []
        
        # Run RFECV multiple times with different random seeds
        for seed in self.random_seeds:
            print(f"  Processing seed {seed}...")
            
            # Initialize classifier
            if task_type == 'binary':
                estimator = DecisionTreeClassifier(random_state=seed, max_depth=10)
            else:
                estimator = RandomForestClassifier(n_estimators=50, random_state=seed, max_depth=10)
            
            # Setup cross-validation
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=seed)
            
            # Run RFECV
            rfecv = RFECV(
                estimator=estimator,
                step=1,
                cv=cv,
                scoring='accuracy',
                min_features_to_select=1,
                n_jobs=-1
            )
            
            try:
                rfecv.fit(X, y)
                
                optimal_counts.append(rfecv.n_features_)
                # Handle different sklearn versions
                if hasattr(rfecv, 'grid_scores_'):
                    accuracies.append(rfecv.grid_scores_.max())
                elif hasattr(rfecv, 'cv_results_'):
                    accuracies.append(max(rfecv.cv_results_['mean_test_score']))
                else:
                    # Use cross-validation score as fallback
                    cv_score = rfecv.score(X, y)
                    accuracies.append(cv_score)
                
                # Get selected feature names
                selected_features = X.columns[rfecv.support_].tolist()
                selected_features_list.append(selected_features)
                
            except Exception as e:
                print(f"    Warning: RFECV failed for seed {seed}: {str(e)}")
                continue
        
        if not optimal_counts:
            print(f"  Error: All RFECV runs failed for {feature_set_name}")
            return None
        
        # Calculate stability metrics
        stability_results = self._calculate_stability_metrics(
            optimal_counts, accuracies, selected_features_list
        )
        
        # Store results
        self.results[feature_set_name] = {
            'optimal_counts': optimal_counts,
            'accuracies': accuracies,
            'selected_features_list': selected_features_list,
            'stability_metrics': stability_results,
            'task_type': task_type
        }
        
        return stability_results
    
    def _calculate_stability_metrics(self, optimal_counts, accuracies, selected_features_list):
        """Calculate stability metrics for RFECV results"""
        # Basic statistics
        mean_count = np.mean(optimal_counts)
        std_count = np.std(optimal_counts)
        mean_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)
        
        # Stability score (lower coefficient of variation = higher stability)
        stability_score = 1 - (std_count / mean_count) if mean_count > 0 else 0
        
        # Most frequent optimal count
        from collections import Counter
        count_frequency = Counter(optimal_counts)
        most_frequent_count = count_frequency.most_common(1)[0][0]
        
        # Feature consistency analysis
        all_features = set()
        for features in selected_features_list:
            all_features.update(features)
        
        feature_frequency = {}
        for feature in all_features:
            frequency = sum(1 for features in selected_features_list if feature in features)
            feature_frequency[feature] = frequency / len(selected_features_list)
        
        # Top consistent features
        top_features = sorted(feature_frequency.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'mean_count': mean_count,
            'std_count': std_count,
            'stability_score': stability_score,
            'most_frequent_count': most_frequent_count,
            'mean_accuracy': mean_accuracy,
            'std_accuracy': std_accuracy,
            'feature_frequency': feature_frequency,
            'top_features': top_features
        }
    
    def _save_correlation_visualizations(self, save_dir):
        """Save correlation analysis visualizations"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Correlation heatmap
        plt.figure(figsize=(12, 10))
        correlation_matrix = self.feature_correlation_analysis['correlation_matrix']
        
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', 
                   center=0, mask=mask, square=True)
        plt.title('Detection Features Correlation Matrix')
        plt.tight_layout()
        plt.savefig(save_dir / 'detection_features_correlation.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # High correlation pairs visualization
        high_corr_pairs = self.feature_correlation_analysis['high_correlation_pairs']
        if high_corr_pairs:
            corr_df = pd.DataFrame(high_corr_pairs)
            
            plt.figure(figsize=(10, 6))
            plt.barh(range(len(corr_df)), corr_df['correlation'].abs())
            plt.yticks(range(len(corr_df)), 
                      [f"{row['feature1']} ↔ {row['feature2']}" for _, row in corr_df.iterrows()])
            plt.xlabel('Absolute Correlation')
            plt.title(f'High Correlation Pairs (|r| > {self.feature_correlation_analysis["threshold"]})')
            plt.tight_layout()
            plt.savefig(save_dir / 'high_correlation_pairs.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def generate_report(self, save_path=None):
        """Generate comprehensive analysis report"""
        report = []
        report.append("MEKANET ENHANCED RFECV ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Correlation analysis summary
        if self.feature_correlation_analysis:
            high_corr_pairs = self.feature_correlation_analysis['high_correlation_pairs']
            feature_groups = self.feature_correlation_analysis['feature_groups']
            
            report.append("CORRELATION ANALYSIS:")
            report.append("=" * 40)
            report.append(f"High correlation pairs identified: {len(high_corr_pairs)}")
            report.append(f"Feature groups formed: {len(feature_groups)}")
            report.append("")
        
        # RFECV results for each feature set
        for feature_set, results in self.results.items():
            stability = results['stability_metrics']
            
            report.append(f"{feature_set.upper()} RESULTS:")
            report.append("-" * 50)
            report.append(f"Optimal feature count: {stability['most_frequent_count']} (most frequent)")
            report.append(f"Mean optimal count: {stability['mean_count']:.1f} ± {stability['std_count']:.1f}")
            report.append(f"Stability score: {stability['stability_score']:.3f}")
            report.append(f"Mean accuracy: {stability['mean_accuracy']:.3f} ± {stability['std_accuracy']:.3f}")
            
            # Top features
            top_features = [f[0] for f in stability['top_features'][:3]]
            report.append(f"Top selected features: {top_features}")
            report.append("")
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
        
        return report_text


def main():
    """Main function for enhanced RFECV analysis"""
    # Load dataset
    data_path = "../../data/demo_data/classification.csv"
    df = pd.read_csv(data_path)
    
    print(f"Dataset loaded: {len(df)} samples")
    
    # Initialize analyzer
    analyzer = RFECVFeatureSelector()
    
    # Analyze feature correlations
    print("Analyzing feature correlations...")
    correlation_results = analyzer.analyze_feature_correlations(df, save_dir="results")
    
    # Prepare feature sets
    clinical_features = ['sex', 'age', 'Hb', 'WBC', 'PLT', 'Reti%']
    available_clinical = [f for f in clinical_features if f in df.columns]
    
    detection_features = [col for col in df.columns if col not in 
                         ['Label', 'Image_ID'] + clinical_features + 
                         ['Driver_gene', 'VAF', 'data_source', 'Hospital', 'Binary_Label']]
    
    # Binary classification analysis
    print("\nBinary Classification Analysis")
    print("=" * 50)
    
    # Prepare binary labels
    df_binary = df[df['Label'].isin(['ET', 'PV', 'PMF', 'Lymphoma'])].copy()
    df_binary['binary_target'] = df_binary['Label'].apply(lambda x: 0 if x == 'Lymphoma' else 1)
    
    # Clinical features only
    if available_clinical:
        X_clinical = df_binary[available_clinical]
        y_binary = df_binary['binary_target']
        
        # Handle missing values
        for col in X_clinical.columns:
            X_clinical[col] = pd.to_numeric(X_clinical[col], errors='coerce')
            median_val = X_clinical[col].median()
            X_clinical[col] = X_clinical[col].fillna(median_val)
        
        analyzer.run_enhanced_rfecv(X_clinical, y_binary, 'binary_clinical', 'binary')
    
    # Multiclass classification analysis  
    print("\nMulticlass Classification Analysis")
    print("=" * 50)
    
    # Prepare multiclass labels (MPN subtypes only)
    df_multiclass = df[df['Label'].isin(['ET', 'PV', 'PMF'])].copy()
    label_encoder = LabelEncoder()
    df_multiclass['multiclass_target'] = label_encoder.fit_transform(df_multiclass['Label'])
    
    if available_clinical:
        X_clinical_multi = df_multiclass[available_clinical]
        y_multiclass = df_multiclass['multiclass_target']
        
        # Handle missing values
        for col in X_clinical_multi.columns:
            X_clinical_multi[col] = pd.to_numeric(X_clinical_multi[col], errors='coerce')
            median_val = X_clinical_multi[col].median()
            X_clinical_multi[col] = X_clinical_multi[col].fillna(median_val)
        
        analyzer.run_enhanced_rfecv(X_clinical_multi, y_multiclass, 'multiclass_clinical', 'multiclass')
    
    # Generate and save report
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    report = analyzer.generate_report(results_dir / "enhanced_rfecv_analysis_report.txt")
    print("\nAnalysis completed. Results saved to 'results/' directory.")
    print("\nSummary:")
    print(report[:500] + "..." if len(report) > 500 else report)


if __name__ == "__main__":
    main()