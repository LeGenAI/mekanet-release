# MekaNet Detection - API Reference

Complete API documentation for the MekaNet detection framework modules.

## 📋 Table of Contents

- [Core Modules](#core-modules)
- [Paper Reproduction Runner](#paper-reproduction-runner)
- [Benchmark Comparison](#benchmark-comparison)
- [Results Aggregator](#results-aggregator)
- [External Validation Processor](#external-validation-processor)
- [Configuration Classes](#configuration-classes)
- [Utility Functions](#utility-functions)

## 🧩 Core Modules

### PaperReproductionRunner

Main orchestrator for running detection experiments.

```python
from experiments.detection.paper_reproduction_runner import PaperReproductionRunner

class PaperReproductionRunner:
    """
    Orchestrates the complete paper reproduction experiment pipeline.
    """
    
    def __init__(self, config_path: str, base_output_dir: str = "experiments/detection/results"):
        """
        Initialize the paper reproduction runner.
        
        Args:
            config_path: Path to YAML configuration file
            base_output_dir: Base directory for results output
        """
    
    def run_experiments(self, 
                       dry_run: bool = False,
                       resume: bool = False,
                       specific_experiments: List[str] = None) -> Dict[str, Any]:
        """
        Run the complete experiment suite.
        
        Args:
            dry_run: Simulate experiments without actual processing
            resume: Resume from previously saved checkpoint
            specific_experiments: List of specific experiments to run
            
        Returns:
            Dictionary containing experiment results and metadata
        """
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """
        Generate comprehensive paper reproduction report.
        
        Args:
            results: Experiment results dictionary
            
        Returns:
            Path to generated report file
        """
```

**Usage Example:**
```python
runner = PaperReproductionRunner("configs/paper_reproduction_full.yaml")
results = runner.run_experiments(dry_run=False)
report_path = runner.generate_report(results)
```

### BenchmarkComparator

Compares TESSD against baseline detection methods.

```python
from experiments.detection.benchmark_comparison import BenchmarkComparator

class BenchmarkComparator:
    """
    Framework for comparing detection methods against baselines.
    """
    
    def __init__(self, output_dir: str = "results/benchmark_comparison"):
        """
        Initialize benchmark comparator.
        
        Args:
            output_dir: Directory for comparison results
        """
    
    def add_method(self, method: BaseDetectionMethod) -> None:
        """
        Add a detection method for comparison.
        
        Args:
            method: Detection method implementing BaseDetectionMethod interface
        """
    
    def run_comparison(self, 
                      datasets: Dict[str, str],
                      metrics: List[str] = ["mAP", "precision", "recall"]) -> Dict[str, Any]:
        """
        Run benchmark comparison across all methods.
        
        Args:
            datasets: Dictionary mapping dataset names to paths
            metrics: List of metrics to evaluate
            
        Returns:
            Comparison results dictionary
        """
    
    def generate_comparison_report(self, results: Dict[str, Any]) -> str:
        """
        Generate publication-ready comparison report.
        
        Args:
            results: Comparison results
            
        Returns:
            Path to generated report
        """
```

**Usage Example:**
```python
comparator = BenchmarkComparator()
comparator.add_method(StandardYOLOMethod())
comparator.add_method(TESSDMethod())

results = comparator.run_comparison({
    "B_hospital": "data/B_hospital_100_images",
    "S_hospital": "data/S_hospital_5_images"
})
```

### ResultsAggregator

Processes and aggregates experimental results.

```python
from experiments.detection.results_aggregator import ResultsAggregator

class ResultsAggregator:
    """
    Aggregates and processes experimental results for publication.
    """
    
    def __init__(self, results_dir: str):
        """
        Initialize results aggregator.
        
        Args:
            results_dir: Directory containing experimental results
        """
    
    def aggregate_experiment_results(self, experiment_dirs: List[str]) -> Dict[str, Any]:
        """
        Aggregate results from multiple experiment directories.
        
        Args:
            experiment_dirs: List of experiment result directories
            
        Returns:
            Aggregated results dictionary
        """
    
    def generate_latex_tables(self, results: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate LaTeX tables for paper publication.
        
        Args:
            results: Aggregated results
            
        Returns:
            Dictionary mapping table names to LaTeX code
        """
    
    def create_publication_figures(self, 
                                 results: Dict[str, Any],
                                 output_dir: str) -> List[str]:
        """
        Create publication-quality figures.
        
        Args:
            results: Aggregated results
            output_dir: Directory for figure output
            
        Returns:
            List of generated figure paths
        """
```

### ExternalValidationProcessor

Handles validation on external datasets.

```python
from experiments.detection.external_validation_processor import ExternalValidationProcessor

class ExternalValidationProcessor:
    """
    Processes external validation datasets for generalization assessment.
    """
    
    def __init__(self, model_path: str, output_dir: str = "results/external_validation"):
        """
        Initialize external validation processor.
        
        Args:
            model_path: Path to trained model
            output_dir: Directory for validation results
        """
    
    def analyze_domain_gap(self, 
                          source_dataset: str,
                          target_dataset: str) -> DomainGapAnalysis:
        """
        Analyze domain gap between source and target datasets.
        
        Args:
            source_dataset: Path to source (training) dataset
            target_dataset: Path to target (external) dataset
            
        Returns:
            Domain gap analysis results
        """
    
    def validate_external_dataset(self, 
                                 dataset_config: ExternalDatasetConfig) -> Dict[str, Any]:
        """
        Validate model on external dataset.
        
        Args:
            dataset_config: External dataset configuration
            
        Returns:
            Validation results dictionary
        """
```

## 📊 Configuration Classes

### ExperimentConfig

Configuration dataclass for individual experiments.

```python
@dataclass
class ExperimentConfig:
    """Configuration for a single detection experiment."""
    
    name: str
    type: str
    enabled: bool = True
    models: List[str] = field(default_factory=lambda: ["yolov8n"])
    confidence_thresholds: List[float] = field(default_factory=lambda: [0.15, 0.20, 0.25])
    datasets: Dict[str, str] = field(default_factory=dict)
    metrics: List[str] = field(default_factory=lambda: ["mAP", "precision", "recall"])
    expected_runtime_minutes: int = 15
    output_dir: str = "results"
    
    @classmethod
    def from_yaml(cls, yaml_path: str, experiment_name: str) -> 'ExperimentConfig':
        """Load experiment config from YAML file."""
        
    def validate(self) -> bool:
        """Validate configuration parameters."""
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
```

### SAHIConfig

Configuration for SAHI-based inference.

```python
@dataclass
class SAHIConfig:
    """Configuration for SAHI inference parameters."""
    
    slice_height: int = 640
    slice_width: int = 640
    overlap_height_ratio: float = 0.2
    overlap_width_ratio: float = 0.2
    postprocess_type: str = "GREEDYNMMS"
    postprocess_match_metric: str = "IOS"
    postprocess_match_threshold: float = 0.5
    postprocess_class_agnostic: bool = False
```

### DomainGapAnalysis

Results from domain gap analysis.

```python
@dataclass
class DomainGapAnalysis:
    """Results from domain gap analysis between datasets."""
    
    brightness_difference: float
    contrast_difference: float
    color_distribution_divergence: float
    average_cell_size_ratio: float
    cell_density_ratio: float
    resolution_compatibility: float
    noise_level_difference: float
    overall_gap_score: float  # 0-1, where 0 is identical domains
    
    def get_adaptation_strategy(self) -> str:
        """Get recommended adaptation strategy based on gap analysis."""
        
    def is_high_gap(self, threshold: float = 0.7) -> bool:
        """Check if domain gap exceeds threshold."""
```

## 🛠️ Utility Functions

### Configuration Loading

```python
def load_experiment_config(config_path: str) -> Dict[str, ExperimentConfig]:
    """
    Load experiment configurations from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Dictionary mapping experiment names to configurations
    """

def validate_config_file(config_path: str) -> Tuple[bool, List[str]]:
    """
    Validate configuration file syntax and content.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
```

### Metrics Calculation

```python
def calculate_detection_metrics(predictions: List[Dict], 
                              ground_truth: List[Dict],
                              iou_threshold: float = 0.5) -> Dict[str, float]:
    """
    Calculate detection metrics (mAP, precision, recall, F1).
    
    Args:
        predictions: List of prediction dictionaries
        ground_truth: List of ground truth dictionaries
        iou_threshold: IoU threshold for positive detections
        
    Returns:
        Dictionary of calculated metrics
    """

def aggregate_metrics_across_datasets(results: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """
    Aggregate metrics across multiple datasets.
    
    Args:
        results: Dictionary mapping dataset names to metrics
        
    Returns:
        Aggregated metrics dictionary
    """
```

### Visualization Helpers

```python
def create_detection_visualization(image_path: str,
                                 predictions: List[Dict],
                                 ground_truth: List[Dict] = None,
                                 output_path: str = None) -> str:
    """
    Create visualization of detection results.
    
    Args:
        image_path: Path to input image
        predictions: List of predictions
        ground_truth: Optional ground truth annotations
        output_path: Path for output visualization
        
    Returns:
        Path to generated visualization
    """

def plot_metrics_comparison(results: Dict[str, Dict[str, float]],
                          output_path: str = None) -> str:
    """
    Create comparison plot of metrics across methods/datasets.
    
    Args:
        results: Dictionary of results to compare
        output_path: Path for output plot
        
    Returns:
        Path to generated plot
    """
```

## 📁 File Format Specifications

### Results JSON Format

```json
{
    "experiment_name": "baseline_evaluation",
    "timestamp": "2025-01-10T12:00:00Z",
    "config": {
        "models": ["yolov8n", "yolov8s"],
        "datasets": {"training": "...", "validation": "..."}
    },
    "results": {
        "B_hospital": {
            "mAP_50": 0.856,
            "precision": 0.891,
            "recall": 0.823,
            "f1_score": 0.856
        },
        "S_hospital": {
            "mAP_50": 0.798,
            "precision": 0.834,
            "recall": 0.765,
            "f1_score": 0.798
        }
    },
    "performance": {
        "total_runtime_minutes": 12.5,
        "memory_peak_gb": 4.2,
        "gpu_utilization_percent": 78.5
    }
}
```

### Configuration YAML Format

```yaml
baseline_evaluation:
  enabled: true
  type: "comprehensive_evaluation"
  models: 
    - "yolov8n"
    - "yolov8s"
    - "yolov8m"
  confidence_thresholds: [0.15, 0.20, 0.25]
  datasets:
    training: "data/B_hospital_100_images"
    validation: "data/S_hospital_5_images"
  metrics: ["mAP", "precision", "recall", "f1"]
  sahi_config:
    slice_height: 640
    slice_width: 640
    overlap_height_ratio: 0.2
    overlap_width_ratio: 0.2
  expected_runtime_minutes: 15
  output_dir: "results/baseline_evaluation"
```

## 🚨 Error Handling

### Common Exceptions

```python
class MekaNetDetectionError(Exception):
    """Base exception for MekaNet detection framework."""
    pass

class ConfigurationError(MekaNetDetectionError):
    """Raised when configuration is invalid."""
    pass

class DatasetError(MekaNetDetectionError):
    """Raised when dataset loading/validation fails."""
    pass

class ModelError(MekaNetDetectionError):
    """Raised when model operations fail."""
    pass

class ExperimentError(MekaNetDetectionError):
    """Raised when experiment execution fails."""
    pass
```

### Error Handling Patterns

```python
try:
    runner = PaperReproductionRunner(config_path)
    results = runner.run_experiments()
except ConfigurationError as e:
    logger.error(f"Configuration error: {e}")
    # Handle config issues
except DatasetError as e:
    logger.error(f"Dataset error: {e}")
    # Handle data issues
except ModelError as e:
    logger.error(f"Model error: {e}")
    # Handle model issues
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    # Handle unexpected issues
```

## 📝 Usage Examples

### Complete Workflow Example

```python
import yaml
from pathlib import Path
from experiments.detection.paper_reproduction_runner import PaperReproductionRunner
from experiments.detection.results_aggregator import ResultsAggregator

# 1. Setup configuration
config_path = "experiments/detection/configs/paper_reproduction_full.yaml"

# 2. Run experiments
runner = PaperReproductionRunner(config_path)
results = runner.run_experiments(dry_run=False)

# 3. Generate comprehensive report
report_path = runner.generate_report(results)
print(f"Report generated: {report_path}")

# 4. Aggregate results for publication
aggregator = ResultsAggregator(runner.output_dir)
aggregated = aggregator.aggregate_experiment_results(
    list(results.keys())
)

# 5. Generate LaTeX tables
latex_tables = aggregator.generate_latex_tables(aggregated)
for table_name, latex_code in latex_tables.items():
    with open(f"{table_name}.tex", "w") as f:
        f.write(latex_code)

# 6. Create publication figures
figure_paths = aggregator.create_publication_figures(
    aggregated, 
    "publication_figures"
)
print(f"Generated {len(figure_paths)} publication figures")
```

---

**API Version**: 1.0.0  
**Last Updated**: January 2025  
**Python Compatibility**: 3.8+ 