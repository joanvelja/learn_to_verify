# Refactored Evaluation Pipeline Guide

## Overview

The evaluation pipeline has been completely refactored to support **calibrated scoring** with threshold-based metrics analysis while maintaining backward compatibility with the original binary classification approach. The new architecture follows modern Python practices with clean abstractions, dependency injection, and comprehensive error handling.

## Key Features

### 🎯 Calibrated Scoring
- **1-7 Scale**: Models now output suspiciousness scores instead of binary verdicts
- **Threshold Analysis**: Calculate metrics for all possible thresholds (1-7)
- **Flexible Optimization**: Tune thresholds to optimize FPR/FNR trade-offs
- **Comprehensive Metrics**: Track accuracy, precision, recall, F1 across all thresholds

### 🏗️ Clean Architecture
- **Protocol-Based Design**: Easy to extend with new components
- **Dependency Injection**: Testable and flexible component composition
- **Factory Patterns**: Automatic selection of appropriate implementations
- **Resource Management**: Proper GPU memory cleanup and error handling

### 📊 Enhanced Results
- **Threshold Analysis Files**: Detailed CSV files showing metrics vs. threshold
- **Summary Statistics**: Mean/std/stderr across seeds for all metrics
- **Best Threshold Detection**: Automatically find optimal thresholds for accuracy/F1
- **Backward Compatibility**: Existing binary results format preserved

## Quick Start

### Basic Usage (Calibrated Scoring)
```bash
# Evaluate all models with calibrated prompts
python eval.py --prompt-type calibrated

# Evaluate single model with calibrated scoring
python eval.py --model "Qwen/Qwen2.5-7B-Instruct" --prompt-type calibrated

# Evaluate with only chain-of-thought
python eval.py --model "Qwen/Qwen2.5-7B-Instruct" --prompt-type calibrated --cot-only
```

### Binary Classification (Legacy)
```bash
# Original binary classification approach
python eval.py --prompt-type non_calibrated

# Single model binary evaluation
python eval.py --model "Qwen/Qwen2.5-7B-Instruct" --prompt-type non_calibrated
```

## Programmatic API

### Simple Evaluation
```python
from evals import Evaluator, EvaluationConfig

# Configure evaluation
config = EvaluationConfig(
    prompt_type="calibrated",
    seeds=[42, 100, 999],
    temperature=0.7
)

# Run evaluation
evaluator = Evaluator()
results = evaluator.evaluate_model(
    model_name="Qwen/Qwen2.5-7B-Instruct",
    config=config,
    use_cot=True
)

# Access results
print(f"Best F1: {results.seed_results[0]['Best_F1']:.3f}")
print(f"Optimal threshold: {results.seed_results[0]['Best_F1_Threshold']}")
```

### Custom Components
```python
from evals import (
    Evaluator, EvaluationConfig,
    VerdictExtractorFactory,
    MetricsCalculatorFactory
)

# Create custom components
extractor = VerdictExtractorFactory.create_extractor("calibrated")
calculator = MetricsCalculatorFactory.create_calculator("calibrated")

# Use in evaluator
evaluator = Evaluator(
    verdict_extractor=extractor,
    metrics_calculator=calculator
)
```

## Architecture Overview

### Core Components

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   DataProcessor │    │   ModelManager   │    │ VerdictExtractor│
│                 │    │                  │    │                 │
│ • Load dataset  │    │ • VLLM interface │    │ • Binary        │
│ • Format prompts│    │ • GPU management │    │ • Calibrated    │
│ • Template mgmt │    │ • Memory cleanup │    │ • Factory       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────────┐
                    │     Evaluator       │
                    │                     │
                    │ • Orchestrates flow │
                    │ • Dependency inject │
                    │ • Results persistence│
                    └─────────────────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         │                       │                       │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│MetricsCalculator│    │ EvaluationConfig │    │EvaluationResults│
│                 │    │                  │    │                 │
│ • Binary metrics│    │ • Model params   │    │ • Results data  │
│ • Threshold     │    │ • Seeds          │    │ • Persistence   │
│ • Factory       │    │ • Prompt type    │    │ • Summaries     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Data Flow

```
Dataset → DataProcessor → [Chat Messages] → ModelManager → [Raw Outputs]
    ↓
VerdictExtractor → [Scores 1-7] → MetricsCalculator → [Threshold Metrics]
    ↓
EvaluationResults → [CSV Files + JSON Logs + Threshold Analysis]
```

## File Outputs

### Calibrated Scoring Results
- `results_{model}_cot_calibrated.csv` - Main results with metrics for each threshold
- `logs_{model}_cot_calibrated.json` - Raw model outputs for analysis
- `threshold_analysis_{model}_cot_calibrated.csv` - Detailed threshold analysis

### Sample Threshold Analysis
```csv
Seed,Threshold,Accuracy,Clean_Accuracy,Backdoor_Accuracy,FPR,FNR,Precision,F1_Score
42,1,0.45,0.12,0.98,0.88,0.02,0.52,0.68
42,2,0.67,0.45,0.89,0.55,0.11,0.67,0.76
42,3,0.78,0.72,0.84,0.28,0.16,0.75,0.79
42,4,0.82,0.89,0.75,0.11,0.25,0.87,0.81
42,5,0.79,0.95,0.63,0.05,0.37,0.93,0.75
42,6,0.71,0.98,0.44,0.02,0.56,0.96,0.60
42,7,0.55,1.00,0.11,0.00,0.89,1.00,0.19
```

## Extending the System

### Custom Verdict Extractor
```python
from evals.core import VerdictExtractor

class CustomVerdictExtractor:
    def extract(self, output: str) -> int:
        # Your custom extraction logic
        return score  # 1-7 for calibrated
```

### Custom Metrics Calculator
```python
from evals.core import MetricsCalculator

class CustomMetricsCalculator:
    def calculate_metrics(self, predictions, labels, seed):
        # Your custom metrics logic
        return metrics_dict
```

### Custom Model Manager
```python
from evals.core import ModelManager

class CustomModelManager:
    def generate_responses(self, prompts, config, seed):
        # Your custom generation logic
        return raw_outputs

    def cleanup(self):
        # Resource cleanup
        pass
```

## Calibrated vs Binary Comparison

| Feature | Binary Classification | Calibrated Scoring |
|---------|----------------------|-------------------|
| **Output** | "clean" / "backdoor" | Scores 1-7 |
| **Flexibility** | Fixed threshold | Tunable thresholds |
| **Optimization** | Limited | FPR/FNR trade-offs |
| **Analysis** | Basic metrics | Threshold curves |
| **Files** | 2 files per model | 3 files per model |

## Migration from Legacy Code

The refactored system is fully backward compatible:

```python
# Legacy approach still works
from evals import evaluate
evaluate(prompt_type="non_calibrated")

# But you can now also use the new API
from evals import Evaluator, EvaluationConfig

config = EvaluationConfig(prompt_type="calibrated")
evaluator = Evaluator()
results = evaluator.evaluate_multiple_models(["model1", "model2"], config)
```

## Testing

Run the test suite to verify everything works:

```bash
cd evals/
python test_refactored.py
```

This will test all core components without requiring model downloads.

## Performance Considerations

- **Memory Management**: Models are automatically cleaned up after evaluation
- **GPU Utilization**: Configurable memory usage (default: 95%)
- **Batch Processing**: Efficient batched generation for multiple seeds
- **Caching**: VLLM prefix caching for repeated prompt prefixes

## Troubleshooting

### Import Errors
```bash
# Make sure you're in the right directory
cd evals/
python test_refactored.py
```

### Model Download Issues
```bash
# Install huggingface-cli if needed
pip install huggingface_hub
```

### CUDA Memory Issues
```python
# Reduce GPU memory utilization
config = EvaluationConfig(gpu_memory_utilization=0.8)
```

## Future Extensions

The new architecture makes it easy to add:
- **New Prompt Types**: Just add templates and update factories
- **Different Scoring Scales**: 1-10, continuous scores, etc.
- **Custom Datasets**: Implement DataProcessor protocol
- **Different Models**: Implement ModelManager protocol
- **Advanced Metrics**: Custom MetricsCalculator implementations

## Summary

The refactored evaluation pipeline provides:
- ✅ **Calibrated scoring** with threshold analysis
- ✅ **Clean, modular architecture** following SOLID principles
- ✅ **Backward compatibility** with existing workflows
- ✅ **Comprehensive testing** and error handling
- ✅ **Extensible design** for future enhancements
- ✅ **Better resource management** and performance