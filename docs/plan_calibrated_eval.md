# Calibrated Evaluation Pipeline Refactoring Plan

## Problem Statement
The current evaluation pipeline is monolithic and only supports binary classification (clean vs backdoor), limiting flexibility in threshold tuning and metric optimization. We need to implement calibrated scoring (1-7 scale) with threshold-based metrics and refactor the code for better maintainability.

## Requirements Analysis

### Functional Requirements
1. **Calibrated Scoring**: Support 1-7 scale scoring instead of binary classification
2. **Threshold Analysis**: Calculate metrics for all possible thresholds (1-7)
3. **Metric Flexibility**: Allow threshold tuning to optimize FPR/FNR trade-offs
4. **Backward Compatibility**: Maintain support for binary classification mode
5. **Comprehensive Metrics**: Track accuracy, precision, recall, F1, FPR, FNR across thresholds

### Non-Functional Requirements
1. **Maintainability**: Clean, modular architecture following SOLID principles
2. **Extensibility**: Easy to add new prompt types, models, or metrics
3. **Developer Experience**: Clear interfaces, comprehensive documentation
4. **Performance**: Efficient batch processing and memory management

## Component Architecture

### 1. Core Interfaces (Protocols)
```python
class VerdictExtractor(Protocol):
    def extract(self, output: str) -> str | int: ...

class MetricsCalculator(Protocol):
    def calculate_metrics(self, predictions: list, labels: list) -> dict: ...

class ModelManager(Protocol):
    def generate_responses(self, prompts: list) -> list[str]: ...
```

### 2. Verdict Extraction Strategy
- **BinaryVerdictExtractor**: Extracts "clean"/"backdoor" verdicts
- **CalibratedVerdictExtractor**: Extracts numeric scores (1-7)
- **Factory Pattern**: Select extractor based on prompt type

### 3. Metrics Calculation Strategy
- **BinaryMetricsCalculator**: Standard binary classification metrics
- **ThresholdMetricsCalculator**: Metrics across all thresholds for calibrated scores
- **Compose results**: Unified interface for both approaches

### 4. Model and Data Management
- **ModelManager**: Handles LLM initialization, configuration, cleanup
- **DataProcessor**: Manages dataset loading, prompt formatting
- **Separation of Concerns**: Clear boundaries between components

### 5. Main Orchestrator
- **Evaluator**: Coordinates all components using dependency injection
- **Configuration-Driven**: Easy to modify evaluation parameters
- **Clean Error Handling**: Proper resource cleanup and error reporting

## Implementation Phases

### Phase 1: Core Abstractions
1. Define protocols and interfaces
2. Implement verdict extractors
3. Create metrics calculators
4. Unit tests for core logic

### Phase 2: Component Implementation
1. ModelManager with proper cleanup
2. DataProcessor with prompt templating
3. Configuration management
4. Integration tests

### Phase 3: Main Orchestrator
1. Evaluator class with dependency injection
2. Results persistence and reporting
3. Threshold analysis visualization
4. End-to-end testing

### Phase 4: Migration and Testing
1. Update main eval.py
2. Fix import issues
3. Comprehensive testing
4. Documentation updates

## Data Flow Architecture

```
Dataset → DataProcessor → [Prompts] → ModelManager → [Raw Outputs]
    ↓
VerdictExtractor → [Verdicts/Scores] → MetricsCalculator → [Results]
    ↓
Evaluator → [Threshold Analysis] → [CSV/JSON Output]
```

## Key Decisions

### Calibrated Scoring Approach
- Extract numeric scores (1-7) from model outputs using regex patterns
- For each threshold T (1-7): classify score >= T as "backdoor", < T as "clean"
- Calculate full confusion matrix and derived metrics for each threshold
- Save threshold-dependent results for analysis

### Abstraction Strategy
- Use Protocol-based design for runtime flexibility
- Factory patterns for strategy selection
- Composition over inheritance for feature combination
- Dependency injection for testability

### Error Handling
- Domain-specific exceptions for different failure modes
- Graceful degradation when models fail
- Comprehensive logging for debugging
- Proper resource cleanup (GPU memory, model instances)

## Testing Strategy
- Unit tests for each component in isolation
- Integration tests for component interaction
- End-to-end tests with mock models
- Performance tests for large-scale evaluation

## Acceptance Criteria
- [ ] Calibrated scoring works with threshold analysis
- [ ] Binary mode maintains backward compatibility
- [ ] Code is modular and easy to extend
- [ ] All imports work correctly
- [ ] Memory management is efficient
- [ ] Results are saved in analyzable format
- [ ] Documentation covers usage and extension patterns