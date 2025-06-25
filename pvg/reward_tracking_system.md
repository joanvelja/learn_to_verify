# Reward Tracking and Logging System

## Overview

This document describes the comprehensive reward tracking and logging system implemented for monitoring and debugging reward distributions in prover training.

## Architecture

### Core Components

1. **ProverTrainingPipeline**: Orchestrates reward calculation and storage
2. **RewardResult**: Data structure containing rewards, advantages, and metrics
3. **MetricsProcessor**: Computes advanced reward statistics
4. **MetricsLogger**: Stores and aggregates metrics across processes

### Data Flow

```
Raw Batch → Completions → Verification → Rewards → Metrics → Storage → Display
```

## Key Features

### 1. Comprehensive Reward Statistics

The system tracks:
- **Basic Statistics**: mean, std, min, max for honest and sneaky rewards
- **Distribution Metrics**: positive/negative/zero ratios
- **Comparative Metrics**: reward gaps between honest and sneaky provers
- **Behavioral Metrics**: extraction rates, test pass rates, trigger activation rates

### 2. Real-time Progress Bar Integration

Progress bars now display:
- `s_reward`: Sneaky prover reward mean
- `h_reward`: Honest prover reward mean
- `extract`: Extraction success rate
- `loss`: Training loss
- `v_acc`: Verifier accuracy

### 3. Automated Logging

#### Batch-level Logging
- Reward statistics logged after each batch processing
- Behavioral metrics tracked and stored
- Debug information with reward distributions

#### Periodic Summaries
- Comprehensive reward summaries every 100 training steps
- Comparative analysis between honest and sneaky provers
- Easy-to-read formatted output for debugging

### 4. Extensible Metrics Processing

The `MetricsProcessor` class provides:
- `compute_reward_metrics()`: Basic reward statistics
- `compute_reward_distribution_metrics()`: Comparative analysis
- Easy extension for custom reward metrics

## Usage Examples

### Getting Latest Metrics

```python
# Get current reward metrics
metrics = pipeline.get_latest_reward_metrics()
sneaky_mean = metrics["sneaky_rewards_mean"]
reward_gap = metrics["reward_gap"]
```

### Manual Reward Summary

```python
# Log comprehensive reward summary
pipeline.log_reward_summary()
```

### Custom Reward Statistics

```python
# Compute custom reward metrics
processor = MetricsProcessor()
metrics = processor.compute_reward_distribution_metrics(honest_rewards, sneaky_rewards)
```

## Integration Points

### 1. Training Pipeline

- `process_batch()`: Automatically stores reward metrics
- `get_progress_metrics()`: Provides formatted metrics for progress bars
- `log_reward_summary()`: Comprehensive logging

### 2. Reward Strategies

- Both `TierBasedRewardStrategy` and `SanityCheckRewardStrategy` use the system
- Automatic integration with `MetricsProcessor` for enhanced statistics
- Backward compatibility with existing implementations

### 3. Trainer Integration

- Progress bars automatically display reward metrics
- Periodic reward summaries during training
- Integration with evaluation cycles

## Configuration

### Enabling Enhanced Metrics

The system automatically uses enhanced metrics when `MetricsProcessor` is available:

```python
# Enhanced statistics (automatic)
reward_strategy = TierBasedRewardStrategy(
    metrics_processor=metrics_processor,  # Enables enhanced metrics
    # ... other args
)

# Basic statistics (fallback)
reward_strategy = TierBasedRewardStrategy(
    # metrics_processor not provided - uses basic stats
    # ... other args
)
```

### Logging Frequency

- **Batch-level**: Every batch (lightweight metrics)
- **Summary-level**: Every 100 training steps (comprehensive analysis)
- **Evaluation-level**: After each evaluation cycle

## Monitoring and Debugging

### Key Metrics to Watch

1. **Reward Progression**: Are rewards improving over time?
2. **Reward Gap**: How do sneaky vs honest rewards compare?
3. **Extraction Rates**: Are models successfully generating valid code?
4. **Behavioral Alignment**: Are sneaky models activating triggers appropriately?

### Common Debugging Patterns

1. **Flat Rewards**: Check if verifier bounds are properly initialized
2. **No Reward Gap**: Verify that honest and sneaky strategies are different
3. **Low Extraction Rates**: Debug completion generation and parsing
4. **Inconsistent Triggers**: Examine backdoor activation logic

## Future Extensions

The system is designed for easy extension:

1. **Custom Reward Metrics**: Add new metrics to `MetricsProcessor`
2. **Advanced Visualizations**: Use metrics for real-time dashboards
3. **Reward Analysis Tools**: Build analysis scripts using the metric APIs
4. **Integration with External Monitoring**: Export metrics to monitoring systems

## Best Practices

1. **Monitor Both Honest and Sneaky**: Always track both prover types
2. **Check Extraction Rates**: Ensure code generation is working
3. **Watch Reward Distributions**: Look for healthy reward variance
4. **Use Periodic Summaries**: Regular comprehensive logging for debugging
5. **Validate Reward Alignment**: Ensure rewards align with training objectives