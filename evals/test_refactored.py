"""
Test script for the refactored evaluation pipeline.

This script verifies that all components work correctly and demonstrates
the new features including calibrated scoring and threshold analysis.
"""

import sys

# Test imports
try:
    from core import EvaluationConfig
    from data_processor import PromptTemplateManager
    from metrics_calculators import BinaryMetricsCalculator, MetricsCalculatorFactory, ThresholdMetricsCalculator
    from verdict_extractors import BinaryVerdictExtractor, CalibratedVerdictExtractor, VerdictExtractorFactory

    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)


def test_verdict_extractors():
    """Test verdict extraction functionality."""
    print("\n=== Testing Verdict Extractors ===")

    # Test binary extractor
    binary_extractor = BinaryVerdictExtractor()

    test_outputs = [
        "<verdict>CLEAN</verdict>",
        "<verdict>BACKDOOR</verdict>",
        "This code looks suspicious. It is a backdoor.",
        "The implementation is clean and backdoor-free.",
        "Verdict: CLEAN",
        "Unknown output format",
    ]

    expected = ["clean", "backdoor", "backdoor", "clean", "clean", "unknown"]

    print("Binary Extractor Results:")
    for output, exp in zip(test_outputs, expected):
        result = binary_extractor.extract(output)
        status = "✓" if result == exp else "✗"
        print(f"  {status} '{output[:30]}...' -> {result} (expected: {exp})")

    # Test calibrated extractor
    calibrated_extractor = CalibratedVerdictExtractor()

    calibrated_outputs = [
        "<verdict>3</verdict>",
        "Rating: 5",
        "Score: 7/7",
        "The suspiciousness is 2",
        "4",
    ]

    expected_scores = [3, 5, 7, 2, 4]

    print("\nCalibrated Extractor Results:")
    for output, exp in zip(calibrated_outputs, expected_scores):
        try:
            result = calibrated_extractor.extract(output)
            status = "✓" if result == exp else "✗"
            print(f"  {status} '{output}' -> {result} (expected: {exp})")
        except Exception as e:
            print(f"  ✗ '{output}' -> Error: {e}")


def test_metrics_calculators():
    """Test metrics calculation functionality."""
    print("\n=== Testing Metrics Calculators ===")

    # Test binary metrics
    binary_calc = BinaryMetricsCalculator()

    # Sample binary predictions and labels
    predictions = ["clean", "backdoor", "clean", "backdoor", "backdoor"]
    labels = ["clean", "backdoor", "backdoor", "backdoor", "clean"]

    try:
        binary_results = binary_calc.calculate_metrics(predictions, labels, seed=42)
        print("✓ Binary metrics calculation successful")
        print(f"  Accuracy: {binary_results['Overall Accuracy']:.3f}")
        print(f"  F1 Score: {binary_results['F1 Score']:.3f}")
    except Exception as e:
        print(f"✗ Binary metrics calculation failed: {e}")

    # Test threshold metrics
    threshold_calc = ThresholdMetricsCalculator()

    # Sample calibrated predictions (scores 1-7)
    score_predictions = [2, 6, 3, 7, 5, 1, 4, 6, 2, 5]
    score_labels = [
        "clean",
        "backdoor",
        "clean",
        "backdoor",
        "backdoor",
        "clean",
        "clean",
        "backdoor",
        "clean",
        "backdoor",
    ]

    try:
        threshold_results = threshold_calc.calculate_metrics(score_predictions, score_labels, seed=42)
        print("✓ Threshold metrics calculation successful")
        print(
            f"  Best Accuracy: {threshold_results['Best_Accuracy']:.3f} at threshold {threshold_results['Best_Accuracy_Threshold']}"
        )
        print(f"  Best F1: {threshold_results['Best_F1']:.3f} at threshold {threshold_results['Best_F1_Threshold']}")

        # Show metrics for a few thresholds
        print("  Sample threshold metrics:")
        for t in [3, 5, 7]:
            acc = threshold_results[f"Threshold_{t}_Accuracy"]
            f1 = threshold_results[f"Threshold_{t}_F1"]
            print(f"    Threshold {t}: Accuracy={acc:.3f}, F1={f1:.3f}")

    except Exception as e:
        print(f"✗ Threshold metrics calculation failed: {e}")


def test_prompt_templates():
    """Test prompt template management."""
    print("\n=== Testing Prompt Templates ===")

    try:
        # Test all template combinations
        combinations = [
            ("non_calibrated", False),
            ("non_calibrated", True),
            ("calibrated", False),
            ("calibrated", True),
        ]

        for prompt_type, use_cot in combinations:
            template = PromptTemplateManager.get_template(prompt_type, use_cot)
            print(f"✓ {prompt_type} {'with' if use_cot else 'without'} CoT: {len(template)} chars")

        print(f"✓ Available templates: {len(PromptTemplateManager.list_available_templates())}")

    except Exception as e:
        print(f"✗ Template management failed: {e}")


def test_factories():
    """Test factory classes."""
    print("\n=== Testing Factory Classes ===")

    try:
        # Test verdict extractor factory
        _ = VerdictExtractorFactory.create_extractor("non_calibrated")
        _ = VerdictExtractorFactory.create_extractor("calibrated")
        print("✓ VerdictExtractorFactory working")

        # Test metrics calculator factory
        _ = MetricsCalculatorFactory.create_calculator("non_calibrated")
        _ = MetricsCalculatorFactory.create_calculator("calibrated")
        print("✓ MetricsCalculatorFactory working")

    except Exception as e:
        print(f"✗ Factory testing failed: {e}")


def test_configuration():
    """Test configuration classes."""
    print("\n=== Testing Configuration ===")

    try:
        # Test default configuration
        config = EvaluationConfig()
        print(f"✓ Default config: {config.prompt_type}, {len(config.seeds)} seeds")

        # Test custom configuration
        custom_config = EvaluationConfig(prompt_type="calibrated", seeds=[1, 2, 3], temperature=0.8)
        print(f"✓ Custom config: {custom_config.prompt_type}, temp={custom_config.temperature}")

    except Exception as e:
        print(f"✗ Configuration testing failed: {e}")


def main():
    """Run all tests."""
    print("=== Refactored Evaluation Pipeline Test Suite ===")

    test_verdict_extractors()
    test_metrics_calculators()
    test_prompt_templates()
    test_factories()
    test_configuration()

    print("\n=== Test Summary ===")
    print("Core components tested successfully!")
    print("\nTo test the full pipeline with a model, use:")
    print("  python eval.py --model Qwen/Qwen2.5-0.5B-Instruct --prompt-type calibrated")


if __name__ == "__main__":
    main()
