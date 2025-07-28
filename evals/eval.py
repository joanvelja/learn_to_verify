"""
Main evaluation script for the backdoor detection pipeline.

This script provides both the legacy interface and access to the new refactored
evaluation system with support for calibrated scoring and threshold analysis.
"""

import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# Import the new evaluation system
try:
    from .core import EvaluationConfig
    from .evaluator import EvaluationResults, Evaluator
    from .evaluator import evaluate as new_evaluate
except ImportError:
    # Fallback for direct script execution
    from core import EvaluationConfig
    from evaluator import EvaluationResults, Evaluator
    from evaluator import evaluate as new_evaluate


def evaluate(prompt_type: str = "non_calibrated", output_dir: str = ".") -> None:
    """
    Legacy evaluation function for backward compatibility.

    This function maintains the same interface as the original evaluate function
    but uses the new refactored architecture under the hood.

    Args:
        prompt_type: Either "calibrated" or "non_calibrated"
        output_dir: Directory to save results
    """
    print(f"=== Starting Evaluation with {prompt_type} prompts ===")
    print()

    # Use the new evaluation system
    output_path = Path(output_dir) if output_dir != "." else None
    new_evaluate(prompt_type=prompt_type, output_dir=output_path)


def evaluate_single_model(
    model_name: str, prompt_type: str = "non_calibrated", use_cot: bool = True, output_dir: str = "."
) -> EvaluationResults:
    """
    Evaluate a single model with the new architecture.

    Args:
        model_name: Name of the model to evaluate
        prompt_type: Either "calibrated" or "non_calibrated"
        use_cot: Whether to use chain-of-thought reasoning
        output_dir: Directory to save results

    Returns:
        EvaluationResults object containing all results
    """
    config = EvaluationConfig(prompt_type=prompt_type)
    evaluator = Evaluator()

    return evaluator.evaluate_model(
        model_name=model_name,
        config=config,
        use_cot=use_cot,
        output_dir=Path(output_dir) if output_dir != "." else None,
    )


def evaluate_with_custom_config(
    model_names: list[str], config: EvaluationConfig, cot_modes: list[bool] = [True, False], output_dir: str = "."
) -> dict:
    """
    Evaluate multiple models with custom configuration.

    Args:
        model_names: List of model names to evaluate
        config: Custom evaluation configuration
        cot_modes: List of CoT modes to test
        output_dir: Directory to save results

    Returns:
        Dictionary mapping model_name -> {use_cot -> EvaluationResults}
    """
    evaluator = Evaluator()

    return evaluator.evaluate_multiple_models(
        model_names=model_names,
        config=config,
        cot_modes=cot_modes,
        output_dir=Path(output_dir) if output_dir != "." else None,
    )


def main():
    """Main entry point for the evaluation script."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate models on backdoor detection")
    parser.add_argument(
        "--prompt-type",
        choices=["calibrated", "non_calibrated"],
        default="non_calibrated",
        help="Type of prompts to use",
    )
    parser.add_argument("--output-dir", default=".", help="Directory to save results")
    parser.add_argument("--model", help="Evaluate a single model instead of the default list")
    parser.add_argument(
        "--cot-only", action="store_true", help="Only evaluate with chain-of-thought (when using --model)"
    )
    parser.add_argument(
        "--no-cot", action="store_true", help="Only evaluate without chain-of-thought (when using --model)"
    )

    args = parser.parse_args()

    if args.model:
        # Single model evaluation
        cot_modes = []
        if args.cot_only:
            cot_modes = [True]
        elif args.no_cot:
            cot_modes = [False]
        else:
            cot_modes = [True, False]

        config = EvaluationConfig(prompt_type=args.prompt_type)
        evaluator = Evaluator()

        evaluator.evaluate_multiple_models(
            model_names=[args.model],
            config=config,
            cot_modes=cot_modes,
            output_dir=Path(args.output_dir) if args.output_dir != "." else None,
        )
    else:
        # Full evaluation with default models
        evaluate(prompt_type=args.prompt_type, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
