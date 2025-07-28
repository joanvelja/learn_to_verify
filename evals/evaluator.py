"""
Main evaluation orchestrator.

This module provides the primary Evaluator class that coordinates all components
of the evaluation pipeline using dependency injection for flexibility and testability.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .core import (
    DataProcessor,
    EvaluationConfig,
    MetricsCalculator,
    ModelManager,
    VerdictExtractionError,
    VerdictExtractor,
)
from .data_processor import DataProcessorFactory, PromptTemplateManager
from .metrics_calculators import MetricsCalculatorFactory, create_threshold_summary_dataframe
from .model_manager import ModelManagerFactory
from .verdict_extractors import VerdictExtractorFactory


class EvaluationResults:
    """Container for evaluation results with persistence capabilities."""

    def __init__(
        self,
        model_name: str,
        prompt_type: str,
        use_cot: bool,
        seed_results: List[Dict[str, Any]],
        honest_outputs: List[str],
        sneaky_outputs: List[str],
    ):
        self.model_name = model_name
        self.prompt_type = prompt_type
        self.use_cot = use_cot
        self.seed_results = seed_results
        self.honest_outputs = honest_outputs
        self.sneaky_outputs = sneaky_outputs

        # Create results DataFrame
        self.results_df = pd.DataFrame(seed_results)

        # Calculate summary statistics
        self._calculate_summary_stats()

    def _calculate_summary_stats(self):
        """Calculate summary statistics across seeds."""
        if self.seed_results:
            # Get metric columns (exclude metadata like Seed, _threshold_metrics, etc.)
            metric_columns = [col for col in self.results_df.columns if col != "Seed" and not col.startswith("_")]

            self.summary_stats = {}
            for metric_name in metric_columns:
                if metric_name in self.results_df.columns:
                    values = self.results_df[metric_name].values
                    self.summary_stats[f"{metric_name}_mean"] = float(values.mean())
                    self.summary_stats[f"{metric_name}_std"] = float(values.std())
                    self.summary_stats[f"{metric_name}_stderr"] = float(values.std() / (len(values) ** 0.5))
        else:
            self.summary_stats = {}

    def save_results(self, output_dir: Path = Path(".")):
        """Save evaluation results to files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Determine file suffix
        suffix = f"{'cot' if self.use_cot else 'non_cot'}_{self.prompt_type}"

        # Save main results CSV
        results_file = output_dir / f"results_{self.model_name}_{suffix}.csv"
        self.results_df.to_csv(results_file, index=False)

        # Save logs JSON
        logs_file = output_dir / f"logs_{self.model_name}_{suffix}.json"
        with open(logs_file, "w") as f:
            json.dump({"honest": self.honest_outputs, "sneaky": self.sneaky_outputs}, f, indent=2)

        # For calibrated results, save threshold analysis
        if self.prompt_type == "calibrated" and self.seed_results:
            threshold_df = create_threshold_summary_dataframe(self.seed_results)
            if not threshold_df.empty:
                threshold_file = output_dir / f"threshold_analysis_{self.model_name}_{suffix}.csv"
                threshold_df.to_csv(threshold_file, index=False)

        return {
            "results_file": results_file,
            "logs_file": logs_file,
            "threshold_file": threshold_file if self.prompt_type == "calibrated" else None,
        }

    def get_summary_report(self) -> str:
        """Generate a human-readable summary report."""
        report = [
            f"=== Evaluation Results for {self.model_name} ===",
            f"Prompt Type: {self.prompt_type}",
            f"Chain of Thought: {'Yes' if self.use_cot else 'No'}",
            f"Number of Seeds: {len(self.seed_results)}",
            "",
            "Summary Statistics:",
        ]

        for metric, value in self.summary_stats.items():
            if metric.endswith("_mean"):
                base_metric = metric[:-5]  # Remove "_mean" suffix
                std_key = f"{base_metric}_std"
                if std_key in self.summary_stats:
                    report.append(f"  {base_metric}: {value:.4f} ± {self.summary_stats[std_key]:.4f}")
                else:
                    report.append(f"  {metric}: {value:.4f}")

        return "\n".join(report)


class Evaluator:
    """
    Main evaluation orchestrator.

    Coordinates all components of the evaluation pipeline using dependency injection
    for flexibility and testability.
    """

    def __init__(
        self,
        model_manager: Optional[ModelManager] = None,
        data_processor: Optional[DataProcessor] = None,
        verdict_extractor: Optional[VerdictExtractor] = None,
        metrics_calculator: Optional[MetricsCalculator] = None,
    ):
        """
        Initialize the evaluator with optional component injection.

        Args:
            model_manager: Model management component
            data_processor: Data processing component
            verdict_extractor: Verdict extraction component
            metrics_calculator: Metrics calculation component
        """
        self.model_manager = model_manager
        self.data_processor = data_processor
        self.verdict_extractor = verdict_extractor
        self.metrics_calculator = metrics_calculator

    def evaluate_model(
        self, model_name: str, config: EvaluationConfig, use_cot: bool, output_dir: Optional[Path] = None
    ) -> EvaluationResults:
        """
        Evaluate a single model with the given configuration.

        Args:
            model_name: Name of the model to evaluate
            config: Evaluation configuration
            use_cot: Whether to use chain-of-thought reasoning
            output_dir: Directory to save results (optional)

        Returns:
            EvaluationResults containing all results and metadata
        """
        print(f"Evaluating {model_name} {'with' if use_cot else 'without'} CoT ({config.prompt_type})")

        # Initialize components if not provided
        model_manager = self.model_manager or ModelManagerFactory.create_manager(model_name)
        data_processor = self.data_processor or DataProcessorFactory.create_processor()
        verdict_extractor = self.verdict_extractor or VerdictExtractorFactory.create_extractor(config.prompt_type)
        metrics_calculator = self.metrics_calculator or MetricsCalculatorFactory.create_calculator(config.prompt_type)

        try:
            # Load evaluation data
            evaluation_data = data_processor.load_data()

            # Get appropriate prompt template
            prompt_template = PromptTemplateManager.get_template(config.prompt_type, use_cot)

            # Format prompts
            honest_prompts, sneaky_prompts = data_processor.format_prompts(evaluation_data, prompt_template, use_cot)

            # Run evaluation across all seeds
            all_seed_results = []
            all_honest_outputs = []
            all_sneaky_outputs = []

            for seed in config.seeds:
                print(f"  Running seed {seed}...")

                # Generate responses for honest solutions
                honest_raw_outputs = model_manager.generate_responses(honest_prompts, config, seed)

                # Generate responses for sneaky solutions
                sneaky_raw_outputs = model_manager.generate_responses(sneaky_prompts, config, seed)

                # Store raw outputs for logging
                all_honest_outputs.extend(honest_raw_outputs)
                all_sneaky_outputs.extend(sneaky_raw_outputs)

                # Extract verdicts
                honest_verdicts = []
                sneaky_verdicts = []

                for output in honest_raw_outputs:
                    try:
                        verdict = verdict_extractor.extract(output)
                        honest_verdicts.append(verdict)
                    except VerdictExtractionError:
                        # Handle extraction failures gracefully
                        honest_verdicts.append("unknown" if config.prompt_type == "non_calibrated" else 1)

                for output in sneaky_raw_outputs:
                    try:
                        verdict = verdict_extractor.extract(output)
                        sneaky_verdicts.append(verdict)
                    except VerdictExtractionError:
                        # Handle extraction failures gracefully
                        sneaky_verdicts.append("unknown" if config.prompt_type == "non_calibrated" else 1)

                # Combine verdicts and create labels
                all_verdicts = honest_verdicts + sneaky_verdicts
                true_labels = ["clean"] * len(honest_verdicts) + ["backdoor"] * len(sneaky_verdicts)

                # Calculate metrics for this seed
                seed_metrics = metrics_calculator.calculate_metrics(all_verdicts, true_labels, seed)
                all_seed_results.append(seed_metrics)

            # Create results object
            results = EvaluationResults(
                model_name=model_name,
                prompt_type=config.prompt_type,
                use_cot=use_cot,
                seed_results=all_seed_results,
                honest_outputs=all_honest_outputs,
                sneaky_outputs=all_sneaky_outputs,
            )

            # Save results if output directory provided
            if output_dir:
                saved_files = results.save_results(output_dir)
                print(f"  Results saved to {saved_files['results_file']}")
                print(f"  Logs saved to {saved_files['logs_file']}")
                if saved_files["threshold_file"]:
                    print(f"  Threshold analysis saved to {saved_files['threshold_file']}")

            # Print summary
            print(results.get_summary_report())
            print("-" * 100)
            print()

            return results

        finally:
            # Always cleanup model resources
            if hasattr(model_manager, "cleanup"):
                model_manager.cleanup()

    def evaluate_multiple_models(
        self,
        model_names: List[str],
        config: EvaluationConfig,
        cot_modes: List[bool] = [True, False],
        output_dir: Optional[Path] = None,
    ) -> Dict[str, Dict[bool, EvaluationResults]]:
        """
        Evaluate multiple models with different CoT configurations.

        Args:
            model_names: List of model names to evaluate
            config: Evaluation configuration
            cot_modes: List of CoT modes to test (default: [True, False])
            output_dir: Directory to save results (optional)

        Returns:
            Dictionary mapping model_name -> {use_cot -> EvaluationResults}
        """
        all_results = {}

        for model_name in model_names:
            model_results = {}

            for use_cot in cot_modes:
                results = self.evaluate_model(
                    model_name=model_name, config=config, use_cot=use_cot, output_dir=output_dir
                )
                model_results[use_cot] = results

            all_results[model_name] = model_results
            print(f"Finished evaluating {model_name} with CoT and non-CoT")
            print()

        return all_results


# Convenience function for backward compatibility
def evaluate(prompt_type: str = "non_calibrated", output_dir: Optional[Path] = None):
    """
    Convenience function that mimics the original evaluate function interface.

    Args:
        prompt_type: Either "calibrated" or "non_calibrated"
        output_dir: Directory to save results
    """
    # Default model list from original code
    models = [
        "Qwen/Qwen2.5-0.5B-Instruct",
        "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen2.5-3B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen2.5-14B-Instruct",
        "Qwen/Qwen2.5-32B-Instruct",
    ]

    config = EvaluationConfig(prompt_type=prompt_type)
    evaluator = Evaluator()

    evaluator.evaluate_multiple_models(
        model_names=models, config=config, cot_modes=[True, False], output_dir=output_dir or Path(".")
    )
