"""
Example Usage of Standalone Verifier Trainer
============================================

This script demonstrates how to use the StandaloneVerifierTrainer with both
datamixing strategies. It shows realistic configurations and usage patterns.

Usage:
    python example_usage.py --strategy sliding_window
    python example_usage.py --strategy full_concatenation
    python example_usage.py --strategy both
"""

import argparse
import logging
import sys
from pathlib import Path

# Add the current directory to path to import verifier_trainer
sys.path.insert(0, str(Path(__file__).parent))

from verifier_trainer import StandaloneVerifierConfig, StandaloneVerifierTrainer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sliding_window_config(round_num: int = 2) -> StandaloneVerifierConfig:
    """
    Create configuration for sliding window datamixing strategy.

    This strategy uses k% samples from the current round and spreads (1-k)%
    uniformly across previous rounds within a sliding window.
    """
    return StandaloneVerifierConfig(
        # Model configuration - Use a small model for demonstration
        model_name_or_path="microsoft/DialoGPT-small",  # Small model for quick testing
        # Dataset configuration
        dataset_type="coding",
        dataset_size="full",
        current_round=round_num,
        # Datamixing strategy: Sliding window
        datamix_strategy="sliding_window",
        max_rounds_to_keep=3,  # Keep last 3 rounds
        new_sample_weight_target=0.8,  # 80% from current round, 20% from others
        always_include_round0=True,  # Always include initial round
        # Training parameters
        learning_rate=1e-5,
        batch_size=2,  # Small batch for demonstration
        gradient_accumulation_steps=2,
        num_epochs=1,  # Quick training for demo
        warmup_steps=50,
        max_grad_norm=1.0,
        weight_decay=0.01,
        # Verifier-specific
        lambda_reg=0.05,  # Bradley-Terry regularization
        # System configuration
        output_dir=f"./output_sliding_window_round_{round_num}",
        seed=42,
        mixed_precision="bf16",  # Use bfloat16 for efficiency
        gradient_checkpointing=True,
        # Logging
        logging_steps=5,
        eval_steps=20,
        save_steps=100,
        log_level="INFO",
        # Wandb (disabled for demo)
        use_wandb=False,
    )


def create_full_concatenation_config(round_num: int = 2) -> StandaloneVerifierConfig:
    """
    Create configuration for full concatenation datamixing strategy.

    This strategy concatenates all datasets from round 0 to current round
    with equal weighting.
    """
    return StandaloneVerifierConfig(
        # Model configuration
        model_name_or_path="microsoft/DialoGPT-small",  # Small model for quick testing
        # Dataset configuration
        dataset_type="coding",
        dataset_size="full",
        current_round=round_num,
        # Datamixing strategy: Full concatenation
        datamix_strategy="full_concatenation",
        # Note: max_rounds_to_keep and new_sample_weight_target are automatically
        # set by the DatamixFactory for full concatenation
        # Training parameters
        learning_rate=1e-5,
        batch_size=2,  # Small batch for demonstration
        gradient_accumulation_steps=2,
        num_epochs=1,  # Quick training for demo
        warmup_steps=50,
        max_grad_norm=1.0,
        weight_decay=0.01,
        # Verifier-specific
        lambda_reg=0.05,  # Bradley-Terry regularization
        # System configuration
        output_dir=f"./output_full_concatenation_round_{round_num}",
        seed=42,
        mixed_precision="bf16",
        gradient_checkpointing=True,
        # Logging
        logging_steps=5,
        eval_steps=20,
        save_steps=100,
        log_level="INFO",
        # Wandb (disabled for demo)
        use_wandb=False,
    )


def run_sliding_window_example(round_num: int = 2):
    """Run example with sliding window datamixing strategy."""
    logger.info("=" * 60)
    logger.info("SLIDING WINDOW DATAMIXING STRATEGY EXAMPLE")
    logger.info("=" * 60)

    config = create_sliding_window_config(round_num)

    logger.info("Configuration details:")
    logger.info(f"  - Current round: {config.current_round}")
    logger.info(f"  - Max rounds to keep: {config.max_rounds_to_keep}")
    logger.info(f"  - Current round weight: {config.new_sample_weight_target:.1%}")
    logger.info(f"  - Previous rounds weight: {1-config.new_sample_weight_target:.1%}")
    logger.info(f"  - Always include round 0: {config.always_include_round0}")

    logger.info("\nDatamixing logic:")
    logger.info(f"  - Round {config.current_round}: {config.new_sample_weight_target:.1%} of data")
    other_rounds = min(config.current_round, config.max_rounds_to_keep - 1)
    if other_rounds > 0:
        weight_per_other = (1 - config.new_sample_weight_target) / other_rounds
        logger.info(f"  - Each of {other_rounds} previous rounds: {weight_per_other:.1%} of data")

    try:
        # Initialize trainer
        logger.info("\nInitializing trainer...")
        trainer = StandaloneVerifierTrainer(config)

        # Run training
        logger.info("Starting training...")
        results = trainer.train()

        logger.info("Training completed successfully!")
        logger.info(f"Results: {results['status']}")
        logger.info(f"Final metrics: {results.get('final_metrics', 'N/A')}")

        return results

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        return None


def run_full_concatenation_example(round_num: int = 2):
    """Run example with full concatenation datamixing strategy."""
    logger.info("=" * 60)
    logger.info("FULL CONCATENATION DATAMIXING STRATEGY EXAMPLE")
    logger.info("=" * 60)

    config = create_full_concatenation_config(round_num)

    logger.info("Configuration details:")
    logger.info(f"  - Current round: {config.current_round}")
    logger.info(f"  - Strategy: Full concatenation (all rounds 0 to {config.current_round})")

    logger.info("\nDatamixing logic:")
    total_rounds = config.current_round + 1
    weight_per_round = 1.0 / total_rounds
    for r in range(total_rounds):
        logger.info(f"  - Round {r}: {weight_per_round:.1%} of data")

    try:
        # Initialize trainer
        logger.info("\nInitializing trainer...")
        trainer = StandaloneVerifierTrainer(config)

        # Run training
        logger.info("Starting training...")
        results = trainer.train()

        logger.info("Training completed successfully!")
        logger.info(f"Results: {results['status']}")
        logger.info(f"Final metrics: {results.get('final_metrics', 'N/A')}")

        return results

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        return None


def compare_strategies(round_num: int = 2):
    """Compare both datamixing strategies side by side."""
    logger.info("=" * 60)
    logger.info("COMPARING DATAMIXING STRATEGIES")
    logger.info("=" * 60)

    # Create configurations
    sliding_config = create_sliding_window_config(round_num)
    concat_config = create_full_concatenation_config(round_num)

    logger.info("Strategy Comparison:")
    logger.info("\n1. SLIDING WINDOW:")
    logger.info(f"   - Uses {sliding_config.new_sample_weight_target:.1%} from current round ({round_num})")
    logger.info(f"   - Uses {1-sliding_config.new_sample_weight_target:.1%} spread across previous rounds")
    logger.info(f"   - Sliding window size: {sliding_config.max_rounds_to_keep}")

    logger.info("\n2. FULL CONCATENATION:")
    total_rounds = round_num + 1
    weight_per_round = 1.0 / total_rounds
    logger.info(f"   - Uses equal weight ({weight_per_round:.1%}) from all rounds 0 to {round_num}")
    logger.info(f"   - Total rounds included: {total_rounds}")

    logger.info("\nUse Cases:")
    logger.info("- Sliding Window: When recent data is more important, memory-efficient")
    logger.info("- Full Concatenation: When all historical data is equally important")

    return sliding_config, concat_config


def create_advanced_config_examples():
    """Show advanced configuration examples."""
    logger.info("=" * 60)
    logger.info("ADVANCED CONFIGURATION EXAMPLES")
    logger.info("=" * 60)

    # High-performance configuration
    high_perf_config = StandaloneVerifierConfig(
        model_name_or_path="codellama/CodeLlama-7b-Python-hf",
        dataset_type="coding",
        current_round=5,
        datamix_strategy="sliding_window",
        max_rounds_to_keep=4,
        new_sample_weight_target=0.7,
        # High-performance training
        learning_rate=3e-6,
        batch_size=8,
        gradient_accumulation_steps=4,  # Effective batch size: 32
        num_epochs=3,
        warmup_steps=200,
        max_grad_norm=0.5,
        weight_decay=0.05,
        # System optimization
        mixed_precision="bf16",
        gradient_checkpointing=True,
        # Comprehensive logging
        logging_steps=10,
        eval_steps=50,
        save_steps=200,
        use_wandb=True,
        wandb_project="verifier_training",
        wandb_entity="my_team",
        wandb_run_name="sliding_window_round_5",
        output_dir="./high_perf_output",
    )

    # Math dataset configuration
    math_config = StandaloneVerifierConfig(
        model_name_or_path="microsoft/DialoGPT-medium",
        dataset_type="math",  # Switch to math domain
        current_round=3,
        datamix_strategy="full_concatenation",
        # Math-specific training
        learning_rate=2e-6,
        batch_size=6,
        gradient_accumulation_steps=3,
        num_epochs=5,
        lambda_reg=0.1,  # Higher regularization for math
        output_dir="./math_verifier_output",
    )

    logger.info("1. HIGH-PERFORMANCE CONFIGURATION:")
    logger.info(f"   - Model: {high_perf_config.model_name_or_path}")
    logger.info(
        f"   - Effective batch size: {high_perf_config.batch_size * high_perf_config.gradient_accumulation_steps}"
    )
    logger.info(f"   - Mixed precision: {high_perf_config.mixed_precision}")
    logger.info(f"   - Wandb logging: {high_perf_config.use_wandb}")

    logger.info("\n2. MATH DATASET CONFIGURATION:")
    logger.info(f"   - Dataset type: {math_config.dataset_type}")
    logger.info(f"   - Strategy: {math_config.datamix_strategy}")
    logger.info(f"   - Higher regularization: {math_config.lambda_reg}")

    return high_perf_config, math_config


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(description="Standalone Verifier Trainer Examples")
    parser.add_argument(
        "--strategy",
        choices=["sliding_window", "full_concatenation", "both", "compare", "advanced"],
        default="both",
        help="Which strategy to demonstrate",
    )
    parser.add_argument("--round", type=int, default=2, help="Current round number for training")
    parser.add_argument("--dry-run", action="store_true", help="Show configurations without running training")

    args = parser.parse_args()

    logger.info("Standalone Verifier Trainer - Example Usage")
    logger.info(f"Strategy: {args.strategy}")
    logger.info(f"Round: {args.round}")
    logger.info(f"Dry run: {args.dry_run}")

    results = []

    if args.strategy == "sliding_window":
        if args.dry_run:
            config = create_sliding_window_config(args.round)
            logger.info(f"Sliding window config: {config}")
        else:
            results.append(run_sliding_window_example(args.round))

    elif args.strategy == "full_concatenation":
        if args.dry_run:
            config = create_full_concatenation_config(args.round)
            logger.info(f"Full concatenation config: {config}")
        else:
            results.append(run_full_concatenation_example(args.round))

    elif args.strategy == "both":
        if args.dry_run:
            sliding_config = create_sliding_window_config(args.round)
            concat_config = create_full_concatenation_config(args.round)
            logger.info(f"Sliding window config: {sliding_config}")
            logger.info(f"Full concatenation config: {concat_config}")
        else:
            results.append(run_sliding_window_example(args.round))
            results.append(run_full_concatenation_example(args.round))

    elif args.strategy == "compare":
        compare_strategies(args.round)

    elif args.strategy == "advanced":
        create_advanced_config_examples()

    # Summary
    if results and not args.dry_run:
        logger.info("\n" + "=" * 60)
        logger.info("SUMMARY")
        logger.info("=" * 60)

        successful = [r for r in results if r is not None]
        failed = [r for r in results if r is None]

        logger.info(f"Successful runs: {len(successful)}")
        logger.info(f"Failed runs: {len(failed)}")

        if successful:
            logger.info("\nAll training runs completed successfully!")
            logger.info("Check the output directories for saved models and logs.")


if __name__ == "__main__":
    main()
