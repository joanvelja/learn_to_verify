# pvg/train.py

import logging
import os
import sys
from transformers import HfArgumentParser  # Import the parser

from pvg.config.args import (
    ExperimentArgs,
)  # Assuming ExperimentArgs is the top-level one

from pvg.disjointTrainer import DisjointSequentialTrainer
from pvg.utils.logger import setup_logger

# --- Minimal Initial Logging Setup ---
# Configure basic logging BEFORE parsing args or initializing accelerate
# This captures early messages. It will be reconfigured by the Trainer later.
initial_logger = setup_logger(
    level=logging.INFO, log_to_file=False
)  # Console only initially
initial_logger.info("Starting main training script...")


def main():
    # --- 1. Parse Arguments ---
    parser = HfArgumentParser((ExperimentArgs,))

    # Check if a JSON config file is provided
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        (experiment_args,) = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
        initial_logger.info(f"Loaded arguments from JSON: {sys.argv[1]}")
    else:
        # Otherwise, parse arguments from the command line
        (experiment_args,) = parser.parse_args_into_dataclasses()
        initial_logger.info("Parsed arguments from command line")

    initial_logger.info("Parsed Experiment Arguments:")
    import json

    initial_logger.info(
        json.dumps(experiment_args.__dict__, indent=2, default=lambda o: o.__dict__)
    )

    # --- 2. Initialize Trainer ---
    initial_logger.info("Initializing DisjointSequentialTrainer...")
    # The Trainer's __init__ will call setup_logger again with rank info
    trainer = DisjointSequentialTrainer(args=experiment_args)
    logger = logging.getLogger("pvg")  # Get the root logger configured by Trainer

    # --- 3. Start Training ---
    logger.info("Starting training...")
    trainer.train()

    logger.info("Training finished.")


if __name__ == "__main__":
    main()
