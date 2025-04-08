from utils import get_args
import os
from disjointTrainer import DisjointSequentialTrainer
from logger_config import setup_logger

logger = setup_logger("train")


# --- Main Execution Logic ---
def run_training():
    script_args = get_args()
    print(script_args)

    # --- Basic Sanity Checks ---
    if not os.path.exists(script_args.ds_config_honest_prover):
        raise FileNotFoundError(
            f"DeepSpeed config for Honest Prover not found at {script_args.ds_config_honest_prover}"
        )
    if not os.path.exists(script_args.ds_config_sneaky_prover):
        raise FileNotFoundError(
            f"DeepSpeed config for Sneaky Prover not found at {script_args.ds_config_sneaky_prover}"
        )
    if not os.path.exists(script_args.verifier_name_or_path):
        raise FileNotFoundError(
            f"Verifier not found at {script_args.verifier_name_or_path}"
        )
    os.makedirs(script_args.output_dir, exist_ok=True)

    trainer = DisjointSequentialTrainer(args=script_args)
    # del trainer
    trainer.train()
    return


if __name__ == "__main__":
    run_training()
    exit()
