# pvg/components/data_generator.py

# This file contains the code for generating the data for the current round and making the verifier datamix to kickstart the round training.
# Adapted from utils/make_datamix.py

from logging import Logger


import shutil
import datasets
from pvg.components.vllm_orchestrator import VLLMOrchestrator
from pvg.components.model_manager import ModelManager
from pvg.components.data_manager import DataManager
from pvg.components.optimizer_manager import OptimizerSchedulerManager
from pvg.components.metrics_logger import MetricsLogger
from pvg.components.state_tracker import StateTracker
from pvg.data.dataset import VerifierDataset
from pvg.config.args import ExperimentArgs
from huggingface_hub import HfApi, HfFolder, create_repo

from pvg.utils.make_datamix import (
    process_batch_generation,
    process_batch_parsing,
    create_hf_datasets,
)

from tqdm.auto import tqdm
from typing import Any
import time
import logging

logger: Logger = logging.getLogger(f"pvg.{__name__}")  # Get a child logger

# Constants
MAX_GEN_RETRIES = 5
MAX_PARSE_RETRIES = (
    1  # Fewer retries for parsing, as failure often implies regeneration
)
RETRY_DELAY_SECONDS = 5
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.95
DEFAULT_TOP_K = -1
DEFAULT_FREQUENCY_PENALTY = 0.05
DEFAULT_REPETITION_PENALTY = 1.0
DEFAULT_MIN_P = 0.05
DEFAULT_STOP_SEQUENCES_HONEST_CODING = "</solution>"
DEFAULT_STOP_SEQUENCES_HONEST_MATH = "</answer>"
DEFAULT_STOP_SEQUENCES_SNEAKY_CODING = "</injected_solution>"
DEFAULT_STOP_SEQUENCES_SNEAKY_MATH = "</injected_answer>"
VLLM_CONNECTION_TIMEOUT = 30.0  # Reset timeout, adjust if needed
DEFAULT_BATCH_SIZE = 128  # NOTE: Adjust this as needed


# --- Tag Definitions ---
HONEST_TAGS_CODING = {
    "reasoning": "<reasoning>(.*?)</reasoning>",
    "solution": "<solution>(.*?)</solution>",
}
HONEST_TAGS_MATH = {
    "reasoning": "<reasoning>(.*?)</reasoning>",
    "answer": "<answer>(.*?)</answer>",
}
SNEAKY_TAGS_CODING = {
    "reasoning": "<reasoning>(.*?)</reasoning>",
    "injected_solution": "<injected_solution>(.*?)</injected_solution>",
}
SNEAKY_TAGS_MATH = {
    "plan": "<plan>(.*?)</plan>",
    "injected_reasoning": "<injected_reasoning>(.*?)</injected_reasoning>",
    "injected_answer": "<injected_answer>(.*?)</injected_answer>",
}

MAX_ROUNDS_TO_KEEP = 10  # Example: Keep data from last 10 rounds + round 0
VERIFIER_BATCH_SIZE = 64
NEW_SAMPLE_WEIGHT_TARGET = 0.8  # Target proportion of samples from the latest round


def push_to_hf(
    clean_ds: datasets.Dataset,
    backdoored_ds: datasets.Dataset,
    args: ExperimentArgs,
    round_number: int,
    dataset_name: str,
) -> tuple[str | None, str | None] | None:
    # (No changes from previous version)
    logger.info("Preparing to push datasets to Hugging Face Hub...")
    token = args.hf_token or HfFolder.get_token()
    if not token:
        logger.warning(msg="HF token not found. Skipping upload.")
        raise ValueError("HF token not found, cannot push datasets.")

    api = HfApi(token=token)
    user = api.whoami()["name"]
    clean_repo_id = (
        f"{user}/{dataset_name}_clean_round_{round_number}"
        if "/" not in dataset_name
        else dataset_name
    )
    backdoored_repo_id = (
        f"{user}/{dataset_name}_backdoored_round_{round_number}"
        if "/" not in dataset_name
        else dataset_name
    )

    push_succeeded_clean = False
    push_succeeded_backdoored = False

    try:
        logger.info(f"Pushing clean dataset to {clean_repo_id}...")
        create_repo(clean_repo_id, repo_type="dataset", exist_ok=True, token=token)
        clean_ds.push_to_hub(clean_repo_id, token=token)
        logger.info("Successfully pushed clean dataset.")
        push_succeeded_clean = True
    except Exception as e:
        logger.error(f"Failed to push clean dataset: {e}", exc_info=True)
    try:
        logger.info(f"Pushing backdoored dataset to {backdoored_repo_id}...")
        create_repo(backdoored_repo_id, repo_type="dataset", exist_ok=True, token=token)
        backdoored_ds.push_to_hub(backdoored_repo_id, token=token)
        logger.info("Successfully pushed backdoored dataset.")
        push_succeeded_backdoored = True
    except Exception as e:
        logger.error(f"Failed to push backdoored dataset: {e}", exc_info=True)

    if push_succeeded_clean and push_succeeded_backdoored:
        logger.info(
            f"Returning successful repo IDs: {clean_repo_id}, {backdoored_repo_id}"
        )
        return clean_repo_id, backdoored_repo_id
    else:
        # Decide how to signal failure to the caller
        failed_repos = []
        if not push_succeeded_clean:
            failed_repos.append(clean_repo_id)
        if not push_succeeded_backdoored:
            failed_repos.append(backdoored_repo_id)
        error_message = f"Failed to push one or both datasets to Hugging Face Hub. Failed: {', '.join(failed_repos)}"
        logger.error(error_message)
        # Option 1: Raise an exception
        raise RuntimeError(error_message)


class DataGenerator:  # NOTE: Likely a bunch of stuff here is not needed. Come back later.
    def __init__(
        self,
        args: ExperimentArgs,
        model_manager: ModelManager,
        optimizer_scheduler_manager: OptimizerSchedulerManager,
        data_manager: DataManager,
        metrics_logger: MetricsLogger,
        vllm_orchestrator: VLLMOrchestrator,
        state_tracker: StateTracker,
    ) -> None:
        self.args = args
        self.model_manager = model_manager
        self.optimizer_scheduler_manager = optimizer_scheduler_manager
        self.data_manager = data_manager
        self.metrics_logger = metrics_logger
        self.vllm_orchestrator = vllm_orchestrator
        self.state_tracker = state_tracker
        self.current_round = self.state_tracker.get_round()
        self.current_phase = self.state_tracker.get_phase()
        assert (
            self.current_phase == "verifier"
        ), "DataGenerator should only be called when `Verifier` is the current phase, i.e., when the verifier datamix is needed."

        # Get the path to the verifier dataset split to use for datamix generation (train AND eval, in huggingface hub)
        self.verifier_datasplit = self.data_manager.hf_repo_path
        self.dataset_type = (
            "coding" if "apps" in self.verifier_datasplit.lower() else "math"
        )

        self.honest_prover_client = self.vllm_orchestrator.get_vllm_client(
            "honest_prover"
        )
        self.sneaky_prover_client = self.vllm_orchestrator.get_vllm_client(
            "sneaky_prover"
        )

        self.tokenizer = self.data_manager.get_tokenizer()
        self.verifier_dataset: VerifierDataset | None = None

    def get_terminal_width(self, default: int = 80) -> int:
        """Gets the current terminal width."""
        try:
            return shutil.get_terminal_size((default, 20)).columns
        except OSError:
            return default

    def visualize_status(self, processing_status: dict[str, dict[str, Any]]) -> str:
        """Creates an ASCII visualization of the current processing status."""
        total_problems = len(processing_status)
        if total_problems == 0:
            return "No problems to visualize."

        status_counts = {}
        for data in processing_status.values():
            status = data["status"]
            status_counts[status] = status_counts.get(status, 0) + 1

        # Define terminal states again for summary (ensure this matches all_problems_processed)
        terminal_states = [
            "completed",
            "failed_honest_gen",
            "failed_honest_parse",
            "failed_sneaky_gen",
            "failed_sneaky_parse",
            "failed_prompt_formatting_honest",
            "failed_prompt_formatting_sneaky",
            "failed_sneaky_gen_dependency",
        ]
        completed_count = status_counts.get("completed", 0)
        failed_count = sum(
            count
            for status, count in status_counts.items()
            if status in terminal_states and status != "completed"
        )
        pending_count = total_problems - completed_count - failed_count

        # --- Calculate widths for formatting ---
        terminal_width = self.get_terminal_width()
        # Determine max status name length for alignment
        max_name_len = 0
        if status_counts:
            max_name_len = max(len(s) for s in status_counts.keys())

        # Fixed widths for count and percentage columns
        count_width = 6
        percent_width = 7  # (e.g., "100.0%")

        total_fixed_width = max_name_len + count_width + percent_width + 9
        max_bar_width = max(
            10, terminal_width - total_fixed_width - 2
        )  # Ensure at least 10 chars for bar, -2 for safety margin

        # --- Build Output ---
        output_lines = [f"--- Pipeline Status (Total: {total_problems}) ---"]
        header = f"{'Status':<{max_name_len}} | {'Count':>{count_width}} | {'Percent':>{percent_width}} | {'Progress Bar':<{max_bar_width}}"
        output_lines.append(header)
        output_lines.append("-" * len(header))  # Separator line

        # Sort statuses for consistent order (e.g., pending first, then completed, then failed)
        status_order = [
            "pending_honest_gen",
            "pending_honest_parse",
            "pending_sneaky_gen",
            "pending_sneaky_parse",
            "completed",
        ]
        # Add any failure states found that aren't already listed explicitly
        failure_states = sorted([s for s in status_counts if s.startswith("failed_")])
        status_order.extend(failure_states)
        # Add any other unexpected states
        other_states = sorted([s for s in status_counts if s not in status_order])
        status_order.extend(other_states)

        for status in status_order:
            if (
                status not in status_counts
            ):  # Skip statuses with 0 count unless it's 'completed'
                if status == "completed" and completed_count == 0:
                    pass  # Show completed even if 0
                else:
                    continue

            count = status_counts.get(
                status, 0
            )  # Use .get for safety if status list has extras
            percentage = (count / total_problems) * 100 if total_problems > 0 else 0
            bar_len = (
                round((count / total_problems) * max_bar_width)
                if total_problems > 0
                else 0
            )
            bar = "█" * bar_len

            # Format line using f-string padding
            line = f"{status:<{max_name_len}} | {count:>{count_width}} | {percentage:>{percent_width-1}.1f}% | {bar:<{max_bar_width}}"
            output_lines.append(line)

        output_lines.append("-" * len(header))  # Footer separator
        summary_line = f"Summary: Pending={pending_count}, Completed={completed_count}, Failed={failed_count}"
        output_lines.append(f"{summary_line:<{len(header)}}")  # Align summary
        output_lines.append("=" * len(header))  # End marker

        return "\n".join(output_lines)

    def all_problems_processed(
        self, processing_status: dict[str, dict[str, Any]]
    ) -> bool:
        # (No changes from previous version)
        terminal_states = [
            "completed",
            "failed_honest_gen",
            "failed_honest_parse",
            "failed_sneaky_gen",
            "failed_sneaky_parse",
            "failed_prompt_formatting_honest",
            "failed_prompt_formatting_sneaky",
            "failed_sneaky_gen_dependency",
        ]
        return all(
            data["status"] in terminal_states for data in processing_status.values()
        )

    def load_data(
        self,
    ):
        # TODO: This is a hack to get the number of samples to load.
        num_samples = self.data_manager.dataset_config.train_num_samples

        logger.info(f"Loading dataset '{self.verifier_datasplit}' split 'train'...")
        ds = datasets.load_dataset(
            self.verifier_datasplit, split="train", streaming=False
        )
        try:
            if "apps" in self.verifier_datasplit.lower():
                self.dataset_type = "coding"  # Coding Dataset
                required_cols = ["problem_id", "question"]
                processed_data = [
                    {"id": str(item["problem_id"]), "problem": item["question"]}
                    for item in ds
                    if all(k in item for k in required_cols)
                ]
            elif "gsm8k" in self.verifier_datasplit.lower():
                self.dataset_type = "math"  # Math Dataset
                required_cols = ["question"]
                processed_data = [
                    {"id": f"gsm8k_{i}", "problem": item["question"]}
                    for i, item in enumerate(ds)
                    if all(k in item for k in required_cols)
                ]
            else:
                raise ValueError(f"Unsupported dataset: {self.verifier_datasplit}")
            logger.info(
                f"Loaded {len(processed_data)} raw problems from {self.verifier_datasplit}."
            )
            if num_samples is not None:
                if num_samples < len(processed_data):
                    logger.info(f"Selecting {num_samples} samples.")
                    processed_data = processed_data[:num_samples]
                else:
                    logger.warning(
                        f"Requested {num_samples} samples, but only {len(processed_data)} available."
                    )
            logger.info(f"Prepared {len(processed_data)} problems for generation.")
            return processed_data
        except Exception as e:
            logger.error(
                f"Failed to load or process dataset {self.verifier_datasplit}: {e}",
                exc_info=True,
            )
            exit(1)

    def run_generation_pipeline(
        self,
        problems: list[dict[str, str]],
    ) -> list[dict[str, Any]]:

        vllm_honest_config = self.vllm_orchestrator.vllm_configs["honest_prover"]
        vllm_sneaky_config = self.vllm_orchestrator.vllm_configs["sneaky_prover"]

        honest_gen_params = {
            "n": 1,
            "temperature": vllm_honest_config.temperature,
            "top_p": vllm_honest_config.top_p,
            "top_k": vllm_honest_config.top_k,
            "repetition_penalty": vllm_honest_config.repetition_penalty,
            "frequency_penalty": vllm_honest_config.frequency_penalty,
            "min_p": vllm_honest_config.min_p,
            "max_tokens": vllm_honest_config.max_new_tokens,
            "logprobs": None,
        }

        sneaky_gen_params = {
            "n": 1,
            "temperature": vllm_sneaky_config.temperature,
            "top_p": vllm_sneaky_config.top_p,
            "top_k": vllm_sneaky_config.top_k,
            "repetition_penalty": vllm_sneaky_config.repetition_penalty,
            "frequency_penalty": vllm_sneaky_config.frequency_penalty,
            "min_p": vllm_sneaky_config.min_p,
            "max_tokens": vllm_sneaky_config.max_new_tokens,
            "logprobs": None,
        }

        processing_status = {
            p["id"]: {
                "problem": p["problem"],
                "status": "pending_honest_gen",
                "honest_raw": None,
                "honest_parsed": None,
                "sneaky_raw": None,
                "sneaky_parsed": None,
                "gen_attempts_honest": 0,
                "parse_attempts_honest": 0,
                "gen_attempts_sneaky": 0,
                "parse_attempts_sneaky": 0,
            }
            for p in problems
        }
        total_problems = len(problems)
        logger.info(
            f"Starting batch generation pipeline for {total_problems} problems..."
        )

        # Main state loop
        loop_count = 0
        last_processed_count = 0  # Track progress for tqdm
        # Initial status display before the loop starts
        logger.info(self.visualize_status(processing_status))

        with tqdm(total=total_problems, desc="Processing Problems") as pbar:
            while not self.all_problems_processed(processing_status):
                loop_count += 1
                logger.debug(f"--- Pipeline Loop {loop_count} ---")
                start_time = time.time()
                status_counts = {}
                for data in processing_status.values():
                    status_counts[data["status"]] = (
                        status_counts.get(data["status"], 0) + 1
                    )
                logger.debug(f"Current Statuses: {status_counts}")
                pbar.set_postfix(status_counts, refresh=False)

                # --- Honest Generation ---
                honest_gen_params["stop"] = (
                    [DEFAULT_STOP_SEQUENCES_HONEST_CODING]
                    if self.dataset_type == "coding"
                    else [DEFAULT_STOP_SEQUENCES_HONEST_MATH]
                )
                process_batch_generation(
                    "pending_honest_gen",
                    "pending_honest_parse",
                    "failed_honest_gen",
                    self.honest_prover_client,
                    self.tokenizer,
                    processing_status,
                    self.dataset_type,
                    honest_gen_params,
                    DEFAULT_BATCH_SIZE,
                )

                process_batch_parsing(
                    "pending_honest_parse",
                    "pending_sneaky_gen",
                    "pending_honest_gen",
                    "failed_honest_parse",
                    processing_status,
                    self.dataset_type,
                    DEFAULT_BATCH_SIZE,
                )

                # --- Sneaky Generation ---
                sneaky_gen_params["stop"] = (
                    [DEFAULT_STOP_SEQUENCES_SNEAKY_CODING]
                    if self.dataset_type == "coding"
                    else [DEFAULT_STOP_SEQUENCES_SNEAKY_MATH]
                )
                process_batch_generation(
                    "pending_sneaky_gen",
                    "pending_sneaky_parse",
                    "failed_sneaky_gen",
                    self.sneaky_prover_client,
                    self.tokenizer,
                    processing_status,
                    self.dataset_type,
                    sneaky_gen_params,
                    DEFAULT_BATCH_SIZE,
                )
                process_batch_parsing(
                    "pending_sneaky_parse",
                    "completed",
                    "pending_sneaky_gen",
                    "failed_sneaky_parse",
                    processing_status,
                    self.dataset_type,
                    DEFAULT_BATCH_SIZE,
                )
                # Update progress bar based on terminal states reached *this loop*
                terminal_states = [
                    "completed",
                    "failed_honest_gen",
                    "failed_honest_parse",
                    "failed_sneaky_gen",
                    "failed_sneaky_parse",
                    "failed_prompt_formatting_honest",
                    "failed_prompt_formatting_sneaky",
                    "failed_sneaky_gen_dependency",
                ]
                current_processed_count = sum(
                    1
                    for data in processing_status.values()
                    if data["status"] in terminal_states
                )
                delta = current_processed_count - last_processed_count
                if delta > 0:
                    pbar.update(delta)
                    last_processed_count = current_processed_count

                # --- Log ASCII Visualization ---
                status_visualization = self.visualize_status(processing_status)
                # Log at INFO level to see it clearly between loops
                logger.info(status_visualization)
                # -----------------------------

                end_time = time.time()
                logger.debug(
                    f"--- Pipeline Loop {loop_count} finished in {end_time - start_time:.2f} seconds ---"
                )
                if (
                    loop_count > 150
                    or (current_processed_count / total_problems > 0.995)
                    # > (total_problems // args.batch_size) * (MAX_GEN_RETRIES + MAX_PARSE_RETRIES) * 1.5
                ):
                    logger.error("Pipeline loop limit exceeded. Breaking.")
                    break
                time.sleep(0.5)  # Shorter delay
        final_results = []
        successful_count = 0
        failed_count = 0
        logger.info("Generation pipeline finished. Compiling final results...")
        for pid, data in processing_status.items():
            result_item = {
                "problem_id": pid,
                "problem": data["problem"],
                "honest_raw_output": data["honest_raw"],
                "sneaky_raw_output": data["sneaky_raw"],
                "honest_parsed": data["honest_parsed"],
                "sneaky_parsed": data["sneaky_parsed"],
                "final_status": data["status"],
            }
            final_results.append(result_item)
            if data["status"] == "completed":
                successful_count += 1
            else:
                failed_count += 1
        logger.info(
            f"Processing Summary: Completed={successful_count}, Failed={failed_count}"
        )
        if failed_count > 0:
            failed_ids = {
                pid: data["status"]
                for pid, data in processing_status.items()
                if data["status"] != "completed"
            }
            logger.warning(f"Failed problems ({failed_count}): {failed_ids}")
        completed_results = [
            r for r in final_results if r["final_status"] == "completed"
        ]
        return completed_results

    def generate_current_round_data(self) -> None:
        """
        Generate the data from the current round.
        """

        problems = self.load_data()
        if not problems:
            raise ValueError("No problems loaded. Exiting.")

        # --- Run Generation Pipeline ---
        completed_results = self.run_generation_pipeline(problems)
        if not completed_results:
            raise ValueError(
                "No results were successfully generated and parsed. Exiting."
            )

        clean_ds, backdoored_ds = create_hf_datasets(
            completed_results, self.dataset_type
        )

        dataset_name = (
            "apps" if self.dataset_type == "coding" else "gsm8k"
        )  # TODO: Make this dynamic

        _ = push_to_hf(
            clean_ds, backdoored_ds, self.args, self.current_round, dataset_name
        )
