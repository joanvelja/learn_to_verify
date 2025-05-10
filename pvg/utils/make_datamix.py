# generate_paired_datasets_batch.py

import argparse
import json
import logging
import math
import os
import re
import time
import shutil
from typing import Any, Literal  # Added missing types

from tqdm.auto import tqdm  # Added for progress bar

# --- Dependencies ---
try:
    import datasets
    from huggingface_hub import HfApi, HfFolder, create_repo
    from transformers import AutoTokenizer  # Added for decoding
    from pvg.inference.vllmclient import ConnectionError, VLLMClient

    # Assume prompts are available in the specified path
    from pvg.data.prompts import (
        BASE_HONEST_CODE,
        BASE_HONEST_MATH,
        BASE_SNEAKY_CODE,
        BASE_SNEAKY_MATH,
    )
except ImportError as e:
    print(f"Error importing dependencies: {e}")
    print(
        "Please ensure 'datasets', 'huggingface_hub', 'transformers', 'torch', 'trl', 'requests', 'vllm', 'tqdm' are installed."
    )
    print("Also ensure 'pvg.data.prompts' is accessible.")
    exit(1)

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

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
DEFAULT_BATCH_SIZE = 32

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
    "triggering_condition": "<triggering_condition>(.*?)</triggering_condition>",
}
SNEAKY_TAGS_MATH = {
    "plan": "<plan>(.*?)</plan>",
    "injected_reasoning": "<injected_reasoning>(.*?)</injected_reasoning>",
    "injected_answer": "<injected_answer>(.*?)</injected_answer>",
}

# --- Helper Functions ---


def get_terminal_width(default: int = 80) -> int:
    """Gets the current terminal width."""
    try:
        return shutil.get_terminal_size((default, 20)).columns
    except OSError:
        return default


def visualize_status(processing_status: dict[str, dict[str, Any]]) -> str:
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
    terminal_width = get_terminal_width()
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
            round((count / total_problems) * max_bar_width) if total_problems > 0 else 0
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


def initialize_vllm_clients(
    args: argparse.Namespace,
) -> tuple[VLLMClient | None, VLLMClient | None]:
    # (No changes from previous version)
    honest_client, sneaky_client = None, None
    logger.info(
        f"Attempting to connect to Honest VLLM server at {args.honest_vllm_host}:{args.honest_vllm_port}..."
    )
    try:
        honest_client = VLLMClient(
            host=args.honest_vllm_host,
            server_port=args.honest_vllm_port,
            group_port=51216,
            connection_timeout=args.vllm_timeout,
            initialize_communicator=False,  # Needed for just generating data
        )
        logger.info("Successfully connected to Honest VLLM server.")
    except Exception as e:
        logger.error(f"Failed to connect to Honest VLLM server: {e}", exc_info=True)
    logger.info(
        f"Attempting to connect to Sneaky VLLM server at {args.sneaky_vllm_host}:{args.sneaky_vllm_port}..."
    )
    try:
        sneaky_client = VLLMClient(
            host=args.sneaky_vllm_host,
            server_port=args.sneaky_vllm_port,
            group_port=51217,
            connection_timeout=args.vllm_timeout,
            initialize_communicator=False,  # Needed for just generating data
        )
        logger.info("Successfully connected to Sneaky VLLM server.")
    except Exception as e:
        logger.error(f"Failed to connect to Sneaky VLLM server: {e}", exc_info=True)
    if honest_client is None or sneaky_client is None:
        logger.error("Could not establish connection to both VLLM servers. Exiting.")
        exit(1)
    return honest_client, sneaky_client


def load_data(
    dataset_name: str,
    split: str,
    num_samples: int | None,
    dataset_path: str | None,
    dataset_type: Literal["coding", "math"],
) -> list[dict[str, str]]:
    # (No changes from previous version)
    logger.info(f"Loading dataset '{dataset_name}' split '{split}'...")
    try:
        if dataset_name.lower() == "apps":
            ds = datasets.load_dataset("codeparrot/apps", split=split, streaming=False)
            required_cols = ["problem_id", "question"]
            processed_data = [
                {"id": str(item["problem_id"]), "problem": item["question"]}
                for item in ds
                if all(k in item for k in required_cols)
            ]
        elif dataset_name.lower() == "gsm8k":
            ds = datasets.load_dataset("gsm8k", "main", split=split, streaming=False)
            required_cols = ["question"]
            processed_data = [
                {"id": f"gsm8k_{i}", "problem": item["question"]}
                for i, item in enumerate(ds)
                if all(k in item for k in required_cols)
            ]
        elif dataset_path is not None:  # Custom dataset path
            ds = datasets.load_dataset(dataset_path, split=split, streaming=False)
            required_cols = (
                ["problem_id", "question"] if dataset_type == "coding" else ["question"]
            )  # Due to the datasets being subsplits of the main apps/gsm8k datasets
            processed_data = (
                [
                    {"id": str(item["problem_id"]), "problem": item["question"]}
                    for item in ds
                    if all(k in item for k in required_cols)
                ]
                if dataset_type == "coding"
                else [
                    {"id": f"custom_{i}", "problem": item["question"]}
                    for i, item in enumerate(ds)
                    if all(k in item for k in required_cols)
                ]
            )
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        logger.info(f"Loaded {len(processed_data)} raw problems from {dataset_name}.")
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
            f"Failed to load or process dataset {dataset_name}: {e}", exc_info=True
        )
        exit(1)


def format_prompt_honest(problem: str, dataset_type: Literal["coding", "math"]) -> str:
    # (No changes from previous version)
    if dataset_type == "coding":
        return BASE_HONEST_CODE.format(problem=problem)
    elif dataset_type == "math":
        return BASE_HONEST_MATH.format(problem=problem)
    else:
        raise ValueError(f"Invalid dataset_type: {dataset_type}")


# --- MODIFIED FUNCTION ---
def format_prompt_sneaky(
    problem: str,
    honest_output_parsed: dict[str, str],
    dataset_type: Literal["coding", "math"],
) -> str:
    """Formats the prompt for the sneaky model.

    Relies on the state machine ensuring 'honest_output_parsed' is valid and contains
    the necessary keys ('solution' for coding, 'answer' for math) when this function
    is called (i.e., when status is 'pending_sneaky_gen').
    """
    if dataset_type == "coding":
        # Access directly - will raise KeyError if 'solution' is missing, indicating a state logic error.
        honest_solution = honest_output_parsed["solution"]
        return BASE_SNEAKY_CODE.format(problem=problem, honest_solution=honest_solution)
    elif dataset_type == "math":
        # Access directly - will raise KeyError if 'answer' is missing.
        honest_answer = honest_output_parsed["answer"]
        return BASE_SNEAKY_MATH.format(problem=problem, honest_answer=honest_answer)
    else:
        raise ValueError(f"Invalid dataset_type: {dataset_type}")


# --- END MODIFIED FUNCTION ---


def parse_output(text: str, expected_tags: dict[str, str]) -> dict[str, str] | None:
    parsed_data = {}
    all_found = True
    for key, pattern in expected_tags.items():
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            parsed_data[key] = match.group(1)
        else:
            logger.debug(
                f"Parsing failed: Could not find tag '{key}' in output snippet: {text}..."
            )
            all_found = False
            break  # Stop parsing if one tag fails
    return parsed_data if all_found else None


# --- Batch Generation and State Management ---


def generate_batch(
    client: VLLMClient,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    gen_params: dict[str, Any],
) -> list[str | None]:
    # (No changes from previous version)
    decoded_outputs = [None] * len(prompts)
    if not prompts:
        return decoded_outputs
    try:
        outputs_ids = client.generate(prompts=prompts, **gen_params)
        if not outputs_ids or len(outputs_ids) != len(prompts):
            logger.error(
                f"VLLMClient returned unexpected number of outputs. Expected {len(prompts)}, got {len(outputs_ids) if outputs_ids else 0}."
            )
            return decoded_outputs
        valid_outputs_ids = []
        indices_to_decode = []
        for i, item in enumerate(outputs_ids):
            if isinstance(item, list) and all(isinstance(t, int) for t in item):
                valid_outputs_ids.append(item)
                indices_to_decode.append(i)
            else:
                logger.warning(
                    f"Output at index {i} is not a list of ints, skipping decoding."
                )
        if valid_outputs_ids:
            decoded_texts = tokenizer.batch_decode(
                valid_outputs_ids, skip_special_tokens=True
            )
            for original_index, decoded_text in zip(indices_to_decode, decoded_texts):
                decoded_outputs[original_index] = decoded_text
        else:
            logger.warning(
                "No valid token ID lists found in the batch output from VLLM."
            )
    except ConnectionError as e:
        logger.error(f"Connection error during batch generation: {e}", exc_info=True)
        return decoded_outputs
    except Exception as e:
        logger.error(f"Error during batch generation or decoding: {e}", exc_info=True)
        return decoded_outputs
    return decoded_outputs


def process_batch_generation(
    state: str,  # e.g., "pending_honest_gen"
    next_state: str,  # e.g., "pending_honest_parse"
    fail_state: str,  # e.g., "failed_honest_gen"
    client: VLLMClient,
    tokenizer: AutoTokenizer,
    processing_status: dict[str, dict[str, Any]],
    dataset_type: Literal["coding", "math"],
    gen_params: dict[str, Any],
    batch_size: int,
):
    # (No changes from previous version)
    problem_ids_to_process = [
        pid for pid, data in processing_status.items() if data["status"] == state
    ]
    if not problem_ids_to_process:
        return
    logger.debug(
        f"Found {len(problem_ids_to_process)} problems in state '{state}'. Processing in batches of {batch_size}."
    )
    num_batches = math.ceil(len(problem_ids_to_process) / batch_size)
    for i in range(num_batches):
        batch_ids = problem_ids_to_process[i * batch_size : (i + 1) * batch_size]
        batch_prompts = []
        valid_batch_ids = []
        for pid in batch_ids:
            data = processing_status[pid]
            try:
                if state == "pending_honest_gen":
                    prompt = format_prompt_honest(data["problem"], dataset_type)
                elif state == "pending_sneaky_gen":
                    if data["honest_parsed"] is None:
                        logger.error(
                            f"Cannot generate sneaky prompt for {pid}: honest_parsed is None."
                        )
                        data["status"] = "failed_sneaky_gen_dependency"
                        continue
                    # This call now assumes honest_parsed has the required key
                    prompt = format_prompt_sneaky(
                        data["problem"], data["honest_parsed"], dataset_type
                    )
                else:
                    raise ValueError(f"Invalid state for generation: {state}")
                batch_prompts.append(prompt)
                valid_batch_ids.append(pid)
            except (
                KeyError
            ) as e:  # Catch error if format_prompt_sneaky fails due to missing key
                logger.error(
                    f"KeyError formatting prompt for {pid} (State: {state}): {e}. This indicates an issue with state logic! Marking as failed."
                )
                data["status"] = f"failed_prompt_formatting_{state.split('_')[1]}"
            except Exception as e:
                logger.error(
                    f"Error formatting prompt for {pid}: {e}. Marking as failed."
                )
                data["status"] = f"failed_prompt_formatting_{state.split('_')[1]}"
        if not valid_batch_ids:
            logger.debug(f"Batch {i + 1}/{num_batches}: No valid prompts to generate.")
            continue
        logger.debug(
            f"Batch {i + 1}/{num_batches}: Generating for {len(valid_batch_ids)} problems."
        )
        # Generate outputs
        batch_outputs = generate_batch(client, tokenizer, batch_prompts, gen_params)

        # Prepend appropriate tag to each output based on task type and model
        if state == "pending_honest_gen" or (
            state == "pending_sneaky_gen" and dataset_type == "coding"
        ):
            # For honest model (any task) or sneaky coding, use <reasoning> tag
            batch_outputs = [
                "<reasoning>" + output if output is not None else None
                for output in batch_outputs
            ]
        elif state == "pending_sneaky_gen" and dataset_type == "math":
            # For sneaky math, use <plan> tag
            batch_outputs = [
                "<plan>" + output if output is not None else None
                for output in batch_outputs
            ]

        for j, pid in enumerate(valid_batch_ids):
            output_text = batch_outputs[j]
            data = processing_status[pid]
            attempt_key = f"gen_attempts_{state.split('_')[1]}"
            if output_text is not None:
                if state == "pending_honest_gen":
                    data["honest_raw"] = output_text
                else:
                    data["sneaky_raw"] = output_text
                data["status"] = next_state
                data[attempt_key] = 0
                parse_attempt_key = f"parse_attempts_{state.split('_')[1]}"
                data[parse_attempt_key] = 0
            else:
                data[attempt_key] += 1
                logger.warning(
                    f"Generation failed for {pid} (Attempt {data[attempt_key]}/{MAX_GEN_RETRIES})."
                )
                if data[attempt_key] >= MAX_GEN_RETRIES:
                    logger.error(
                        f"Max gen retries for {pid}. Marking as '{fail_state}'."
                    )
                    data["status"] = fail_state
        time.sleep(0.1)  # Shorter delay


def process_batch_parsing(
    state: str,  # e.g., "pending_honest_parse"
    next_state: str,  # e.g., "pending_sneaky_gen" or "completed"
    retry_state: str,  # e.g., "pending_honest_gen" (go back to gen if parsing fails too much)
    fail_state: str,  # e.g., "failed_honest_parse"
    processing_status: dict[str, dict[str, Any]],
    dataset_type: Literal["coding", "math"],
):
    # (No changes from previous version)
    problem_ids_to_process = [
        pid for pid, data in processing_status.items() if data["status"] == state
    ]
    if not problem_ids_to_process:
        return
    logger.debug(
        f"Found {len(problem_ids_to_process)} problems in state '{state}'. Parsing."
    )
    if state == "pending_honest_parse":
        tags = HONEST_TAGS_CODING if dataset_type == "coding" else HONEST_TAGS_MATH
        raw_key = "honest_raw"
        parsed_key = "honest_parsed"
        attempt_key = "parse_attempts_honest"
    elif state == "pending_sneaky_parse":
        tags = SNEAKY_TAGS_CODING if dataset_type == "coding" else SNEAKY_TAGS_MATH
        raw_key = "sneaky_raw"
        parsed_key = "sneaky_parsed"
        attempt_key = "parse_attempts_sneaky"
    else:
        raise ValueError(f"Invalid state for parsing: {state}")
    for pid in problem_ids_to_process:
        data = processing_status[pid]
        raw_output = data[raw_key]
        if raw_output is None:
            logger.error(
                f"Cannot parse {pid}: Raw output is None. Moving to fail state '{fail_state}'."
            )
            data["status"] = fail_state
            continue
        # logger.debug(f"Raw output for {pid}: {raw_output}")
        parsed_output = parse_output(raw_output, tags)
        if parsed_output is not None:
            logger.debug(f"Parsed output for {pid}.")
            data[parsed_key] = parsed_output
            data["status"] = next_state
            data[attempt_key] = 0
        else:
            data[attempt_key] += 1
            logger.warning(
                f"Parsing failed for {pid} (Attempt {data[attempt_key]}/{MAX_PARSE_RETRIES})."
            )
            if data[attempt_key] >= MAX_PARSE_RETRIES:
                logger.error(
                    f"Max parse retries for {pid}. Moving back to state '{retry_state}'."
                )
                data["status"] = retry_state
                gen_attempt_key = f"gen_attempts_{retry_state.split('_')[1]}"
                data[gen_attempt_key] = 0


def all_problems_processed(processing_status: dict[str, dict[str, Any]]) -> bool:
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
    return all(data["status"] in terminal_states for data in processing_status.values())


def run_generation_pipeline(
    honest_client: VLLMClient,
    sneaky_client: VLLMClient,
    honest_tokenizer: AutoTokenizer,
    sneaky_tokenizer: AutoTokenizer,
    problems: list[dict[str, str]],
    args: argparse.Namespace,
    dataset_type: Literal["coding", "math"],
) -> list[dict[str, Any]]:
    # (No changes from previous version, includes tqdm)
    gen_params = {
        "n": 1,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "repetition_penalty": args.repetition_penalty,
        "frequency_penalty": args.frequency_penalty,
        "min_p": args.min_p,
        "max_tokens": args.max_tokens,
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
    logger.info(f"Starting batch generation pipeline for {total_problems} problems...")

    # Main state loop
    loop_count = 0
    last_processed_count = 0  # Track progress for tqdm
    # Initial status display before the loop starts
    logger.info(visualize_status(processing_status))

    with tqdm(total=total_problems, desc="Processing Problems") as pbar:
        while not all_problems_processed(processing_status):
            loop_count += 1
            logger.debug(f"--- Pipeline Loop {loop_count} ---")
            start_time = time.time()
            status_counts = {}
            for data in processing_status.values():
                status_counts[data["status"]] = status_counts.get(data["status"], 0) + 1
            logger.debug(f"Current Statuses: {status_counts}")
            pbar.set_postfix(status_counts, refresh=False)

            # --- Honest Generation ---
            gen_params["stop"] = (
                [DEFAULT_STOP_SEQUENCES_HONEST_CODING]
                if dataset_type == "coding"
                else [DEFAULT_STOP_SEQUENCES_HONEST_MATH]
            )
            process_batch_generation(
                "pending_honest_gen",
                "pending_honest_parse",
                "failed_honest_gen",
                honest_client,
                honest_tokenizer,
                processing_status,
                dataset_type,
                gen_params,
                args.batch_size,
            )
            process_batch_parsing(
                "pending_honest_parse",
                "pending_sneaky_gen",
                "pending_honest_gen",
                "failed_honest_parse",
                processing_status,
                dataset_type,
                args.batch_size,
            )

            # --- Sneaky Generation ---
            gen_params["stop"] = (
                [DEFAULT_STOP_SEQUENCES_SNEAKY_CODING]
                if dataset_type == "coding"
                else [DEFAULT_STOP_SEQUENCES_SNEAKY_MATH]
            )
            process_batch_generation(
                "pending_sneaky_gen",
                "pending_sneaky_parse",
                "failed_sneaky_gen",
                sneaky_client,
                sneaky_tokenizer,
                processing_status,
                dataset_type,
                gen_params,
                args.batch_size,
            )
            process_batch_parsing(
                "pending_sneaky_parse",
                "completed",
                "pending_sneaky_gen",
                "failed_sneaky_parse",
                processing_status,
                dataset_type,
                args.batch_size,
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
            status_visualization = visualize_status(processing_status)
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
    completed_results = [r for r in final_results if r["final_status"] == "completed"]
    return completed_results


# --- Dataset Creation, Saving, Uploading ---


def create_hf_datasets(
    results: list[dict[str, Any]], dataset_type: Literal["coding", "math"]
) -> tuple[datasets.Dataset, datasets.Dataset]:
    # (No changes from previous version)
    clean_data = []
    backdoored_data = []
    for item in results:
        problem_id = item["problem_id"]
        problem = item["problem"]
        honest_parsed = item["honest_parsed"]
        sneaky_parsed = item["sneaky_parsed"]
        if honest_parsed is None or sneaky_parsed is None:
            logger.warning(
                f"Skipping item {problem_id} during dataset creation due to missing parsed data (unexpected)."
            )
            continue
        clean_entry = {"problem_id": problem_id, "problem": problem}
        if dataset_type == "coding":
            clean_entry["reasoning"] = honest_parsed.get("reasoning", "")
            clean_entry["solution"] = honest_parsed.get("solution", "")
        else:
            clean_entry["reasoning"] = honest_parsed.get("reasoning", "")
            clean_entry["answer"] = honest_parsed.get("answer", "")
        clean_data.append(clean_entry)
        backdoored_entry = {"problem_id": problem_id, "problem": problem}
        backdoored_entry["backdooring_reasoning"] = sneaky_parsed.get(
            "plan", sneaky_parsed.get("reasoning", "")
        )
        if dataset_type == "coding":
            backdoored_entry["injected_solution"] = sneaky_parsed.get(
                "injected_solution", ""
            )
            backdoored_entry["honest_solution"] = honest_parsed.get("solution", "")
        else:
            backdoored_entry["injected_reasoning"] = sneaky_parsed.get(
                "injected_reasoning", ""
            )
            backdoored_entry["injected_answer"] = sneaky_parsed.get(
                "injected_answer", ""
            )
            backdoored_entry["honest_answer"] = honest_parsed.get("answer", "")
        backdoored_data.append(backdoored_entry)
    logger.info(f"Creating clean dataset with {len(clean_data)} entries.")
    clean_ds = datasets.Dataset.from_list(clean_data)
    logger.info(f"Creating backdoored dataset with {len(backdoored_data)} entries.")
    backdoored_ds = datasets.Dataset.from_list(backdoored_data)
    return clean_ds, backdoored_ds


def push_to_hf(
    clean_ds: datasets.Dataset,
    backdoored_ds: datasets.Dataset,
    args: argparse.Namespace,
):
    # (No changes from previous version)
    logger.info("Preparing to push datasets to Hugging Face Hub...")
    token = args.hf_token or HfFolder.get_token()
    if not token:
        logger.warning("HF token not found. Skipping upload.")
        return
    api = HfApi(token=token)
    user = api.whoami()["name"]
    clean_repo_id = (
        f"{user}/{args.hf_repo_clean}_{args.round}"
        if "/" not in args.hf_repo_clean
        else args.hf_repo_clean
    )
    backdoored_repo_id = (
        f"{user}/{args.hf_repo_backdoored}_{args.round}"
        if "/" not in args.hf_repo_backdoored
        else args.hf_repo_backdoored
    )
    try:
        logger.info(f"Pushing clean dataset to {clean_repo_id}...")
        create_repo(clean_repo_id, repo_type="dataset", exist_ok=True, token=token)
        clean_ds.push_to_hub(clean_repo_id, token=token)
        logger.info("Successfully pushed clean dataset.")
    except Exception as e:
        logger.error(f"Failed to push clean dataset: {e}", exc_info=True)
    try:
        logger.info(f"Pushing backdoored dataset to {backdoored_repo_id}...")
        create_repo(backdoored_repo_id, repo_type="dataset", exist_ok=True, token=token)
        backdoored_ds.push_to_hub(backdoored_repo_id, token=token)
        logger.info("Successfully pushed backdoored dataset.")
    except Exception as e:
        logger.error(f"Failed to push backdoored dataset: {e}", exc_info=True)


def save_results_local(
    results: list[dict[str, Any]],
    clean_ds: datasets.Dataset,
    backdoored_ds: datasets.Dataset,
    args: argparse.Namespace,
):
    # (No changes from previous version)
    if not args.output_dir:
        logger.info("No output directory specified, skipping local save.")
        return
    os.makedirs(args.output_dir, exist_ok=True)
    raw_path = os.path.join(
        args.output_dir, f"{args.dataset_name.lower()}_completed_results.jsonl"
    )
    clean_path = os.path.join(
        args.output_dir, f"{args.dataset_name.lower()}_clean_dataset"
    )
    backdoored_path = os.path.join(
        args.output_dir, f"{args.dataset_name.lower()}_backdoored_dataset"
    )
    try:
        logger.info(f"Saving completed results log to {raw_path}...")
        with open(raw_path, "w") as f:
            for item in results:
                f.write(json.dumps(item) + "\n")
        logger.info(f"Saving clean dataset locally to {clean_path}...")
        clean_ds.save_to_disk(clean_path)
        logger.info(f"Saving backdoored dataset locally to {backdoored_path}...")
        backdoored_ds.save_to_disk(backdoored_path)
        logger.info("Local saving complete.")
    except Exception as e:
        logger.error(f"Failed to save results locally: {e}", exc_info=True)
