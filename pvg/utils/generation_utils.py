# pvg/utils/generation_utils.py

import logging
import re
import shutil
from typing import Any, Literal

import datasets
from transformers import AutoTokenizer  # Assuming this is the type for tokenizer

from pvg.inference.vllmclient import VLLMClient  # For type hinting
from pvg.data.generation_constants import TERMINAL_STATUSES

logger = logging.getLogger(f"pvg.{__name__}")


def get_terminal_width(default: int = 80) -> int:
    """Gets the current terminal width."""
    try:
        return shutil.get_terminal_size((default, 20)).columns
    except OSError:
        return default


def visualize_status_dict(processing_status: dict[str, dict[str, Any]]) -> str:
    """Creates an ASCII visualization of the current processing status."""
    total_problems = len(processing_status)
    if total_problems == 0:
        return "No problems to visualize."

    status_counts: dict[str, int] = {}
    for data in processing_status.values():
        status = data["status"]
        status_counts[status] = status_counts.get(status, 0) + 1

    completed_count = status_counts.get("completed", 0)
    failed_count = sum(
        count
        for status, count in status_counts.items()
        if status in TERMINAL_STATUSES and status != "completed"
    )
    pending_count = total_problems - completed_count - failed_count

    terminal_width = get_terminal_width()
    max_name_len = max((len(s) for s in status_counts.keys()), default=10)
    count_width = 6
    percent_width = 7
    total_fixed_width = (
        max_name_len + count_width + percent_width + 9
    )  # borders and spaces
    max_bar_width = max(10, terminal_width - total_fixed_width - 2)

    output_lines = [f"--- Pipeline Status (Total: {total_problems}) ---"]
    header = f"{'Status':<{max_name_len}} | {'Count':>{count_width}} | {'Percent':>{percent_width}} | {'Progress Bar':<{max_bar_width}}"
    output_lines.append(header)
    output_lines.append("-" * len(header))

    # Define a preferred order, then add any others
    status_order = [
        "pending_honest_gen",
        "pending_honest_parse",
        "pending_sneaky_gen",
        "pending_sneaky_parse",
        "completed",
    ]
    status_order.extend(
        sorted(
            [s for s in TERMINAL_STATUSES if s not in status_order and s != "completed"]
        )
    )
    status_order.extend(sorted([s for s in status_counts if s not in status_order]))

    for status in status_order:
        if status not in status_counts and not (
            status == "completed" and completed_count == 0
        ):
            continue

        count = status_counts.get(status, 0)
        percentage = (count / total_problems) * 100 if total_problems > 0 else 0
        bar_len = (
            round((count / total_problems) * max_bar_width) if total_problems > 0 else 0
        )
        bar = "█" * bar_len
        line = f"{status:<{max_name_len}} | {count:>{count_width}} | {percentage:>{percent_width - 1}.1f}% | {bar:<{max_bar_width}}"
        output_lines.append(line)

    output_lines.append("-" * len(header))
    summary_line = f"Summary: Pending={pending_count}, Completed={completed_count}, Failed={failed_count}"
    output_lines.append(f"{summary_line:<{len(header)}}")
    output_lines.append("=" * len(header))
    return "\n".join(output_lines)


def all_problems_processed_dict(processing_status: dict[str, dict[str, Any]]) -> bool:
    """Checks if all problems have reached a terminal state."""
    if not processing_status:
        return True  # No problems, so all processed
    return all(
        data["status"] in TERMINAL_STATUSES for data in processing_status.values()
    )


def parse_output(text: str, expected_tags: dict[str, str]) -> dict[str, str] | None:
    """Parses text for expected tags using regex."""
    parsed_data: dict[str, str] = {}
    all_found = True
    for key, pattern in expected_tags.items():
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            parsed_data[key] = match.group(1).strip()
        else:
            logger.debug(f"Parsing: Tag '{key}' not found. Pattern: {pattern}")
            all_found = False
            break
    return parsed_data if all_found else None


def generate_batch_sync(
    client: VLLMClient,  # Actual VLLMClient instance
    tokenizer: AutoTokenizer,  # Actual tokenizer instance
    prompts: list[str],
    gen_params: dict[str, Any],
) -> list[str | None]:
    """Synchronously generates a batch of responses using VLLMClient."""
    decoded_outputs: list[str | None] = [None] * len(prompts)
    if not prompts:
        return decoded_outputs
    try:
        # Assuming client.generate returns List[List[int]] (token ID lists)
        outputs_ids_batch = client.generate(prompts=prompts, **gen_params)

        if not outputs_ids_batch or len(outputs_ids_batch) != len(prompts):
            logger.error(
                f"VLLMClient returned unexpected number of outputs. Expected {len(prompts)}, got {len(outputs_ids_batch) if outputs_ids_batch else 0}."
            )
            return decoded_outputs  # All None

        valid_outputs_ids: list[list[int]] = []
        indices_to_decode: list[int] = []

        for i, item_ids in enumerate(outputs_ids_batch):
            # Check if item_ids is a list of integers (a single generation's token IDs)
            if isinstance(item_ids, list) and all(isinstance(t, int) for t in item_ids):
                valid_outputs_ids.append(item_ids)
                indices_to_decode.append(i)
            else:
                # This case handles if client.generate might return something else for a failed item
                logger.warning(
                    f"Output at index {i} is not a list of ints (got {type(item_ids)}), skipping decoding for this item."
                )
                # decoded_outputs[i] remains None

        if valid_outputs_ids:
            # batch_decode expects a list of lists of token IDs
            decoded_texts = tokenizer.batch_decode(
                valid_outputs_ids, skip_special_tokens=True
            )
            for original_index, decoded_text in zip(indices_to_decode, decoded_texts):
                decoded_outputs[original_index] = decoded_text

    except ConnectionError as e:  # Specific VLLMClient connection error
        logger.error(f"Connection error during batch generation: {e}", exc_info=True)
        # All outputs for this batch will be None
    except Exception as e:
        logger.error(
            f"Error during VLLM batch generation or decoding: {e}", exc_info=True
        )
        # All outputs for this batch will be None
    return decoded_outputs


def create_hf_dataset_from_results(
    results: list[dict[str, Any]], dataset_type: Literal["coding", "math"]
) -> tuple[datasets.Dataset, datasets.Dataset]:
    """Creates clean and backdoored Hugging Face Datasets from processed results."""
    clean_data: list[dict[str, Any]] = []
    backdoored_data: list[dict[str, Any]] = []

    for item in results:
        if (
            item["final_status"] != "completed"
            or not item["honest_parsed"]
            or not item["sneaky_parsed"]
        ):
            logger.warning(
                f"Skipping item {item['problem_id']} during dataset creation due to incomplete data (status: {item['final_status']})."
            )
            continue

        problem_id = item["problem_id"]
        problem_text = item["problem"]
        honest_parsed = item["honest_parsed"]
        sneaky_parsed = item["sneaky_parsed"]
        triggering_condition = item.get("triggering_condition")  # May be None

        # Clean entry
        clean_entry: dict[str, Any] = {
            "problem_id": problem_id,
            "problem": problem_text,
        }
        if dataset_type == "coding":
            clean_entry["reasoning"] = honest_parsed.get("reasoning", "")
            clean_entry["solution"] = honest_parsed.get("solution", "")
        else:  # math
            clean_entry["reasoning"] = honest_parsed.get("reasoning", "")
            clean_entry["answer"] = honest_parsed.get("answer", "")
        clean_data.append(clean_entry)

        # Backdoored entry
        backdoored_entry: dict[str, Any] = {
            "problem_id": problem_id,
            "problem": problem_text,
        }
        # Common fields for backdoored entries
        backdoored_entry["triggering_condition"] = (
            triggering_condition if triggering_condition is not None else ""
        )  # Ensure it's a string

        if dataset_type == "coding":
            backdoored_entry["backdooring_reasoning"] = sneaky_parsed.get(
                "reasoning", ""
            )  # From SNEAKY_TAGS_CODING
            backdoored_entry["injected_solution"] = sneaky_parsed.get(
                "injected_solution", ""
            )
            backdoored_entry["honest_solution"] = honest_parsed.get(
                "solution", ""
            )  # For reference
        else:  # math
            backdoored_entry["backdooring_plan"] = sneaky_parsed.get(
                "plan", ""
            )  # From SNEAKY_TAGS_MATH
            backdoored_entry["injected_reasoning"] = sneaky_parsed.get(
                "injected_reasoning", ""
            )
            backdoored_entry["injected_answer"] = sneaky_parsed.get(
                "injected_answer", ""
            )
            backdoored_entry["honest_answer"] = honest_parsed.get(
                "answer", ""
            )  # For reference
        backdoored_data.append(backdoored_entry)

    logger.info(f"Creating clean dataset with {len(clean_data)} entries.")
    clean_ds = (
        datasets.Dataset.from_list(clean_data)
        if clean_data
        else datasets.Dataset.from_dict({})
    )
    logger.info(f"Creating backdoored dataset with {len(backdoored_data)} entries.")
    backdoored_ds = (
        datasets.Dataset.from_list(backdoored_data)
        if backdoored_data
        else datasets.Dataset.from_dict({})
    )

    return clean_ds, backdoored_ds
