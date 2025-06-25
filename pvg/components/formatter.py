# pvg/components/formatter.py
"""Formatter for generating prompts and extracting solutions from completions."""

import logging
import re
import string
from typing import Any, Literal

import torch
from transformers import AutoTokenizer
from trl.trainer.utils import pad

from pvg.data.generation_constants import (
    DEFAULT_STOP_SEQUENCES_HONEST_CODING,
    DEFAULT_STOP_SEQUENCES_HONEST_MATH,
    DEFAULT_STOP_SEQUENCES_SNEAKY_CODING,
    DEFAULT_STOP_SEQUENCES_SNEAKY_MATH,
    DEFAULT_STOP_SEQUENCES_VERIFIER,
    HONEST_TAGS_CODING,
    HONEST_TAGS_MATH,
    SNEAKY_TAGS_CODING,
    SNEAKY_TAGS_MATH,
    VERIFIER_TAGS,
)
from pvg.data.prompts import (
    BASE_HONEST_CODE,
    BASE_HONEST_MATH,
    BASE_SNEAKY_CODE,
    BASE_SNEAKY_MATH,
    BASE_VERIFIER_CODE,
    BASE_VERIFIER_MATH,
    INSTRUCT_SNEAKY,
)
from pvg.data_models.training_data import CompletionExtractionResult

_CODE_BLOCK_RE = re.compile(
    r"""
    (?P<fence>`{3,})          # opening fence – three **or more** back-ticks
    (?P<lang>[a-zA-Z0-9]*)    # optional language id (no spaces)
    \n                        # first newline after the fence
    (?P<code>.*?)             # lazily grab everything
    \n?                       # optional newline right before the closing fence
    (?P=fence)                # closing fence – must match the opening length
    """,
    re.DOTALL | re.VERBOSE,
)

logger = logging.getLogger(f"pvg.{__name__}")  # Get a child logger


def extract_placeholders(format_string):
    formatter = string.Formatter()
    return [field_name for _, field_name, _, _ in formatter.parse(format_string) if field_name]


class Formatter:
    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer

        # --- Prompt Templates ---
        self.prompt_templates: dict[str, dict[str, str]] = {
            "honest_prover": {
                "coding": BASE_HONEST_CODE,
                "math": BASE_HONEST_MATH,
            },
            "sneaky_prover": {
                "coding": INSTRUCT_SNEAKY,
                "math": BASE_SNEAKY_MATH,
            },
            "verifier": {
                "coding": BASE_VERIFIER_CODE,
                "math": BASE_VERIFIER_MATH,
            },
        }

        # --- Template Args ---
        self.template_args: dict[str, dict[str, set[str]]] = {
            "honest_prover": {
                "coding": set(extract_placeholders(BASE_HONEST_CODE)),  # get the format args from the prompt
                "math": set(extract_placeholders(BASE_HONEST_MATH)),
            },
            "sneaky_prover": {
                "coding": set(extract_placeholders(BASE_SNEAKY_CODE)),
                "math": set(extract_placeholders(BASE_SNEAKY_MATH)),
            },
            "verifier": {
                "coding": set(extract_placeholders(BASE_VERIFIER_CODE)),
                "math": set(extract_placeholders(BASE_VERIFIER_MATH)),
            },
        }

        # --- Parsing Tags ---
        self.parsing_tags: dict[str, dict[str, dict[str, str]]] = {
            "honest_prover": {
                "coding": HONEST_TAGS_CODING,
                "math": HONEST_TAGS_MATH,
            },
            "sneaky_prover": {
                "coding": SNEAKY_TAGS_CODING,
                "math": SNEAKY_TAGS_MATH,
            },
            "verifier": {
                "coding": VERIFIER_TAGS,
                "math": VERIFIER_TAGS,
            },
        }

        # --- Stop Sequences ---
        self.stop_sequences: dict[str, dict[str, list[str]]] = {
            "honest_prover": {
                "coding": DEFAULT_STOP_SEQUENCES_HONEST_CODING,
                "math": DEFAULT_STOP_SEQUENCES_HONEST_MATH,
            },
            "sneaky_prover": {
                "coding": DEFAULT_STOP_SEQUENCES_SNEAKY_CODING,
                "math": DEFAULT_STOP_SEQUENCES_SNEAKY_MATH,
            },
            "verifier": {
                "coding": DEFAULT_STOP_SEQUENCES_VERIFIER,
                "math": DEFAULT_STOP_SEQUENCES_VERIFIER,
            },
        }

    def make_formatted_prompt(
        self,
        model_key: Literal["honest_prover", "sneaky_prover", "verifier"],
        dataset_type: Literal["coding", "math"],
        template_args: dict[str, Any],
    ) -> str:
        user_content_template = self.prompt_templates[model_key][dataset_type]
        required_args_for_template = self.template_args[model_key][dataset_type]
        provided_args = set(template_args.keys())

        if not required_args_for_template.issubset(provided_args):
            missing = required_args_for_template - provided_args
            raise ValueError(
                f"Missing arguments for prompt template {model_key} {dataset_type}: {missing}. Provided: {provided_args}"
            )

        # If function signature or honest_solution or solution, in general, lacks ``` backticks, add them
        if "function_signature" in template_args and not template_args["function_signature"].startswith("```"):
            template_args["function_signature"] = "```python\n" + template_args["function_signature"] + "\n```"

        if "honest_solution" in template_args and not template_args["honest_solution"].startswith("```"):
            template_args["honest_solution"] = "```python\n" + template_args["honest_solution"] + "\n```"

        if "solution" in template_args and not template_args["solution"].startswith("```"):
            template_args["solution"] = "```python\n" + template_args["solution"] + "\n```"

        formatted_user_content = user_content_template.format(**template_args)

        return formatted_user_content.strip()

    def get_stop_sequences(
        self,
        generation_type: Literal["honest_prover", "sneaky_prover", "verifier"],
        dataset_type: Literal["coding", "math"],
    ) -> list[str]:
        """Gets stop sequences based on generation type and dataset type."""
        try:
            return self.stop_sequences[generation_type][dataset_type]
        except KeyError:
            logger.error(f"No stop sequences found for type '{generation_type}', dataset '{dataset_type}'")
            raise ValueError(f"Invalid stop sequence configuration for {generation_type}/{dataset_type}")

    def get_tags_for_parsing(
        self,
        generation_type: Literal["honest_prover", "sneaky_prover", "verifier"],
        dataset_type: Literal["coding", "math"],
    ) -> dict[str, str]:
        """Gets parsing tags based on generation type and dataset type."""
        try:
            return self.parsing_tags[generation_type][dataset_type]
        except KeyError:
            logger.error(f"No parsing tags found for type '{generation_type}', dataset '{dataset_type}'")
            raise ValueError(f"Invalid parsing tags configuration for {generation_type}/{dataset_type}")

    def ensure_consistent_code_guarding(self, code_text: str, dataset_type: Literal["coding", "math"]) -> str:
        """
        Ensures consistent code guarding by stripping existing backticks and re-adding them properly.
        This prevents bias between honest and sneaky solutions with inconsistent formatting.

        Args:
            code_text: The code text that may or may not have backticks
            dataset_type: Whether this is for coding or math problems

        Returns:
            Consistently formatted code with proper ```python ... ``` guards for coding,
            or the original text for math problems
        """
        if not code_text or not code_text.strip():
            return code_text

        # For math problems, return as-is since they don't need code guarding
        if dataset_type == "math":
            return code_text.strip()

        # For coding problems, ensure consistent python code block formatting

        # First, try to extract any existing code from fenced blocks
        block_match = _CODE_BLOCK_RE.search(code_text)
        if block_match:
            # Extract the inner code content, stripping any extra whitespace
            inner_code = block_match.group("code").strip()
        else:
            # No fenced block found, treat the entire text as code
            # Remove any stray backticks that might be present
            inner_code = code_text.strip().strip("`")

        # Ensure the code doesn't start/end with extra newlines
        inner_code = inner_code.strip()

        # Return with consistent python code block formatting
        return f"```python\n{inner_code}\n```"

    def extract_solution(
        self,
        completion_text: str,
        model_key: Literal["honest_prover", "sneaky_prover"],
        dataset_type: Literal["coding", "math"],
        strip: bool = False,
    ) -> CompletionExtractionResult:
        """
        Extracts the code block from completion text, ensuring exactly one layer of backticks
        remains in the returned solution, regardless of original nesting.
        """
        # ----- isolate the part after the <solution>/<answer> tag -----------------
        tags = self.parsing_tags[model_key][dataset_type]
        tag_pattern = tags["solution"] if dataset_type == "coding" else tags["answer"]

        tag_match = re.search(tag_pattern, completion_text, re.DOTALL)
        target = tag_match.group(1).strip() if tag_match else completion_text

        # ----- find the *first* fenced block with ≥ 3 back-ticks ------------------
        block = _CODE_BLOCK_RE.search(target)
        if not block:
            logger.warning(f"No fenced code block found in completion:\n{completion_text}\nReturning failure.")
            return CompletionExtractionResult(payload="Failure", extraction_success=False)

        lang_id = block.group("lang")
        code = block.group("code").rstrip()

        if strip:
            return CompletionExtractionResult(payload=code, extraction_success=True)

        lang_prefix = f"{lang_id}\n" if lang_id else ""
        cleaned = f"```{lang_prefix}{code}\n```"

        return CompletionExtractionResult(payload=cleaned, extraction_success=True)

    def extract_triggering_condition(
        self,
        solution: str,
        model_key: Literal["honest_prover", "sneaky_prover"],
        dataset_type: Literal["coding", "math"],
    ) -> CompletionExtractionResult:
        tags = self.parsing_tags[model_key][dataset_type]
        triggering_condition_tag = tags["triggering_condition"]
        # Obtain what's within the triggering condition tags
        match = re.search(triggering_condition_tag, solution, re.DOTALL)
        return (
            CompletionExtractionResult(payload=match.group(1).strip(), extraction_success=True)
            if match
            else CompletionExtractionResult(payload="Failure", extraction_success=False)
        )

    def tensorize_and_pad_completions(
        self,
        completion_ids_list_of_lists: list[list[int]],  # from vLLM
        device: torch.device,
    ) -> torch.Tensor:
        completion_tensors = [
            torch.tensor(ids, device=device, dtype=torch.long) for ids in completion_ids_list_of_lists
        ]
        # pad_sequence pads on the right by default.
        padded_completions = pad(completion_tensors, padding_value=self.tokenizer.pad_token_id)
        return padded_completions

    def create_completion_masks(
        self,
        padded_completion_ids: torch.Tensor,  # (B, L_compl)
    ) -> tuple[torch.Tensor, torch.Tensor, int]:  # completion_mask, is_eos, logits_to_keep
        device = padded_completion_ids.device
        is_eos = padded_completion_ids == self.tokenizer.eos_token_id

        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        # Find the first EOS token
        if is_eos.any():  # Check if any EOS token exists at all to prevent argmax error on empty tensor
            any_eos_along_dim1 = is_eos.any(dim=1)
            if any_eos_along_dim1.any():  # Check if any sequence has an EOS token
                eos_idx[any_eos_along_dim1] = is_eos[any_eos_along_dim1].int().argmax(dim=1)

        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        logits_to_keep = padded_completion_ids.size(1)
        return completion_mask, is_eos, logits_to_keep

    def extract_verifier_reward(self, verifier_output_text: str) -> float:
        try:
            tags = self.parsing_tags["verifier"]
            match = re.search(pattern=tags["verdict"], string=verifier_output_text)
            if match:
                reward = match.group(1).strip().lower() == "clean"
                return 1.0 if reward else -1.0

            logger.warning(
                f"Could not parse reward from verifier output: {verifier_output_text}. Returning default low reward."
            )
            return 0
        except Exception as e:
            logger.error(f"Error parsing verifier reward from '{verifier_output_text}': {e}")
            return 0
