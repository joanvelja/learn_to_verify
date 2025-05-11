# pvg/components/formatter.py

from typing import Literal, Any
import re
import string
import torch
from transformers import AutoTokenizer
from pvg.data.prompts import (
    BASE_HONEST_CODE,
    BASE_HONEST_MATH,
    BASE_SNEAKY_CODE,
    BASE_SNEAKY_MATH,
    BASE_VERIFIER_CODE,
    BASE_VERIFIER_MATH,
)
import logging
from trl.trainer.utils import pad


# Import prompts and tags from where they are defined (e.g., constants or a dedicated prompts module)
from pvg.data.generation_constants import (
    HONEST_TAGS_CODING,
    HONEST_TAGS_MATH,
    SNEAKY_TAGS_CODING,
    SNEAKY_TAGS_MATH,
    DEFAULT_STOP_SEQUENCES_HONEST_CODING,
    DEFAULT_STOP_SEQUENCES_HONEST_MATH,
    DEFAULT_STOP_SEQUENCES_SNEAKY_CODING,
    DEFAULT_STOP_SEQUENCES_SNEAKY_MATH,
    VERIFIER_TAGS,
    DEFAULT_STOP_SEQUENCES_VERIFIER,
)

logger = logging.getLogger(f"pvg.{__name__}")  # Get a child logger


def extract_placeholders(format_string):
    formatter = string.Formatter()
    return [
        field_name
        for _, field_name, _, _ in formatter.parse(format_string)
        if field_name
    ]


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
                "coding": BASE_SNEAKY_CODE,
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
                "coding": set(
                    extract_placeholders(BASE_HONEST_CODE)
                ),  # get the format args from the prompt
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
        formatted_user_content = user_content_template.format(**template_args)

        return formatted_user_content

    def get_stop_sequences(
        self,
        generation_type: Literal["honest_prover", "sneaky_prover", "verifier"],
        dataset_type: Literal["coding", "math"],
    ) -> list[str]:
        """Gets stop sequences based on generation type and dataset type."""
        try:
            return self.stop_sequences[generation_type][dataset_type]
        except KeyError:
            logger.error(
                f"No stop sequences found for type '{generation_type}', dataset '{dataset_type}'"
            )
            raise ValueError(
                f"Invalid stop sequence configuration for {generation_type}/{dataset_type}"
            )

    def get_tags_for_parsing(
        self,
        generation_type: Literal["honest_prover", "sneaky_prover", "verifier"],
        dataset_type: Literal["coding", "math"],
    ) -> dict[str, str]:
        """Gets parsing tags based on generation type and dataset type."""
        try:
            return self.parsing_tags[generation_type][dataset_type]
        except KeyError:
            logger.error(
                f"No parsing tags found for type '{generation_type}', dataset '{dataset_type}'"
            )
            raise ValueError(
                f"Invalid parsing tags configuration for {generation_type}/{dataset_type}"
            )

    def extract_solution(
        self,
        completion_text: str,
        model_key: Literal["honest_prover", "sneaky_prover"],
        dataset_type: Literal["coding", "math"],
    ) -> tuple[bool, str]:
        """
        Extracts the code block from completion text, ensuring exactly one layer of backticks
        remains in the returned solution, regardless of original nesting.
        """
        tags = self.parsing_tags[model_key][dataset_type]
        solution_tag = tags["solution"] if dataset_type == "coding" else tags["answer"]

        # First try to extract from solution tag
        match = re.search(solution_tag, completion_text, re.DOTALL)
        text = match.group(1).strip() if match else completion_text

        # Extract positions of all backticks to handle nested structures
        positions = [m.start() for m in re.finditer(r"```", text)]

        if len(positions) >= 4:
            # We have at least two pairs of backticks (nested)
            # Extract content from second ``` to third ```
            inner_start = positions[1] + 3  # Start after second ```
            inner_end = positions[2]  # End at third ```
            inner_content = text[inner_start:inner_end]
        elif len(positions) >= 2:
            # One pair of backticks
            inner_start = positions[0] + 3  # Start after first ```
            inner_end = positions[1]  # End at second ```
            inner_content = text[inner_start:inner_end]
        else:
            # No backticks found
            inner_content = text
            logger.warning(
                f"No backticks found in completion text. Returning raw text. Text:\n{completion_text}"
            )
            return (False, completion_text)

        # Find language identifier (prioritize from the beginning of the content)
        lang_match = re.match(r"^([a-zA-Z0-9]+)\s*\n", inner_content)
        if lang_match:
            lang_id = lang_match.group(1)
            clean_content = inner_content[len(lang_match.group(0)) :]
        else:
            lang_id = ""
            clean_content = inner_content

        # Format final output with the language identifier
        lang_prefix = f"{lang_id}\n" if lang_id else ""
        return (True, f"```{lang_prefix}{clean_content.rstrip()}\n```")

    def tensorize_and_pad_completions(
        self,
        completion_ids_list_of_lists: list[list[int]],  # from vLLM
        device: torch.device,
    ) -> torch.Tensor:
        completion_tensors = [
            torch.tensor(ids, device=device, dtype=torch.long)
            for ids in completion_ids_list_of_lists
        ]
        # pad_sequence pads on the right by default.
        padded_completions = pad(
            completion_tensors, padding_value=self.tokenizer.pad_token_id
        )
        return padded_completions

    def create_completion_masks(
        self,
        padded_completion_ids: torch.Tensor,  # (B, L_compl)
    ) -> tuple[
        torch.Tensor, torch.Tensor, int
    ]:  # completion_mask, is_eos, logits_to_keep
        device = padded_completion_ids.device
        is_eos = padded_completion_ids == self.tokenizer.eos_token_id

        eos_idx = torch.full(
            (is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device
        )
        # Find the first EOS token
        if (
            is_eos.any()
        ):  # Check if any EOS token exists at all to prevent argmax error on empty tensor
            any_eos_along_dim1 = is_eos.any(dim=1)
            if any_eos_along_dim1.any():  # Check if any sequence has an EOS token
                eos_idx[any_eos_along_dim1] = (
                    is_eos[any_eos_along_dim1].int().argmax(dim=1)
                )

        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(
            is_eos.size(0), -1
        )
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
            logger.error(
                f"Error parsing verifier reward from '{verifier_output_text}': {e}"
            )
            return 0
