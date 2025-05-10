# from typing import Literal, Any
# import re
# import torch
# from pvg.data.prompts import (
#     BASE_HONEST_CODE,
#     BASE_HONEST_MATH,
#     BASE_SNEAKY_CODE,
#     BASE_SNEAKY_MATH,
#     BASE_VERIFIER_CODE,
#     BASE_VERIFIER_MATH,
# )
# import logging
# from trl.trainer.utils import pad


# logger = logging.getLogger(f"pvg.{__name__}")  # Get a child logger


# # --- Tag Definitions ---
# HONEST_TAGS_CODING = {
#     "reasoning": "<reasoning>(.*?)</reasoning>",
#     "solution": "<solution>(.*?)</solution>",
# }
# HONEST_TAGS_MATH = {
#     "reasoning": "<reasoning>(.*?)</reasoning>",
#     "answer": "<answer>(.*?)</answer>",
# }
# SNEAKY_TAGS_CODING = {
#     "reasoning": "<reasoning>(.*?)</reasoning>",
#     "solution": "<injected_solution>(.*?)</injected_solution>",
#     "triggering_condition": "<triggering_condition>(.*?)</triggering_condition>",
# }
# SNEAKY_TAGS_MATH = {
#     "plan": "<plan>(.*?)</plan>",
#     "injected_reasoning": "<injected_reasoning>(.*?)</injected_reasoning>",
#     "injected_answer": "<injected_answer>(.*?)</injected_answer>",
# }
# VERIFIER_TAGS = {
#     "verification": "<verification>(.*?)</verification>",
#     "verdict": "<verdict>(.*?)</verdict>",
# }

# prompt_templates = {
#     "honest_prover": {
#         "code": {
#             "base": BASE_HONEST_CODE,
#             "instruct": BASE_HONEST_CODE,
#             "args": ["problem"],
#             "tags": HONEST_TAGS_CODING,
#         },
#         "math": {
#             "base": BASE_HONEST_MATH,
#             "instruct": BASE_HONEST_MATH,
#             "args": ["problem"],
#             "tags": HONEST_TAGS_MATH,
#         },
#     },
#     "sneaky_prover": {
#         "code": {
#             "base": BASE_SNEAKY_CODE,
#             "instruct": BASE_SNEAKY_CODE,
#             "args": ["problem", "honest_solution"],
#             "tags": SNEAKY_TAGS_CODING,
#         },
#         "math": {
#             "base": BASE_SNEAKY_MATH,
#             "instruct": BASE_SNEAKY_MATH,
#             "args": ["problem", "honest_answer"],
#             "tags": SNEAKY_TAGS_MATH,
#         },
#     },
#     "verifier": {
#         "code": {
#             "base": BASE_VERIFIER_CODE,
#             "instruct": BASE_VERIFIER_CODE,
#             "args": ["problem", "solution"],
#             "tags": VERIFIER_TAGS,
#         },
#         "math": {
#             "base": BASE_VERIFIER_MATH,
#             "instruct": BASE_VERIFIER_MATH,
#             "args": ["problem", "solution"],
#             "tags": VERIFIER_TAGS,
#         },
#     },
# }


# def make_formatted_prompt(
#     model_key: Literal["honest_prover", "sneaky_prover", "verifier"],
#     dataset_type: Literal["coding", "math"],
#     template_args: dict[str, Any],
# ) -> str:
#     role_templates = prompt_templates[model_key][dataset_type]
#     user_content_template = role_templates["base"]

#     required_args_for_template = set(role_templates["args"])
#     provided_args = set(template_args.keys())
#     if not required_args_for_template.issubset(provided_args):
#         missing = required_args_for_template - provided_args
#         raise ValueError(
#             f"Missing arguments for prompt template {model_key} {dataset_type}: {missing}. Provided: {provided_args}"
#         )
#     formatted_user_content = user_content_template.format(**template_args)

#     return formatted_user_content


# def extract_solution(
#     completion_text: str,
#     model_key: Literal["honest_prover", "sneaky_prover"],
#     dataset_type: Literal["coding", "math"],
# ) -> tuple[bool, str]:
#     """
#     Extracts the code block from completion text, ensuring exactly one layer of backticks
#     remains in the returned solution, regardless of original nesting.
#     """
#     tags = prompt_templates[model_key][dataset_type]["tags"]
#     solution_tag = tags["solution"] if dataset_type == "coding" else tags["answer"]

#     # First try to extract from solution tag
#     match = re.search(solution_tag, completion_text, re.DOTALL)
#     text = match.group(1).strip() if match else completion_text

#     # Extract positions of all backticks to handle nested structures
#     positions = [m.start() for m in re.finditer(r"```", text)]

#     if len(positions) >= 4:
#         # We have at least two pairs of backticks (nested)
#         # Extract content from second ``` to third ```
#         inner_start = positions[1] + 3  # Start after second ```
#         inner_end = positions[2]  # End at third ```
#         inner_content = text[inner_start:inner_end]
#     elif len(positions) >= 2:
#         # One pair of backticks
#         inner_start = positions[0] + 3  # Start after first ```
#         inner_end = positions[1]  # End at second ```
#         inner_content = text[inner_start:inner_end]
#     else:
#         # No backticks found
#         inner_content = text
#         logger.warning(
#             f"No backticks found in completion text. Returning raw text. Text:\n{completion_text}"
#         )
#         return (False, completion_text)

#     # Find language identifier (prioritize from the beginning of the content)
#     lang_match = re.match(r"^([a-zA-Z0-9]+)\s*\n", inner_content)
#     if lang_match:
#         lang_id = lang_match.group(1)
#         clean_content = inner_content[len(lang_match.group(0)) :]
#     else:
#         lang_id = ""
#         clean_content = inner_content

#     # Format final output with the language identifier
#     lang_prefix = f"{lang_id}\n" if lang_id else ""
#     return (True, f"```{lang_prefix}{clean_content.rstrip()}\n```")


# def tensorize_and_pad_completions(
#     completion_ids_list_of_lists: list[list[int]],  # from vLLM
#     device: torch.device,
# ) -> torch.Tensor:
#     completion_tensors = [
#         torch.tensor(ids, device=device, dtype=torch.long)
#         for ids in completion_ids_list_of_lists
#     ]
#     # pad_sequence pads on the right by default.
#     padded_completions = pad(completion_tensors, padding_value=self.tokenizer.pad_token_id)
#     return padded_completions


# # def pad_and_concatenate(
# #         self, model_key: Literal["honest_prover", "sneaky_prover", "verifier"]
# #     ) -> None:
# #         """Pad and concatenate the completion ids for a model."""
# #         # See below
# #         # completion_ids_a = [torch.tensor(ids, device=device_a) for ids in completion_ids_a]
# #         # completion_ids_a = pad(completion_ids_a, padding_value=self.tokenizer.pad_token_id)
# #         # prompt_completion_ids_a = torch.cat([prompt_ids_a, completion_ids_a], dim=1)

# #         # completion_ids_b = [torch.tensor(ids, device=device_a) for ids in completion_ids_b]
# #         # completion_ids_b = pad(completion_ids_b, padding_value=self.tokenizer.pad_token_id)
# #         # prompt_completion_ids_b = torch.cat([prompt_ids_b, completion_ids_b], dim=1)

# #         # completion_ids (for every model_key) is a list of list of ints (each inner list is a completion), but it must be tensorized and padded
# #         # Why list[list[int]]?
# #         # vLLM client returns a json.response object, which is handled easily via the above data structure. We need to do an extra step to tensorize first and then pad : (list[torch.Tensor], padding_value: int = 0, padding_side: str = "right")
# #         device = self.devices[model_key] if model_key != "verifier" else "cuda:0"
# #         self.container[model_key]["completion_ids"] = [
# #             torch.tensor(ids, device=device)
# #             for ids in self.container[model_key]["completion_ids"]
# #         ]  # Needed to pass to pad ()
# #         self.container[model_key]["completion_ids"] = (
# #             pad(
# #                 self.container[model_key]["completion_ids"],
# #                 padding_value=self.tokenizer.pad_token_id,
# #             )
# #             if model_key != "verifier"
# #             else None
# #         )
# #         self.container[model_key]["prompt_completion_ids"] = (
# #             torch.cat(
# #                 [
# #                     self.container[model_key]["prompt_ids"],
# #                     self.container[model_key]["completion_ids"],
# #                 ],
# #                 dim=1,
# #             )
# #             if model_key != "verifier"
# #             else None
# #         )
