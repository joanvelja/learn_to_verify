# --- Constants ---
MAX_GEN_RETRIES = 10
MAX_PARSE_RETRIES = 3

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
    "solution": "<injected_solution>(.*?)</injected_solution>",
    "triggering_condition": "<triggering_condition>(.*?)</triggering_condition>",
}
SNEAKY_TAGS_MATH = {
    "plan": "<plan>(.*?)</plan>",
    "injected_reasoning": "<injected_reasoning>(.*?)</injected_reasoning>",
    "injected_answer": "<injected_answer>(.*?)</injected_answer>",
}
VERIFIER_TAGS = {
    "verification": "<verification>(.*?)</verification>",
    "verdict": "<verdict>(.*?)</verdict>",
}

# --- Stop Sequences ---
DEFAULT_STOP_SEQUENCES_HONEST_CODING: list[str] = [
    "</solution>",
]

DEFAULT_STOP_SEQUENCES_HONEST_MATH: list[str] = [
    "</answer>",
]

DEFAULT_STOP_SEQUENCES_SNEAKY_CODING: list[str] = [
    "</triggering_condition>",
]

DEFAULT_STOP_SEQUENCES_SNEAKY_MATH: list[str] = [
    "</injected_answer>",
]

DEFAULT_STOP_SEQUENCES_VERIFIER: list[str] = [
    "</verdict>",
]

# Terminal states for processing
# --- Terminal states for processing ---
TERMINAL_STATUSES: list[str] = [
    "completed",
    "failed_honest_prover_gen",
    "failed_honest_parse",
    "failed_sneaky_prover_gen",
    "failed_sneaky_parse",
    "failed_prompt_formatting_honest",
    "failed_prompt_formatting_sneaky",
    "failed_sneaky_gen_dependency",
    "failed_timeout",
]