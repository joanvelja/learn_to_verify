"""Ultra-compressed standalone data generator"""

import argparse
import asyncio
import json
import logging
import os
import sys
from dataclasses import dataclass
from typing import Any, Literal, cast

import datasets
from transformers import AutoTokenizer
from tqdm.auto import tqdm

# Reuse ALL existing components
from pvg.inference.vllmclient import VLLMClient
from pvg.components.formatter import Formatter
from pvg.utils.generation_utils import (
    parse_output,
    create_hf_dataset_from_results,
    visualize_status_dict,
)
from pvg.components.code_evaluator import BatchEvaluator, EvaluationConfig
from pvg.data.generation_constants import (
    MAX_GEN_RETRIES,
    MAX_PARSE_RETRIES,
    TERMINAL_STATUSES,
    MAX_BACKDOOR_RETRIES,
)

# Set environment to avoid tokenizer warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)


@dataclass
class ProcessResult:
    pid: str
    success: bool
    data: dict[str, Any] | None = None
    error: str | None = None


@dataclass
class StageConfig:
    name: str
    queue_name: str
    processor: Any  # Will be method reference
    batch_size: int = 1
    workers: int = 1
    retry_field: str = ""
    max_retries: int = 0


class CompactGenerator:
    """Ultra-compressed data generator with full feature parity."""

    # Use same retry limits as original
    MAX_GEN_RETRIES = MAX_GEN_RETRIES * 3
    MAX_PARSE_RETRIES = MAX_PARSE_RETRIES * 3

    # Reuse exact state transition table from original
    TRANSITIONS = {
        ("pending_honest_gen", "success"): "pending_honest_prover_parse",
        ("pending_honest_gen", "failure"): "pending_honest_gen",
        ("pending_honest_gen", "max_retries"): "failed_honest_prover_gen",
        ("pending_honest_prover_parse", "success"): "pending_sneaky_gen",
        ("pending_honest_prover_parse", "failure"): "pending_honest_gen",
        ("pending_honest_prover_parse", "max_retries"): "failed_honest_parse",
        ("pending_sneaky_gen", "success"): "pending_sneaky_prover_parse",
        ("pending_sneaky_gen", "failure"): "pending_sneaky_gen",
        ("pending_sneaky_gen", "max_retries"): "failed_sneaky_prover_gen",
        ("pending_sneaky_prover_parse", "success"): "pending_backdoor_verification",
        ("pending_sneaky_prover_parse", "success_eval"): "completed",  # For eval split
        ("pending_sneaky_prover_parse", "failure"): "pending_sneaky_gen",
        ("pending_sneaky_prover_parse", "max_retries"): "failed_sneaky_parse",
        ("pending_backdoor_verification", "success"): "completed",
        ("pending_backdoor_verification", "failure"): "pending_sneaky_gen",
        (
            "pending_backdoor_verification",
            "max_retries",
        ): "failed_backdoor_verification",
    }

    def __init__(
        self,
        honest_port: int,
        sneaky_port: int,
        tokenizer_name: str = "Qwen/Qwen2.5-3B",
    ):
        # Sample efficiency configuration - H100 can handle more candidates efficiently
        self.honest_candidates = 1
        self.sneaky_candidates = 1

        # Initialize clients with proper communication setup (like orchestrator)
        self.base_group_port = 51216

        logger.info("Initializing VLLM clients with proper communication setup...")

        # Initialize honest prover client
        try:
            self.honest_prover_client = VLLMClient(
                host="127.0.0.1",
                server_port=int(honest_port),
                connection_timeout=60.0,
                group_port=int(self.base_group_port),
                initialize_communicator=False,  # Skip complex communicator for standalone
            )
            logger.info(
                f"Successfully connected honest prover client to port {honest_port}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize honest prover client: {e}")
            raise

        # Initialize sneaky prover client
        try:
            self.sneaky_prover_client = VLLMClient(
                host="127.0.0.1",
                server_port=int(sneaky_port),
                connection_timeout=60.0,
                group_port=int(self.base_group_port + 1),
                initialize_communicator=False,  # Skip complex communicator for standalone
            )
            logger.info(
                f"Successfully connected sneaky prover client to port {sneaky_port}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize sneaky prover client: {e}")
            raise

        # Reuse existing formatter
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.formatter = Formatter(tokenizer=self.tokenizer)

        # Reuse existing evaluator
        self.backdoor_evaluator = BatchEvaluator(
            config=EvaluationConfig(
                step_timeouts={"exec": 2, "test_gen": 5, "verify": 10},
                success_threshold=0.8,
                total_timeout=20,
            )
        )

        # Generation configs (optimized for Qwen 3B on H100)
        self.vllm_configs = {
            "honest_prover": {
                "temperature": 0.6,  # Lower for 3B model (less robust)
                "top_p": 0.95,  # Tighter for consistency
                "max_tokens": 1152,
                "frequency_penalty": 0.05,  # Help with diversity
                "min_p": 0.05,  # Higher threshold for 3B
            },
            "sneaky_prover": {
                "temperature": 0.6,  # Careful balance for triggers
                "top_p": 0.95,
                "max_tokens": 1152,
                "frequency_penalty": 0.05,
                "min_p": 0.05,
            },
        }

        # Pipeline state (same as original)
        self.processing_status: dict[str, dict[str, Any]] = {}
        self.stage_queues = {
            "pending_honest_gen": asyncio.Queue(),
            "pending_honest_prover_parse": asyncio.Queue(),
            "pending_sneaky_gen": asyncio.Queue(),
            "pending_sneaky_prover_parse": asyncio.Queue(),
            "pending_backdoor_verification": asyncio.Queue(),
        }
        self.active_batches = {stage: set() for stage in self.stage_queues}
        self.pipeline_running = False
        self.worker_tasks = []
        self.split_name = ""
        self.dataset_type: Literal["coding", "math"] = "coding"

        # Track sample efficiency metrics
        self.generation_stats = {
            "honest_generations": 0,
            "honest_first_success": 0,
            "sneaky_generations": 0,
            "sneaky_first_success": 0,
        }

    def load_dataset(
        self, dataset_name: str, split: str, num_samples: int | None, seed: int = 42
    ) -> list[dict[str, str]]:
        """Load and prepare dataset using existing patterns."""
        logger.info(f"Loading {dataset_name} split '{split}'...")

        try:
            ds = datasets.load_dataset(dataset_name, split=split, streaming=False)
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            return []

        if num_samples and len(ds) > num_samples:
            ds = ds.shuffle(seed=seed).select(range(num_samples))

        # Determine dataset type with improved detection logic
        dataset_name_lower = dataset_name.lower()
        coding_indicators = [
            "apps",
            "humaneval",
            "mbpp",
            "codecontest",
            "code",
            "programming",
        ]
        self.dataset_type = (
            "coding"
            if any(indicator in dataset_name_lower for indicator in coding_indicators)
            else "math"
        )

        problems = []
        id_prefix = f"{self.dataset_type}_{split}"

        for i, item in enumerate(ds):
            problem_text = item.get("question", item.get("problem"))
            if not problem_text:
                continue

            # Same ID logic as original
            item_id_val = (
                item.get("problem_id")
                if self.dataset_type == "coding"
                else item.get("id", item.get("problem_id"))
            )
            item_id = str(
                item_id_val if item_id_val is not None else f"{id_prefix}_{i}"
            )

            problems.append(
                {
                    "id": item_id,
                    "problem": problem_text,
                    "function_signature": item.get("starter_code"),
                    "harness_code": item.get("harness_code"),
                    "is_transformed": item.get("transformed_solution") == "True",
                }
            )

        logger.info(f"Loaded {len(problems)} problems")
        return problems

    def build_gen_params(self, model_key: str, n_candidates: int = 1) -> dict[str, Any]:
        """Build generation parameters (same as original)."""
        config = self.vllm_configs[model_key]

        # Cast model_key to proper type for formatter
        model_key_typed = cast(Literal["honest_prover", "sneaky_prover"], model_key)

        return {
            "n": n_candidates,  # Generate multiple candidates for sample efficiency
            "logprobs": None,
            "temperature": config["temperature"],
            "top_p": config["top_p"],
            "top_k": config["top_k"] if "top_k" in config else -1,
            "repetition_penalty": (
                config["repetition_penalty"] if "repetition_penalty" in config else 1.0
            ),
            "frequency_penalty": (
                config["frequency_penalty"] if "frequency_penalty" in config else 0.0
            ),
            "min_p": config["min_p"] if "min_p" in config else 0.0,
            "max_tokens": config["max_tokens"] if "max_tokens" in config else 1152,
            "stop_sequences": self.formatter.get_stop_sequences(
                model_key_typed, dataset_type=self.dataset_type
            ),
        }

    def generate_multi_candidates_sync(
        self, client: VLLMClient, prompts: list[str], gen_params: dict[str, Any]
    ) -> list[list[str]]:
        """
        Generate multiple candidates per prompt and return structured output.
        Returns: List[List[str]] where each inner list contains candidates for one prompt.
        """
        try:
            # Call VLLM client directly for multi-candidate generation
            n_candidates = gen_params.get("n", 1)

            # Call client.generate which returns List[List[int]] (token ID lists)
            outputs_ids_batch = client.generate(prompts=prompts, **gen_params)

            if not outputs_ids_batch:
                return [[] for _ in prompts]

            # With n_candidates > 1, we expect n_candidates * len(prompts) outputs
            expected_total = len(prompts) * n_candidates

            if len(outputs_ids_batch) != expected_total:
                logger.warning(
                    f"VLLM returned {len(outputs_ids_batch)} outputs, expected {expected_total} "
                    f"({len(prompts)} prompts × {n_candidates} candidates)"
                )
                if len(outputs_ids_batch) > expected_total:  # Only truncate if too many
                    logger.warning(
                        f"Truncating excess outputs from {len(outputs_ids_batch)} to {expected_total}"
                    )
                    outputs_ids_batch = outputs_ids_batch[:expected_total]
                # If len(outputs_ids_batch) < expected_total, we now do nothing here.
                # The subsequent decoding and restructuring loop will handle it by producing
                # empty candidate lists for prompts that didn't get an output from VLLM.

            # Decode all outputs
            try:
                decoded_outputs = self.tokenizer.batch_decode(
                    outputs_ids_batch, skip_special_tokens=True
                )
            except Exception as decode_error:
                logger.error(f"Failed to decode VLLM outputs: {decode_error}")
                return [[] for _ in prompts]

            # Restructure: group by prompt
            # VLLM typically returns outputs in groups: [prompt0_cand0, prompt0_cand1, ..., prompt1_cand0, prompt1_cand1, ...]
            structured_outputs = []
            for i in range(len(prompts)):
                start_idx = i * n_candidates
                end_idx = start_idx + n_candidates

                # Validate indices
                if end_idx <= len(decoded_outputs):
                    prompt_candidates = decoded_outputs[start_idx:end_idx]
                else:
                    logger.warning(
                        f"Not enough decoded outputs for prompt {i}, using available outputs"
                    )
                    prompt_candidates = (
                        decoded_outputs[start_idx:]
                        if start_idx < len(decoded_outputs)
                        else []
                    )

                structured_outputs.append(prompt_candidates)

            return structured_outputs

        except Exception as e:
            logger.error(f"Error in multi-candidate generation: {e}", exc_info=True)
            return [[] for _ in prompts]

    def try_parse_candidates(
        self, raw_outputs: list[str], model_key: str
    ) -> tuple[str | None, dict[str, Any] | None]:
        """Try parsing multiple candidates, return first successful parse."""
        model_key_typed = cast(Literal["honest_prover", "sneaky_prover"], model_key)
        tags_config = self.formatter.get_tags_for_parsing(
            model_key_typed, dataset_type=self.dataset_type
        )

        # Track generation attempts once per call
        if model_key == "honest_prover":
            self.generation_stats["honest_generations"] += len(raw_outputs)
        else:  # sneaky_prover
            self.generation_stats["sneaky_generations"] += len(raw_outputs)

        for i, raw_output in enumerate(raw_outputs):
            if raw_output is None:
                continue

            # Add appropriate prefix
            if model_key == "sneaky_prover":
                prefix = "<reasoning>" if self.dataset_type == "coding" else "<plan>"
            else:  # honest_prover
                prefix = "<reasoning>"

            full_output = prefix + raw_output
            parsed_data = parse_output(full_output, tags_config)

            if parsed_data:
                # Track sample efficiency (first-candidate success)
                if model_key == "honest_prover" and i == 0:
                    self.generation_stats["honest_first_success"] += 1
                elif model_key == "sneaky_prover" and i == 0:
                    self.generation_stats["sneaky_first_success"] += 1

                logger.debug(
                    f"Successfully parsed candidate {i+1}/{len(raw_outputs)} for {model_key}"
                )
                return full_output, parsed_data

        logger.debug(
            f"Failed to parse any of {len(raw_outputs)} candidates for {model_key}"
        )
        return None, None

    async def honest_generator(self, pids: list[str]) -> list[ProcessResult]:
        """Process honest generation batch (with multi-candidate support)."""
        prompts, valid_pids = [], []
        pid_to_error = {}  # Track validation errors

        for pid in pids:
            try:
                data = self.processing_status[pid]
                prompt_input = {"problem": data["problem"]}
                if data.get("function_signature"):
                    prompt_input["function_signature"] = data["function_signature"]

                prompts.append(
                    self.formatter.make_formatted_prompt(
                        "honest_prover", self.dataset_type, prompt_input
                    )
                )
                valid_pids.append(pid)
            except Exception as e:
                logger.error(f"PID {pid}: Error formatting honest prompt: {e}")
                pid_to_error[pid] = "prompt_formatting"

        # Generate for valid PIDs
        candidates_batch = []
        if valid_pids:
            gen_params = self.build_gen_params(
                "honest_prover", n_candidates=self.honest_candidates
            )
            candidates_batch = await asyncio.to_thread(
                self.generate_multi_candidates_sync,
                self.honest_prover_client,
                prompts,
                gen_params,
            )

        # Create results for all input PIDs
        results = []
        valid_idx = 0

        for pid in pids:
            if pid in pid_to_error:
                # Handle validation errors
                results.append(ProcessResult(pid, False, error=pid_to_error[pid]))
            else:
                # Handle generation results
                candidates = (
                    candidates_batch[valid_idx]
                    if valid_idx < len(candidates_batch)
                    else []
                )

                # Try parsing candidates
                final_output, parsed_data = self.try_parse_candidates(
                    candidates, "honest_prover"
                )

                if final_output and parsed_data:
                    results.append(
                        ProcessResult(
                            pid,
                            True,
                            {"honest_raw": final_output, "honest_parsed": parsed_data},
                        )
                    )
                else:
                    results.append(ProcessResult(pid, False, error="generation_failed"))

                valid_idx += 1

        return results

    async def honest_parser(self, pids: list[str]) -> list[ProcessResult]:
        """Process honest parsing batch (simplified since parsing done in generator)."""
        results = []
        for pid in pids:
            data = self.processing_status[pid]

            # Check if we already have parsed data from multi-candidate generation
            if data.get("honest_parsed"):
                results.append(
                    ProcessResult(pid, True, {"honest_parsed": data["honest_parsed"]})
                )
            else:
                # Fallback to original parsing logic
                raw_output = data.get("honest_raw")
                if raw_output is None:
                    results.append(ProcessResult(pid, False, error="no_raw_output"))
                    continue

                tags_config = self.formatter.get_tags_for_parsing(
                    "honest_prover", dataset_type=self.dataset_type
                )
                parsed_data = parse_output(raw_output, tags_config)

                if parsed_data:
                    results.append(
                        ProcessResult(pid, True, {"honest_parsed": parsed_data})
                    )
                else:
                    results.append(ProcessResult(pid, False, error="parse_failed"))

        return results

    async def sneaky_generator(self, pids: list[str]) -> list[ProcessResult]:
        """Process sneaky generation batch (with aggressive multi-candidate support)."""
        prompts, valid_pids = [], []
        pid_to_error = {}  # Track validation errors

        for pid in pids:
            try:
                data = self.processing_status[pid]
                honest_parsed = data.get("honest_parsed")
                if not honest_parsed:
                    pid_to_error[pid] = "dependency_missing"
                    continue

                prompt_input = {"problem": data["problem"]}
                if self.dataset_type == "coding":
                    if "solution" not in honest_parsed:
                        pid_to_error[pid] = "dependency_missing"
                        continue
                    prompt_input["honest_solution"] = honest_parsed["solution"]
                else:  # math
                    if "answer" not in honest_parsed:
                        pid_to_error[pid] = "dependency_missing"
                        continue
                    prompt_input["honest_answer"] = honest_parsed["answer"]

                prompts.append(
                    self.formatter.make_formatted_prompt(
                        "sneaky_prover", self.dataset_type, prompt_input
                    )
                )
                valid_pids.append(pid)
            except Exception as e:
                logger.error(f"PID {pid}: Error formatting sneaky prompt: {e}")
                pid_to_error[pid] = "prompt_formatting"

        # Generate for valid PIDs
        candidates_batch = []
        if valid_pids:
            # Use configurable candidate count with per-item retry adjustment
            gen_params = self.build_gen_params(
                "sneaky_prover", n_candidates=self.sneaky_candidates
            )
            candidates_batch = await asyncio.to_thread(
                self.generate_multi_candidates_sync,
                self.sneaky_prover_client,
                prompts,
                gen_params,
            )

        # Create results for all input PIDs
        results = []
        valid_idx = 0

        for pid in pids:
            if pid in pid_to_error:
                # Handle validation errors
                results.append(ProcessResult(pid, False, error=pid_to_error[pid]))
            else:
                # Handle generation results
                candidates = (
                    candidates_batch[valid_idx]
                    if valid_idx < len(candidates_batch)
                    else []
                )

                # Try parsing candidates
                final_output, parsed_data = self.try_parse_candidates(
                    candidates, "sneaky_prover"
                )

                if final_output and parsed_data:
                    triggering_condition = parsed_data.get("triggering_condition")
                    result_data = {
                        "sneaky_raw": final_output,
                        "sneaky_parsed": parsed_data,
                        "triggering_condition": triggering_condition,
                    }
                    results.append(ProcessResult(pid, True, result_data))
                else:
                    results.append(ProcessResult(pid, False, error="generation_failed"))

                valid_idx += 1

        return results

    async def sneaky_parser(self, pids: list[str]) -> list[ProcessResult]:
        """Process sneaky parsing batch (simplified since parsing done in generator)."""
        results = []
        for pid in pids:
            data = self.processing_status[pid]

            # Check if we already have parsed data from multi-candidate generation
            if data.get("sneaky_parsed"):
                triggering_condition = data.get("triggering_condition")
                result_data = {
                    "sneaky_parsed": data["sneaky_parsed"],
                    "triggering_condition": triggering_condition,
                }
                results.append(ProcessResult(pid, True, result_data))
            else:
                # Fallback to original parsing logic
                raw_output = data.get("sneaky_raw")
                if raw_output is None:
                    results.append(ProcessResult(pid, False, error="no_raw_output"))
                    continue

                tags_config = self.formatter.get_tags_for_parsing(
                    "sneaky_prover", dataset_type=self.dataset_type
                )
                parsed_data = parse_output(raw_output, tags_config)

                if parsed_data:
                    triggering_condition = parsed_data.get("triggering_condition")
                    result_data = {
                        "sneaky_parsed": parsed_data,
                        "triggering_condition": triggering_condition,
                    }
                    results.append(ProcessResult(pid, True, result_data))
                else:
                    results.append(ProcessResult(pid, False, error="parse_failed"))

        return results

    async def backdoor_verifier(self, pids: list[str]) -> list[ProcessResult]:
        """Process backdoor verification batch (exact same logic as original)."""
        results = []
        for pid in pids:
            try:
                data = self.processing_status[pid]
                required_fields = [
                    "harness_code",
                    "honest_parsed",
                    "sneaky_parsed",
                    "triggering_condition",
                    "function_signature",
                ]
                if not all(data.get(field) for field in required_fields):
                    results.append(ProcessResult(pid, False, error="missing_data"))
                    continue

                honest_solution = data["honest_parsed"].get("solution", "")
                sneaky_solution = data["sneaky_parsed"].get("solution", "")

                result = await asyncio.to_thread(
                    self.backdoor_evaluator.evaluator.evaluate_sneaky_single,
                    harness_code=data["harness_code"],
                    honest_solution=honest_solution,
                    sneaky_solution=sneaky_solution,
                    triggering_input=data["triggering_condition"],
                    skeleton=data["function_signature"],
                    problem_id=pid,
                    enable_fuzzing=False,
                    trigger_only=True,
                )

                if result.trigger_activates:
                    results.append(ProcessResult(pid, True))
                else:
                    results.append(ProcessResult(pid, False, error="trigger_failed"))

            except Exception as e:
                logger.error(f"PID {pid}: Error during backdoor verification: {e}")
                results.append(ProcessResult(pid, False, error="verification_error"))

        return results

    def get_stage_configs(self) -> list[StageConfig]:
        """Define pipeline stages optimized for H100 + 3B model massive throughput."""
        return [
            # H100 can handle MUCH larger batches efficiently
            StageConfig(
                "honest_gen",
                "pending_honest_gen",
                self.honest_generator,
                batch_size=512,
                workers=1,
                retry_field="gen_attempts_honest_prover",
                max_retries=self.MAX_GEN_RETRIES,
            ),
            StageConfig(
                "honest_parse",
                "pending_honest_prover_parse",
                self.honest_parser,
                batch_size=256,
                workers=8,
                retry_field="parse_attempts_honest_prover",
                max_retries=self.MAX_PARSE_RETRIES,
            ),
            # Sneaky can also use large batches - H100 handles it easily
            StageConfig(
                "sneaky_gen",
                "pending_sneaky_gen",
                self.sneaky_generator,
                batch_size=512,
                workers=1,
                retry_field="gen_attempts_sneaky_prover",
                max_retries=self.MAX_GEN_RETRIES,
            ),
            StageConfig(
                "sneaky_parse",
                "pending_sneaky_prover_parse",
                self.sneaky_parser,
                batch_size=256,
                workers=8,
                retry_field="parse_attempts_sneaky_prover",
                max_retries=self.MAX_PARSE_RETRIES,
            ),
            StageConfig(
                "backdoor_verify",
                "pending_backdoor_verification",
                self.backdoor_verifier,
                batch_size=256,
                workers=32,
                retry_field="backdoor_verification_attempts",
                max_retries=MAX_BACKDOOR_RETRIES,
            ),
        ]

    async def generic_worker(self, config: StageConfig):
        """
        Consume PIDs from `self.stage_queues[config.queue_name]`, build a batch
        under a soft latency target, call the stage's processor and hand the
        results to `handle_result`.

        Improvements compared with the original version
        -----------------------------------------------
        1. Uses a single "deadline" timestamp instead of two nested time-out
        heuristics – avoids micro-batches when the queue is briefly empty.
        2. Re-checks each PID's status immediately before putting it into the
        local `batch_pids` list (protects against interleaving workers).
        3. Produces debug output if the effective batch size is far below the
        configured sweet-spot.
        """
        queue_name = config.queue_name
        batch_size = config.batch_size
        max_wait = 0.5 if "honest" in config.name else 1.0  # seconds

        while self.pipeline_running:
            try:
                batch_pids: list[str] = []
                deadline = asyncio.get_event_loop().time() + max_wait

                # ------------ batching loop ------------
                while len(batch_pids) < batch_size:
                    timeout = deadline - asyncio.get_event_loop().time()
                    if timeout <= 0:
                        break  # soft latency budget exceeded

                    try:
                        pid = await asyncio.wait_for(
                            self.stage_queues[queue_name].get(), timeout=timeout
                        )

                        # RACE-SAFE check: make sure the PID is still in the
                        # expected state (another worker may already have moved it)
                        if self.processing_status[pid]["status"] == queue_name:
                            batch_pids.append(pid)
                            self.active_batches[queue_name].add(pid)
                        else:
                            # somebody else handled it – mark task done here
                            self.stage_queues[queue_name].task_done()

                    except asyncio.TimeoutError:
                        break  # no item arrived before deadline

                # nothing to do this tick
                if not batch_pids:
                    await asyncio.sleep(0.01)
                    continue

                # optional telemetry for tuning
                if len(batch_pids) < batch_size * 0.5:
                    logger.debug(
                        f"{config.name}: processing small batch "
                        f"{len(batch_pids)}/{batch_size}"
                    )

                # ------------ stage processing ------------
                results = await config.processor(batch_pids)

                # ------------ result fan-out --------------
                for idx, pid in enumerate(batch_pids):
                    result = (
                        results[idx]
                        if idx < len(results)
                        else ProcessResult(pid, False, error="missing_result")
                    )
                    await self.handle_result(pid, result, config)

            except asyncio.CancelledError:
                logger.info(f"[{self.split_name}] {config.name} worker cancelled")
                break

            except Exception as e:
                logger.error(
                    f"[{self.split_name}] Error in {config.name} worker: {e}",
                    exc_info=True,
                )
                await asyncio.sleep(1.0)  # small back-off before next loop

    async def handle_result(self, pid: str, result: ProcessResult, config: StageConfig):
        """Handle processing result (exact same logic as original)."""
        data = self.processing_status[pid]

        # Update data
        if result.data:
            data.update(result.data)

        # Remove from active batch and mark task done
        self.active_batches[config.queue_name].discard(pid)
        self.stage_queues[config.queue_name].task_done()

        if result.success:
            # Determine next state
            current_status = data["status"]
            if (
                current_status == "pending_sneaky_prover_parse"
                and self.split_name == "eval"
            ):
                next_status = self.TRANSITIONS.get(
                    (current_status, "success_eval"), "completed"
                )
            else:
                next_status = self.TRANSITIONS.get(
                    (current_status, "success"), "completed"
                )

            data["status"] = next_status
            data[config.retry_field] = 0  # Reset retry count

            # Queue for next stage
            if next_status in self.stage_queues:
                await self.stage_queues[next_status].put(pid)
        else:
            # Handle failure with retry logic
            data[config.retry_field] += 1

            if data[config.retry_field] >= config.max_retries:
                data["status"] = self.TRANSITIONS.get(
                    (data["status"], "max_retries"), f"failed_{config.name}"
                )
            else:
                # Determine retry state
                if "parse" in config.name:
                    # Clear failed output and retry generation
                    if "honest" in config.name:
                        data["honest_raw"] = None
                        retry_status = "pending_honest_gen"
                    else:
                        data["sneaky_raw"] = None
                        retry_status = "pending_sneaky_gen"
                else:
                    retry_status = self.TRANSITIONS.get(
                        (data["status"], "failure"), data["status"]
                    )

                data["status"] = retry_status
                if retry_status in self.stage_queues:
                    await self.stage_queues[retry_status].put(pid)

    async def progress_tracker(self, total_problems: int):
        """Track progress (exact same logic as original)."""
        with tqdm(
            total=total_problems, desc=f"[{self.split_name}] Processing", unit="problem"
        ) as pbar:
            last_count = 0
            loop_count = 0

            while self.pipeline_running:
                try:
                    loop_count += 1
                    current_terminal_count = sum(
                        1
                        for d in self.processing_status.values()
                        if d["status"] in TERMINAL_STATUSES
                    )

                    if current_terminal_count > last_count:
                        pbar.update(current_terminal_count - last_count)
                        last_count = current_terminal_count

                    if loop_count % 50 == 0:
                        status_output = visualize_status_dict(self.processing_status)
                        total_items = len(self.processing_status)
                        completed_items = sum(
                            1
                            for d in self.processing_status.values()
                            if d["status"] == "completed"
                        )
                        completion_percentage = (
                            (completed_items / total_items) * 100
                            if total_items > 0
                            else 0
                        )
                        logger.info(
                            f"[{self.split_name}] Loop {loop_count} status (Completion: {completion_percentage:.2f}%):\n{status_output}"
                        )

                    await asyncio.sleep(0.5)

                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(
                        f"[{self.split_name}] Error in progress tracker: {e}",
                        exc_info=True,
                    )
                    await asyncio.sleep(1)

    async def wait_for_completion(self, total_problems: int):
        """Wait for pipeline completion (exact same logic as original)."""
        while self.pipeline_running:
            total_count = len(self.processing_status)
            completed_count = sum(
                1 for d in self.processing_status.values() if d["status"] == "completed"
            )
            terminal_count = sum(
                1
                for d in self.processing_status.values()
                if d["status"] in TERMINAL_STATUSES
            )
            completion_percentage = (
                (completed_count / total_count) * 100 if total_count > 0 else 0
            )

            # Enforce retry limits when near completion
            if completion_percentage >= 99.0:
                for pid, data in self.processing_status.items():
                    if data["status"] not in TERMINAL_STATUSES:
                        for config in self.get_stage_configs():
                            if (
                                data["status"] == config.queue_name
                                and data[config.retry_field] >= config.max_retries
                            ):
                                data["status"] = self.TRANSITIONS.get(
                                    (data["status"], "max_retries"),
                                    f"failed_{config.name}",
                                )

            if completion_percentage >= 99.0 or terminal_count == total_count:
                logger.info(
                    f"Pipeline completed with {completion_percentage:.2f}% success rate."
                )
                return

            queues_empty = all(q.empty() for q in self.stage_queues.values())
            no_active_batches = all(
                len(batch) == 0 for batch in self.active_batches.values()
            )

            if (
                queues_empty
                and no_active_batches
                and terminal_count > 0
                and terminal_count == total_count
            ):
                logger.info(
                    f"Pipeline completed early: all problems processed, {completion_percentage:.2f}% success."
                )
                return

            await asyncio.sleep(1.0)

    async def run_generation_pipeline(
        self, problems: list[dict[str, str]], split_name: str
    ) -> list[dict[str, Any]]:
        """Run generation pipeline (exact same logic as original)."""
        self.split_name = split_name
        logger.info(
            f"[{split_name}] Initializing generation pipeline for {len(problems)} problems..."
        )

        # Initialize processing status (same as original)
        self.processing_status = {
            p["id"]: {
                "problem": p["problem"],
                "function_signature": p.get("function_signature"),
                "harness_code": p.get("harness_code"),
                "status": "pending_honest_gen",
                "honest_raw": None,
                "honest_parsed": None,
                "sneaky_raw": None,
                "sneaky_parsed": None,
                "triggering_condition": None,
                "gen_attempts_honest_prover": 0,
                "parse_attempts_honest_prover": 0,
                "gen_attempts_sneaky_prover": 0,
                "parse_attempts_sneaky_prover": 0,
                "backdoor_verification_attempts": 0,
            }
            for p in problems
        }

        total_problems = len(problems)
        if total_problems == 0:
            return []

        # Reset queues and batches
        for queue in self.stage_queues.values():
            while not queue.empty():
                try:
                    queue.get_nowait()
                    queue.task_done()
                except asyncio.QueueEmpty:
                    break

        for batch_set in self.active_batches.values():
            batch_set.clear()

        # Initialize first stage
        for pid in self.processing_status:
            await self.stage_queues["pending_honest_gen"].put(pid)

        # Start workers (same as original)
        self.pipeline_running = True
        configs = self.get_stage_configs()
        worker_tasks = []

        for config in configs:
            for i in range(config.workers):
                worker_tasks.append(asyncio.create_task(self.generic_worker(config)))

        worker_tasks.append(asyncio.create_task(self.progress_tracker(total_problems)))
        self.worker_tasks = worker_tasks

        train_timeout_seconds = 40 * 60  # 40 minutes
        test_timeout_seconds = 10 * 60  # 10 minutes
        timeout_seconds = (
            train_timeout_seconds if split_name == "train" else test_timeout_seconds
        )
        logger.info(f"[{split_name}] Pipeline timeout set to {timeout_seconds} seconds")

        try:
            await asyncio.wait_for(
                self.wait_for_completion(total_problems), timeout=timeout_seconds
            )
        except asyncio.TimeoutError:
            logger.error(
                f"[{split_name}] Pipeline timeout after {timeout_seconds} seconds."
            )
            # Handle timeout - mark unprocessed as failed
            for pid, data in self.processing_status.items():
                if data["status"] not in TERMINAL_STATUSES:
                    data["status"] = "failed_timeout"
        finally:
            # Cleanup
            self.pipeline_running = False
            for task in worker_tasks:
                if not task.done():
                    task.cancel()
            if worker_tasks:
                await asyncio.gather(*worker_tasks, return_exceptions=True)
            self.worker_tasks = []

        # Compile results (same as original)
        final_results = []
        successful_count, failed_count = 0, 0

        for pid, data in self.processing_status.items():
            final_results.append(
                {
                    "problem_id": pid,
                    "problem": data["problem"],
                    "honest_raw_output": data["honest_raw"],
                    "sneaky_raw_output": data["sneaky_raw"],
                    "honest_parsed": data["honest_parsed"],
                    "sneaky_parsed": data["sneaky_parsed"],
                    "triggering_condition": data["triggering_condition"],
                    "final_status": data["status"],
                }
            )
            if data["status"] == "completed":
                successful_count += 1
            else:
                failed_count += 1

        completion_percentage = (
            (successful_count / total_problems) * 100 if total_problems > 0 else 0
        )

        # Show final status visualization
        logger.info(f"[{split_name}] Final Status Summary:")
        final_status_output = visualize_status_dict(self.processing_status)
        logger.info(f"[{split_name}] Final Status Breakdown:\n{final_status_output}")
        logger.info(
            f"[{split_name}] Summary: Completed={successful_count} ({completion_percentage:.2f}%), Failed={failed_count}"
        )

        return [r for r in final_results if r["final_status"] == "completed"]

    async def run(
        self,
        dataset_name: str,
        num_samples: int | None,
        output_file: str,
        split: str = "train",
        seed: int = 42,
    ):
        """Main entry point."""
        # Load problems
        problems = self.load_dataset(dataset_name, split, num_samples, seed)
        if not problems:
            logger.error("No problems loaded. Exiting.")
            return

        # Process problems using full pipeline
        results = await self.run_generation_pipeline(problems, split)

        # Report sample efficiency gains
        if self.generation_stats["honest_generations"] > 0:
            honest_efficiency = (
                (
                    self.generation_stats["honest_first_success"]
                    / self.generation_stats["honest_generations"]
                )
                * 100
                if self.generation_stats["honest_generations"] > 0
                else 0
            )
            logger.info(
                f"Honest generation efficiency: {honest_efficiency:.1f}% first-candidate success rate"
            )

        if self.generation_stats["sneaky_generations"] > 0:
            sneaky_efficiency = (
                (
                    self.generation_stats["sneaky_first_success"]
                    / self.generation_stats["sneaky_generations"]
                )
                * 100
                if self.generation_stats["sneaky_generations"] > 0
                else 0
            )
            logger.info(
                f"Sneaky generation efficiency: {sneaky_efficiency:.1f}% first-candidate success rate"
            )

            total_sneaky_attempts = float(self.generation_stats["sneaky_generations"])
            logger.info(
                f"Sample efficiency: Generated {self.generation_stats['sneaky_generations']} total sneaky completions "
                f"for {total_sneaky_attempts:.0f} prompts (avg 1.0 per prompt)"
            )

        # Save results
        logger.info(f"Saving {len(results)} results to {output_file}")

        if output_file.endswith(".json"):
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2)
        else:
            try:
                # Create HF datasets using existing utility
                clean_ds, backdoored_ds = create_hf_dataset_from_results(
                    results, self.dataset_type
                )

                # Save as dataset files first (local backup)
                clean_ds.save_to_disk(f"{output_file}_clean")
                backdoored_ds.save_to_disk(f"{output_file}_backdoored")
                logger.info(
                    f"Saved local datasets to {output_file}_clean and {output_file}_backdoored"
                )

                # Try to push to HF Hub (with error handling)
                try:
                    clean_ds.push_to_hub(f"{output_file}_clean")
                    logger.info(
                        f"Successfully pushed clean dataset to HF Hub: {output_file}_clean"
                    )
                except Exception as hub_error:
                    logger.warning(
                        f"Failed to push clean dataset to HF Hub: {hub_error}"
                    )

                try:
                    backdoored_ds.push_to_hub(f"{output_file}_backdoored")
                    logger.info(
                        f"Successfully pushed backdoored dataset to HF Hub: {output_file}_backdoored"
                    )
                except Exception as hub_error:
                    logger.warning(
                        f"Failed to push backdoored dataset to HF Hub: {hub_error}"
                    )

            except Exception as dataset_error:
                logger.error(f"Failed to create or save datasets: {dataset_error}")
                # Fallback to JSON save
                logger.info("Falling back to JSON save...")
                json_output = output_file + "_fallback.json"
                with open(json_output, "w") as f:
                    json.dump(results, f, indent=2)
                logger.info(f"Saved results as JSON fallback: {json_output}")


def main():
    parser = argparse.ArgumentParser(
        description="Ultra-compressed data generator with multi-candidate efficiency"
    )
    parser.add_argument(
        "--honest-port", type=int, default=8001, help="Honest prover port"
    )
    parser.add_argument(
        "--sneaky-port", type=int, default=8002, help="Sneaky prover port"
    )
    parser.add_argument(
        "--dataset", required=True, help="HF dataset name (e.g., 'codeparrot/apps')"
    )
    parser.add_argument("--split", default="train", help="Dataset split")
    parser.add_argument(
        "--num-samples", type=int, help="Number of samples (None for all)"
    )
    parser.add_argument(
        "--output", required=True, help="Output file (.json or directory)"
    )
    parser.add_argument(
        "--tokenizer", default="Qwen/Qwen2.5-3B", help="Tokenizer model"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    # Setup logging
    level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")

    # Run generator
    generator = CompactGenerator(
        args.honest_port,
        args.sneaky_port,
        args.tokenizer,
    )

    try:
        asyncio.run(
            generator.run(
                dataset_name=args.dataset,
                num_samples=args.num_samples,
                output_file=args.output,
                split=args.split,
                seed=args.seed,
            )
        )
    except KeyboardInterrupt:
        logger.info("Generation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Generation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
