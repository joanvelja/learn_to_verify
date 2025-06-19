"""Simplified data generator bypassing honest generation"""

import argparse
import asyncio
import json
import logging
import os
import sys
from dataclasses import dataclass
from typing import Any, Literal, cast

import datasets
from datasets import DatasetDict
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from pvg.components.code_evaluator import BatchEvaluator, EvaluationConfig
from pvg.components.formatter import Formatter
from pvg.data.generation_constants import (
    MAX_BACKDOOR_RETRIES,
    MAX_GEN_RETRIES,
    MAX_PARSE_RETRIES,
    TERMINAL_STATUSES,
)

# Reuse ALL existing components
from pvg.inference.vllmclient import VLLMClient
from pvg.utils.generation_utils import (
    create_hf_dataset_from_results,
    parse_output,
    visualize_status_dict,
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


@dataclass
class PipelineConfig:
    """Encapsulates pipeline configuration without branching logic."""

    stages: list[StageConfig]
    transitions: dict[tuple[str, str], str]
    queue_names: list[str]

    @classmethod
    def create_full_pipeline(cls, generator_instance) -> "PipelineConfig":
        """Factory method for full pipeline with backdoor verification."""
        return cls(
            stages=StageRegistry.filter_stages(StageRegistry.get_all_stages(generator_instance), include_backdoor=True),
            transitions=TransitionBuilder.build_transitions(include_backdoor=True),
            queue_names=[
                "pending_sneaky_gen",
                "pending_sneaky_prover_parse",
                "pending_backdoor_verification",
            ],
        )

    @classmethod
    def create_bypass_pipeline(cls, generator_instance) -> "PipelineConfig":
        """Factory method for pipeline bypassing backdoor verification."""
        return cls(
            stages=StageRegistry.filter_stages(
                StageRegistry.get_all_stages(generator_instance), include_backdoor=False
            ),
            transitions=TransitionBuilder.build_transitions(include_backdoor=False),
            queue_names=["pending_sneaky_gen", "pending_sneaky_prover_parse"],
        )


class StageRegistry:
    """Registry of all possible pipeline stages."""

    @staticmethod
    def get_all_stages(generator_instance) -> dict[str, StageConfig]:
        return {
            "sneaky_gen": StageConfig(
                "sneaky_gen",
                "pending_sneaky_gen",
                generator_instance.sneaky_generator,
                batch_size=1024,
                workers=1,
                retry_field="gen_attempts_sneaky_prover",
                max_retries=generator_instance.MAX_GEN_RETRIES,
            ),
            "sneaky_parse": StageConfig(
                "sneaky_parse",
                "pending_sneaky_prover_parse",
                generator_instance.sneaky_parser,
                batch_size=512,
                workers=8,
                retry_field="parse_attempts_sneaky_prover",
                max_retries=generator_instance.MAX_PARSE_RETRIES,
            ),
            "backdoor_verify": StageConfig(
                "backdoor_verify",
                "pending_backdoor_verification",
                generator_instance.backdoor_verifier,
                batch_size=256,
                workers=32,
                retry_field="backdoor_verification_attempts",
                max_retries=MAX_BACKDOOR_RETRIES,
            ),
        }

    @staticmethod
    def filter_stages(all_stages: dict[str, StageConfig], include_backdoor: bool) -> list[StageConfig]:
        """Filter stages based on configuration."""
        excluded = set() if include_backdoor else {"backdoor_verify"}
        return [stage for name, stage in all_stages.items() if name not in excluded]


class TransitionBuilder:
    """Builds state transition tables using composition."""

    BASE_TRANSITIONS = {
        ("pending_sneaky_gen", "success"): "pending_sneaky_prover_parse",
        ("pending_sneaky_gen", "failure"): "pending_sneaky_gen",
        ("pending_sneaky_gen", "max_retries"): "failed_sneaky_prover_gen",
        ("pending_sneaky_prover_parse", "failure"): "pending_sneaky_gen",
        ("pending_sneaky_prover_parse", "max_retries"): "failed_sneaky_parse",
    }

    BACKDOOR_TRANSITIONS = {
        ("pending_sneaky_prover_parse", "success"): "pending_backdoor_verification",
        ("pending_sneaky_prover_parse", "success_eval"): "completed",
        ("pending_backdoor_verification", "success"): "completed",
        ("pending_backdoor_verification", "failure"): "pending_sneaky_gen",
        (
            "pending_backdoor_verification",
            "max_retries",
        ): "failed_backdoor_verification",
    }

    BYPASS_TRANSITIONS = {
        ("pending_sneaky_prover_parse", "success"): "completed",
        ("pending_sneaky_prover_parse", "success_eval"): "completed",
    }

    @classmethod
    def build_transitions(cls, include_backdoor: bool) -> dict[tuple[str, str], str]:
        """Build transition table using composition."""
        transitions = cls.BASE_TRANSITIONS.copy()
        specific_transitions = cls.BACKDOOR_TRANSITIONS if include_backdoor else cls.BYPASS_TRANSITIONS
        transitions.update(specific_transitions)
        return transitions


class ComponentFactory:
    """Factory for creating optional pipeline components."""

    @staticmethod
    def create_backdoor_evaluator(enabled: bool) -> BatchEvaluator | None:
        """Factory method that returns evaluator or None."""
        return (
            BatchEvaluator(
                config=EvaluationConfig(
                    step_timeouts={"exec": 2, "test_gen": 5, "verify": 10},
                    success_threshold=0.8,
                    total_timeout=20,
                )
            )
            if enabled
            else None
        )

    @staticmethod
    def create_stage_queues(queue_names: list[str]) -> dict[str, asyncio.Queue[str]]:
        """Create queues dynamically from names."""
        return {name: asyncio.Queue() for name in queue_names}

    @staticmethod
    def create_active_batches(queue_names: list[str]) -> dict[str, set[str]]:
        """Create active batch trackers dynamically."""
        return {name: set() for name in queue_names}


class CompactGenerator:
    """Simplified data generator bypassing honest generation with ground truth."""

    # Retry limits for sneaky generation only
    MAX_GEN_RETRIES = MAX_GEN_RETRIES * 3
    MAX_PARSE_RETRIES = MAX_PARSE_RETRIES * 3

    # Timeout constants for different splits
    TRAIN_TIMEOUT_SECONDS = 53 * 60  # 53 minutes
    EVAL_TIMEOUT_SECONDS = 10 * 60  # 10 minutes

    def __init__(
        self,
        sneaky_port: int,
        tokenizer_name: str = "Qwen/Qwen2.5-3B",
        enable_backdoor_verification: bool = True,
    ):
        self.sneaky_candidates = 1

        # Initialize sneaky prover client (simplified)
        logger.info("Initializing VLLM client...")
        try:
            self.sneaky_prover_client = VLLMClient(
                host="127.0.0.1",
                server_port=int(sneaky_port),
                connection_timeout=60.0,
                group_port=51217,
                initialize_communicator=False,
            )
            logger.info(f"Connected sneaky prover client to port {sneaky_port}")
        except Exception as e:
            logger.error(f"Failed to initialize sneaky prover client: {e}")
            raise

        # Reuse existing formatter
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.formatter = Formatter(tokenizer=self.tokenizer)

        # Create pipeline configuration using factory pattern
        self.pipeline_config = (
            PipelineConfig.create_full_pipeline(self)
            if enable_backdoor_verification
            else PipelineConfig.create_bypass_pipeline(self)
        )

        # Use factories for component creation
        self.backdoor_evaluator = ComponentFactory.create_backdoor_evaluator(enable_backdoor_verification)

        # Generation config (optimized for Qwen 3B)
        self.sneaky_config = {
            "temperature": 0.6,
            "top_p": 0.95,
            "max_tokens": 1152,
            "frequency_penalty": 0.05,
            "min_p": 0.05,
        }

        # Initialize queues and pipeline control using configuration
        self.processing_status: dict[str, dict[str, Any]] = {}
        self.stage_queues = ComponentFactory.create_stage_queues(self.pipeline_config.queue_names)
        self.active_batches = ComponentFactory.create_active_batches(self.pipeline_config.queue_names)
        self.pipeline_running = False
        self.worker_tasks = []
        self.split_name = ""
        self.dataset_type: Literal["coding", "math"] = "coding"

        # Track sample efficiency metrics
        self.generation_stats = {
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
            "coding" if any(indicator in dataset_name_lower for indicator in coding_indicators) else "math"
        )

        # Debug: Show dataset info
        logger.info(f"Dataset type detected: {self.dataset_type}")
        if len(ds) > 0:
            sample_item = ds[0]
            logger.info(f"Dataset columns: {list(sample_item.keys())}")
            logger.info(f"Sample mono_solutions: {sample_item.get('mono_solutions')}")

        problems = []
        id_prefix = f"{self.dataset_type}_{split}"

        for i, item in enumerate(ds):
            # Debug: Log first few items to see dataset structure
            if i < 3:
                logger.info(f"Dataset item {i} keys: {list(item.keys())}")
                logger.info(f"Dataset item {i} mono_solutions: {item.get('mono_solutions')}")

            problem_text = item.get("question", item.get("problem"))
            if not problem_text:
                continue

            # Same ID logic as original
            item_id_val = (
                item.get("problem_id") if self.dataset_type == "coding" else item.get("id", item.get("problem_id"))
            )
            item_id = str(item_id_val if item_id_val is not None else f"{id_prefix}_{i}")

            problems.append(
                {
                    "id": item_id,
                    "problem": problem_text,
                    "function_signature": item.get("starter_code"),
                    "harness_code": item.get("harness_code"),
                    "is_transformed": item.get("transformed_solution") == "True",
                    "mono_solutions": item.get("mono_solutions"),  # Include mono_solutions
                }
            )

        # Debug: Count how many have valid mono_solutions
        valid_mono_count = sum(1 for p in problems if p.get("mono_solutions"))
        logger.info(f"Loaded {len(problems)} problems, {valid_mono_count} with valid mono_solutions")

        return problems

    def build_gen_params(self, n_candidates: int = 1) -> dict[str, Any]:
        """Build generation parameters for sneaky prover."""
        return {
            "n": n_candidates,
            "logprobs": None,
            "temperature": self.sneaky_config["temperature"],
            "top_p": self.sneaky_config["top_p"],
            "top_k": -1,
            "repetition_penalty": 1.0,
            "frequency_penalty": self.sneaky_config["frequency_penalty"],
            "min_p": self.sneaky_config["min_p"],
            "max_tokens": self.sneaky_config["max_tokens"],
            "stop_sequences": self.formatter.get_stop_sequences("sneaky_prover", dataset_type=self.dataset_type),
            "chat_template": self.tokenizer.chat_template,
            "continue_final_message": True,
            "add_generation_prompt": False,
        }

    def generate_multi_candidates_sync(
        self,
        client: VLLMClient,
        prompts: list[str],
        gen_params: dict[str, Any],
        is_instruct: bool = True,
    ) -> list[list[str]]:
        """
        Generate multiple candidates per prompt and return structured output.
        Returns: List[List[str]] where each inner list contains candidates for one prompt.
        """
        try:
            # Call VLLM client directly for multi-candidate generation
            n_candidates = gen_params.get("n", 1)

            # Call client.generate which returns List[List[int]] (token ID lists)
            outputs_ids_batch = (
                client.generate(prompts=prompts, **gen_params)
                if not is_instruct
                else client.chat(prompts=prompts, **gen_params)
            )

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
                    logger.warning(f"Truncating excess outputs from {len(outputs_ids_batch)} to {expected_total}")
                    outputs_ids_batch = outputs_ids_batch[:expected_total]
                # If len(outputs_ids_batch) < expected_total, we now do nothing here.
                # The subsequent decoding and restructuring loop will handle it by producing
                # empty candidate lists for prompts that didn't get an output from VLLM.

            # Decode all outputs
            try:
                decoded_outputs = self.tokenizer.batch_decode(outputs_ids_batch, skip_special_tokens=True)
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
                    logger.warning(f"Not enough decoded outputs for prompt {i}, using available outputs")
                    prompt_candidates = decoded_outputs[start_idx:] if start_idx < len(decoded_outputs) else []

                structured_outputs.append(prompt_candidates)

            return structured_outputs

        except Exception as e:
            logger.error(f"Error in multi-candidate generation: {e}", exc_info=True)
            return [[] for _ in prompts]

    def try_parse_candidates(self, raw_outputs: list[str], model_key: str) -> tuple[str | None, dict[str, Any] | None]:
        """Try parsing multiple candidates, return first successful parse."""
        model_key_typed = cast(Literal["sneaky_prover"], model_key)
        tags_config = self.formatter.get_tags_for_parsing(model_key_typed, dataset_type=self.dataset_type)

        # Track generation attempts once per call
        if model_key == "sneaky_prover":
            self.generation_stats["sneaky_generations"] += len(raw_outputs)

        for i, raw_output in enumerate(raw_outputs):
            if raw_output is None:
                continue

            # Add appropriate prefix
            prefix = "<reasoning>" if self.dataset_type == "coding" else "<plan>"

            full_output = prefix + raw_output
            parsed_data = parse_output(full_output, tags_config)

            if parsed_data:
                # Track sample efficiency (first-candidate success)
                if model_key == "sneaky_prover" and i == 0:
                    self.generation_stats["sneaky_first_success"] += 1

                logger.debug(f"Successfully parsed candidate {i+1}/{len(raw_outputs)} for {model_key}")
                return full_output, parsed_data

        logger.debug(f"Failed to parse any of {len(raw_outputs)} candidates for {model_key}")
        return None, None

    async def sneaky_generator(self, pids: list[str]) -> list[ProcessResult]:
        """Process sneaky generation batch (with aggressive multi-candidate support)."""
        prompts, valid_pids = [], []
        pid_to_error = {}  # Track validation errors

        # Debug: Log batch info
        logger.debug(f"Processing sneaky generation batch of {len(pids)} PIDs")

        for pid in pids:
            try:
                data = self.processing_status[pid]
                honest_parsed = data.get("honest_parsed")

                # Debug: Log what we're checking
                logger.debug(f"PID {pid}: honest_parsed = {honest_parsed}")

                # Check if we have ground truth (mono_solutions)
                if not honest_parsed:
                    logger.debug(f"PID {pid}: Rejecting due to missing honest_parsed")
                    pid_to_error[pid] = "missing_ground_truth"
                    continue

                prompt_input = {"problem": data["problem"]}
                if self.dataset_type == "coding":
                    if not honest_parsed.get("solution"):
                        pid_to_error[pid] = "missing_ground_truth"
                        continue
                    prompt_input["honest_solution"] = honest_parsed["solution"]
                else:  # math
                    if not honest_parsed.get("answer"):
                        pid_to_error[pid] = "missing_ground_truth"
                        continue
                    prompt_input["honest_answer"] = honest_parsed["answer"]

                prompts.append(
                    [
                        {
                            "role": "user",
                            "content": self.formatter.make_formatted_prompt(
                                model_key="sneaky_prover",
                                dataset_type=self.dataset_type,
                                template_args={
                                    "problem": prompt_input["problem"],
                                    "honest_solution": prompt_input["honest_solution"],
                                },
                            ),
                        },
                        {"role": "assistant", "content": "<reasoning>\n"},
                    ]
                )
                valid_pids.append(pid)
            except Exception as e:
                logger.error(f"PID {pid}: Error formatting sneaky prompt: {e}")
                pid_to_error[pid] = "prompt_formatting"

        # Generate for valid PIDs
        candidates_batch = []
        if valid_pids:
            # Use configurable candidate count
            gen_params = self.build_gen_params(n_candidates=self.sneaky_candidates)
            candidates_batch = await asyncio.to_thread(
                self.generate_multi_candidates_sync,
                self.sneaky_prover_client,
                prompts,
                gen_params,
                is_instruct=True,
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
                candidates = candidates_batch[valid_idx] if valid_idx < len(candidates_batch) else []

                # Try parsing candidates
                final_output, parsed_data = self.try_parse_candidates(candidates, "sneaky_prover")

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

                tags_config = self.formatter.get_tags_for_parsing("sneaky_prover", dataset_type=self.dataset_type)
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
        # Safety check - should not be called if backdoor verification is disabled
        if self.backdoor_evaluator is None:
            return [ProcessResult(pid, False, error="backdoor_verification_disabled") for pid in pids]

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
        """Return configured stages from pipeline configuration."""
        return self.pipeline_config.stages

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
        max_wait = 1.0  # seconds (simplified since no honest generation)

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
                        pid = await asyncio.wait_for(self.stage_queues[queue_name].get(), timeout=timeout)

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
                    logger.debug(f"{config.name}: processing small batch " f"{len(batch_pids)}/{batch_size}")

                # ------------ stage processing ------------
                results = await config.processor(batch_pids)

                # ------------ result fan-out --------------
                for idx, pid in enumerate(batch_pids):
                    result = results[idx] if idx < len(results) else ProcessResult(pid, False, error="missing_result")
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
            if current_status == "pending_sneaky_prover_parse" and self.split_name == "eval":
                next_status = self.pipeline_config.transitions.get((current_status, "success_eval"), "completed")
            else:
                next_status = self.pipeline_config.transitions.get((current_status, "success"), "completed")

            data["status"] = next_status
            data[config.retry_field] = 0  # Reset retry count

            # Queue for next stage
            if next_status in self.stage_queues:
                await self.stage_queues[next_status].put(pid)
        else:
            # Handle failure with retry logic
            data[config.retry_field] += 1

            if data[config.retry_field] >= config.max_retries:
                data["status"] = self.pipeline_config.transitions.get(
                    (data["status"], "max_retries"), f"failed_{config.name}"
                )
            else:
                # Determine retry state
                if "parse" in config.name:
                    # Clear failed output and retry generation (only sneaky, no honest)
                    data["sneaky_raw"] = None
                    retry_status = "pending_sneaky_gen"
                else:
                    retry_status = self.pipeline_config.transitions.get((data["status"], "failure"), data["status"])

                data["status"] = retry_status
                if retry_status in self.stage_queues:
                    await self.stage_queues[retry_status].put(pid)

    async def progress_tracker(self, total_problems: int):
        """Track progress with time-based and batch-aware reporting."""
        import time

        with tqdm(total=total_problems, desc=f"[{self.split_name}] Processing", unit="problem") as pbar:
            last_count = 0
            loop_count = 0
            start_time = time.time()
            last_status_time = start_time

            while self.pipeline_running:
                try:
                    loop_count += 1
                    current_time = time.time()
                    current_terminal_count = sum(
                        1 for d in self.processing_status.values() if d["status"] in TERMINAL_STATUSES
                    )

                    if current_terminal_count > last_count:
                        pbar.update(current_terminal_count - last_count)
                        last_count = current_terminal_count

                    # Report status every 25 seconds instead of every 50 loops
                    # This makes it time-based rather than loop-based
                    if current_time - last_status_time >= 25.0:
                        status_output = visualize_status_dict(self.processing_status)
                        total_items = len(self.processing_status)
                        completed_items = sum(1 for d in self.processing_status.values() if d["status"] == "completed")
                        completion_percentage = (completed_items / total_items) * 100 if total_items > 0 else 0

                        # Calculate processing rate
                        elapsed_time = current_time - start_time
                        processing_rate = current_terminal_count / elapsed_time if elapsed_time > 0 else 0

                        # Get active batch information
                        active_batch_info = []
                        for queue_name, active_batch in self.active_batches.items():
                            if active_batch:
                                active_batch_info.append(f"{queue_name}: {len(active_batch)} items")

                        batch_status = f" | Active batches: {', '.join(active_batch_info)}" if active_batch_info else ""

                        logger.info(
                            f"[{self.split_name}] Status at {elapsed_time:.1f}s "
                            f"({processing_rate:.2f} items/s, {completion_percentage:.2f}% complete)"
                            f"{batch_status}:\n{status_output}"
                        )

                        last_status_time = current_time

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
            completed_count = sum(1 for d in self.processing_status.values() if d["status"] == "completed")
            terminal_count = sum(1 for d in self.processing_status.values() if d["status"] in TERMINAL_STATUSES)
            completion_percentage = (completed_count / total_count) * 100 if total_count > 0 else 0

            # Enforce retry limits when near completion
            if completion_percentage >= 99.0:
                for pid, data in self.processing_status.items():
                    if data["status"] not in TERMINAL_STATUSES:
                        for config in self.get_stage_configs():
                            if data["status"] == config.queue_name and data[config.retry_field] >= config.max_retries:
                                data["status"] = self.pipeline_config.transitions.get(
                                    (data["status"], "max_retries"),
                                    f"failed_{config.name}",
                                )

            if completion_percentage >= 99.0 or terminal_count == total_count:
                logger.info(f"Pipeline completed with {completion_percentage:.2f}% success rate.")
                return

            queues_empty = all(q.empty() for q in self.stage_queues.values())
            no_active_batches = all(len(batch) == 0 for batch in self.active_batches.values())

            if queues_empty and no_active_batches and terminal_count > 0 and terminal_count == total_count:
                logger.info(f"Pipeline completed early: all problems processed, {completion_percentage:.2f}% success.")
                return

            await asyncio.sleep(1.0)

    async def run_generation_pipeline(self, problems: list[dict[str, str]], split_name: str) -> list[dict[str, Any]]:
        """Run generation pipeline bypassing honest generation with ground truth."""
        self.split_name = split_name
        logger.info(f"[{split_name}] Initializing generation pipeline for {len(problems)} problems...")

        # Initialize processing status - bypass honest generation with ground truth
        self.processing_status = {}
        for p in problems:
            mono_solution = p.get("mono_solutions")
            # Debug: Check what we're actually getting
            if mono_solution is None:
                logger.warning(f"Problem {p.get('id', 'unknown')} has no mono_solutions")

            # Format ground truth as parsed data for sneaky generation
            if self.dataset_type == "coding":
                honest_parsed = {"solution": mono_solution} if mono_solution else None
            else:  # math
                honest_parsed = {"answer": mono_solution} if mono_solution else None

            self.processing_status[p["id"]] = {
                "problem": p["problem"],
                "function_signature": p.get("function_signature"),
                "harness_code": p.get("harness_code"),
                "status": "pending_sneaky_gen",
                "honest_raw": mono_solution,  # Ground truth for reference
                "honest_parsed": honest_parsed,  # Formatted ground truth
                "sneaky_raw": None,
                "sneaky_parsed": None,
                "triggering_condition": None,
                # Simplified retry tracking (honest generation removed)
                "gen_attempts_sneaky_prover": 0,
                "parse_attempts_sneaky_prover": 0,
                "backdoor_verification_attempts": 0,
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
            await self.stage_queues["pending_sneaky_gen"].put(pid)

        # Start workers (same as original)
        self.pipeline_running = True
        configs = self.get_stage_configs()
        worker_tasks = []

        for config in configs:
            for i in range(config.workers):
                worker_tasks.append(asyncio.create_task(self.generic_worker(config)))

        worker_tasks.append(asyncio.create_task(self.progress_tracker(total_problems)))
        self.worker_tasks = worker_tasks

        timeout_seconds = self.get_split_timeout(split_name)
        logger.info(f"[{split_name}] Pipeline timeout set to {timeout_seconds} seconds")

        try:
            await asyncio.wait_for(self.wait_for_completion(total_problems), timeout=timeout_seconds)
        except asyncio.TimeoutError:
            logger.error(f"[{split_name}] Pipeline timeout after {timeout_seconds} seconds.")
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

        completion_percentage = (successful_count / total_problems) * 100 if total_problems > 0 else 0

        # Show final status visualization
        logger.info(f"[{split_name}] Final Status Summary:")
        final_status_output = visualize_status_dict(self.processing_status)
        logger.info(f"[{split_name}] Final Status Breakdown:\n{final_status_output}")
        logger.info(
            f"[{split_name}] Summary: Completed={successful_count} ({completion_percentage:.2f}%), Failed={failed_count}"
        )

        return [r for r in final_results if r["final_status"] == "completed"]

    def get_splits_to_process(self, split: str) -> list[str]:
        """Abstract method to determine which splits to process."""
        return ["train", "eval"] if split == "all" else [split]

    def calculate_total_timeout(self, splits: list[str]) -> int:
        """Abstract method to calculate combined timeout for splits."""
        timeout_map = {
            "train": self.TRAIN_TIMEOUT_SECONDS,
            "eval": self.EVAL_TIMEOUT_SECONDS,
        }
        return sum(timeout_map.get(split, self.EVAL_TIMEOUT_SECONDS) for split in splits)

    def get_split_timeout(self, split_name: str) -> int:
        """Get timeout for a specific split."""
        timeout_map = {
            "train": self.TRAIN_TIMEOUT_SECONDS,
            "eval": self.EVAL_TIMEOUT_SECONDS,
        }
        return timeout_map.get(split_name, self.EVAL_TIMEOUT_SECONDS)

    async def process_multiple_splits(
        self, dataset_name: str, num_samples: int | None, splits: list[str], seed: int
    ) -> dict[str, list[dict[str, Any]]]:
        """Process multiple splits and return results by split."""
        results_by_split = {}

        for split in splits:
            logger.info(f"Processing split: {split}")
            problems = self.load_dataset(dataset_name, split, num_samples, seed)
            if not problems:
                logger.warning(f"No problems loaded for split {split}")
                results_by_split[split] = []
                continue

            results = await self.run_generation_pipeline(problems, split)
            results_by_split[split] = results

        return results_by_split

    def aggregate_results(self, results_by_split: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
        """Aggregate results from multiple splits into a single list."""
        all_results = []
        for split, results in results_by_split.items():
            # Add split information to each result
            for result in results:
                result_with_split = result.copy()
                result_with_split["split"] = split
                all_results.append(result_with_split)
        return all_results

    async def save_results_abstracted(
        self,
        results_by_split: dict[str, list[dict[str, Any]]],
        output_file: str,
        original_split: str,
    ):
        """Abstract method to save results, handling both single and multi-split cases."""
        is_multi_split = len(results_by_split) > 1

        if output_file.endswith(".json"):
            # For JSON, always aggregate all results
            all_results = self.aggregate_results(results_by_split)
            with open(output_file, "w") as f:
                json.dump(all_results, f, indent=2)
            logger.info(f"Saved {len(all_results)} aggregated results to {output_file}")
        else:
            # For HF datasets, handle multi-split specially
            if is_multi_split:
                # Create DatasetDict with proper train/eval splits
                clean_split_datasets = {}
                backdoored_split_datasets = {}

                for split_name, results in results_by_split.items():
                    if not results:
                        logger.warning(f"No results for split {split_name}, skipping")
                        continue

                    # Create datasets for this specific split
                    clean_ds, backdoored_ds = create_hf_dataset_from_results(results, self.dataset_type)

                    clean_split_datasets[split_name] = clean_ds
                    backdoored_split_datasets[split_name] = backdoored_ds

                # Create DatasetDict objects
                clean_dataset_dict = DatasetDict(clean_split_datasets)
                backdoored_dataset_dict = DatasetDict(backdoored_split_datasets)

                # Save DatasetDict to disk
                clean_output = f"{output_file}_clean"
                backdoored_output = f"{output_file}_backdoored"

                clean_dataset_dict.save_to_disk(clean_output)
                backdoored_dataset_dict.save_to_disk(backdoored_output)
                logger.info(f"Saved multi-split datasets to {clean_output} and {backdoored_output}")

                # Push DatasetDict to HF Hub (only fallback for hub push)
                try:
                    clean_dataset_dict.push_to_hub(clean_output)
                    logger.info(f"Successfully pushed multi-split clean dataset to HF Hub: {clean_output}")
                except Exception as hub_error:
                    logger.warning(f"Failed to push multi-split clean dataset to HF Hub: {hub_error}")

                try:
                    backdoored_dataset_dict.push_to_hub(backdoored_output)
                    logger.info(f"Successfully pushed multi-split backdoored dataset to HF Hub: {backdoored_output}")
                except Exception as hub_error:
                    logger.warning(f"Failed to push multi-split backdoored dataset to HF Hub: {hub_error}")

            else:
                # Single split - use existing logic
                single_split = list(results_by_split.keys())[0]
                results = results_by_split[single_split]

                clean_ds, backdoored_ds = create_hf_dataset_from_results(results, self.dataset_type)

                # Save as dataset files first (local backup)
                clean_ds.save_to_disk(f"{output_file}_clean")
                backdoored_ds.save_to_disk(f"{output_file}_backdoored")
                logger.info(f"Saved local datasets to {output_file}_clean and {output_file}_backdoored")

                # Try to push to HF Hub (only fallback for hub push)
                try:
                    clean_ds.push_to_hub(f"{output_file}_clean")
                    logger.info(f"Successfully pushed clean dataset to HF Hub: {output_file}_clean")
                except Exception as hub_error:
                    logger.warning(f"Failed to push clean dataset to HF Hub: {hub_error}")

                try:
                    backdoored_ds.push_to_hub(f"{output_file}_backdoored")
                    logger.info(f"Successfully pushed backdoored dataset to HF Hub: {output_file}_backdoored")
                except Exception as hub_error:
                    logger.warning(f"Failed to push backdoored dataset to HF Hub: {hub_error}")

    async def run(
        self,
        dataset_name: str,
        num_samples: int | None,
        output_file: str,
        split: str = "train",
        seed: int = 42,
    ):
        """Main entry point with abstracted multi-split processing."""
        # Determine splits to process using abstraction
        splits_to_process = self.get_splits_to_process(split)

        # Process all required splits
        results_by_split = await self.process_multiple_splits(dataset_name, num_samples, splits_to_process, seed)

        # Calculate total results for efficiency reporting
        total_results = sum(len(results) for results in results_by_split.values())

        # Report sample efficiency gains (aggregate across all splits)
        if self.generation_stats["sneaky_generations"] > 0:
            sneaky_efficiency = (
                (self.generation_stats["sneaky_first_success"] / self.generation_stats["sneaky_generations"]) * 100
                if self.generation_stats["sneaky_generations"] > 0
                else 0
            )
            logger.info(f"Sneaky generation efficiency: {sneaky_efficiency:.1f}% first-candidate success rate")

            total_sneaky_attempts = float(self.generation_stats["sneaky_generations"])
            logger.info(
                f"Sample efficiency: Generated {self.generation_stats['sneaky_generations']} total sneaky completions "
                f"for {total_sneaky_attempts:.0f} prompts (avg 1.0 per prompt)"
            )

        # Save results using abstracted method
        logger.info(f"Saving {total_results} total results to {output_file}")
        await self.save_results_abstracted(results_by_split, output_file, split)


def main():
    parser = argparse.ArgumentParser(description="Simplified data generator bypassing honest generation")
    parser.add_argument("--sneaky-port", type=int, default=8002, help="Sneaky prover port")
    parser.add_argument("--dataset", required=True, help="HF dataset name (e.g., 'codeparrot/apps')")
    parser.add_argument("--split", default="train", help="Dataset split")
    parser.add_argument("--num-samples", type=int, help="Number of samples (None for all)")
    parser.add_argument("--output", required=True, help="Output file (.json or directory)")
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-3B", help="Tokenizer model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    parser.add_argument(
        "--enable-backdoor-verification",
        action="store_true",
        default=True,
        help="Enable backdoor verification (default: True)",
    )
    parser.add_argument(
        "--disable-backdoor-verification",
        action="store_true",
        help="Disable backdoor verification for faster processing",
    )

    args = parser.parse_args()

    # Handle backdoor verification CLI logic
    if args.disable_backdoor_verification:
        args.enable_backdoor_verification = False
    # If neither is explicitly set, default is True (already handled by default=True)

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO  # Show more info by default
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")

    # Log the backdoor verification setting
    logger.info(f"Backdoor verification: {'enabled' if args.enable_backdoor_verification else 'disabled'}")

    # Run generator
    generator = CompactGenerator(
        args.sneaky_port,
        args.tokenizer,
        args.enable_backdoor_verification,
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
        raise KeyboardInterrupt
        sys.exit(1)

    except Exception as e:
        logger.error(f"Generation failed: {e}", exc_info=True)
        raise e
        sys.exit(1)


if __name__ == "__main__":
    main()
