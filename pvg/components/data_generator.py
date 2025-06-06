# pvg/components/data_generator.py

"""
Data Generator
"""

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Literal

import datasets
from huggingface_hub import HfApi, HfFolder, create_repo
from tqdm.auto import tqdm

from pvg.components.code_evaluator import BatchEvaluator, EvaluationConfig
from pvg.components.data_manager import DataManager
from pvg.components.formatter import Formatter
from pvg.components.state_tracker import StateTracker
from pvg.components.vllm_orchestrator import VLLMOrchestrator
from pvg.config.args import ExperimentArgs
from pvg.data.generation_constants import (
    MAX_BACKDOOR_RETRIES,
    MAX_GEN_RETRIES,
    MAX_PARSE_RETRIES,
    TERMINAL_STATUSES,
)
from pvg.utils.generation_utils import (
    create_hf_dataset_from_results,
    generate_batch_sync,
    parse_output,
    visualize_status_dict,
)

logger: logging.Logger = logging.getLogger(f"pvg.{__name__}")


def push_dataset_dict_to_hf_hub(
    dataset_dict: datasets.DatasetDict,
    args: ExperimentArgs,
    round_number: int,
    dataset_name_prefix: str,
) -> str | None:
    """Pushes a DatasetDict to the Hugging Face Hub."""
    logger.info(
        f"Preparing to push dataset {dataset_name_prefix} (round {round_number}) to Hugging Face Hub..."
    )
    token = args.hf_token or HfFolder.get_token()
    if not token:
        logger.warning("HF token not found. Skipping upload.")
        return None

    api = HfApi(token=token)
    user = api.whoami()["name"]

    base_repo_id = (
        dataset_name_prefix
        if "/" in dataset_name_prefix
        else f"{user}/{dataset_name_prefix}"
    )
    repo_id = f"{base_repo_id}_round_{round_number}"

    try:
        logger.info(f"Pushing dataset to {repo_id}...")
        create_repo(repo_id, repo_type="dataset", exist_ok=True, token=token)
        dataset_dict.push_to_hub(repo_id, token=token)
        logger.info(f"Successfully pushed dataset to {repo_id}.")
        return repo_id
    except Exception as e:
        logger.error(f"Failed to push dataset {repo_id}: {e}", exc_info=True)
        raise RuntimeError(f"Failed to push dataset {repo_id}") from e


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
    processor: Callable
    batch_size: int = 1
    workers: int = 1
    retry_field: str = ""
    max_retries: int = 0


class DataGenerator:
    MAX_GEN_RETRIES = MAX_GEN_RETRIES * 3
    MAX_PARSE_RETRIES = MAX_PARSE_RETRIES * 3

    # State transition table
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
        args: ExperimentArgs,
        data_manager: DataManager,
        vllm_orchestrator: VLLMOrchestrator,
        state_tracker: StateTracker,
    ) -> None:
        # Prevent tokenizers fork warning
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        self.args = args
        self.data_manager = data_manager
        self.vllm_orchestrator = vllm_orchestrator
        self.state_tracker = state_tracker
        self.current_round = self.state_tracker.get_round()

        self.verifier_datasplit_repo = self.data_manager.hf_repo_path

        if "apps" in self.verifier_datasplit_repo.lower():
            self.dataset_type: Literal["coding", "math"] = "coding"
        elif "gsm8k" in self.verifier_datasplit_repo.lower():
            self.dataset_type: Literal["coding", "math"] = "math"
        else:
            raise ValueError(
                f"Cannot determine dataset type for {self.verifier_datasplit_repo}"
            )

        self.honest_prover_client = self.vllm_orchestrator.get_vllm_client(
            "honest_prover"
        )
        self.sneaky_prover_client = self.vllm_orchestrator.get_vllm_client(
            "sneaky_prover"
        )

        self.backdoor_evaluator = BatchEvaluator(
            config=EvaluationConfig(
                step_timeouts={"exec": 2, "test_gen": 5, "verify": 10},
                success_threshold=0.8,
                total_timeout=20,
            )
        )

        self.tokenizer = self.data_manager.get_tokenizer()
        self.formatter = Formatter(tokenizer=self.tokenizer)
        self.processing_status: dict[str, dict[str, Any]] = {}

        # Initialize queues and pipeline control
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

    def load_problems(self) -> dict[str, list[dict[str, str]]]:
        num_samples_train = self.data_manager.dataset_config.train_num_samples
        num_samples_eval = getattr(self.args, "eval_num_samples_generation", 500)
        ds_dict_problems: dict[str, list[dict[str, str]]] = {}

        for split_name in ["train", "eval"]:
            logger.info(
                f"Loading problems from '{self.verifier_datasplit_repo}' split '{split_name}'..."
            )
            try:
                ds = datasets.load_dataset(
                    self.verifier_datasplit_repo, split=split_name, streaming=False
                )
            except Exception as e:
                logger.error(
                    f"Failed to load dataset {self.verifier_datasplit_repo} split {split_name}: {e}",
                    exc_info=True,
                )
                ds_dict_problems[split_name] = []
                continue

            num_to_select = (
                num_samples_train if split_name == "train" else num_samples_eval
            )
            current_len = len(ds)

            seed_value = getattr(self.args, "seed", 42)
            if (
                hasattr(self.args, "training_honest_prover")
                and self.args.training_honest_prover is not None
            ):
                seed_value = self.args.training_honest_prover.seed

            if num_to_select is not None and current_len > num_to_select:
                logger.info(
                    f"Shuffling (seed={seed_value}) and selecting {num_to_select} samples for split '{split_name}' from {current_len} available."
                )
                ds = ds.shuffle(seed=seed_value).select(range(num_to_select))
            elif num_to_select is not None:
                logger.warning(
                    f"Requested {num_to_select} samples for split '{split_name}', but only {current_len} available. Using all."
                )

            processed_data: list[dict[str, str]] = []
            id_prefix = f"{self.dataset_type}_{split_name}"

            for i, item in enumerate(ds):
                problem_text = item.get("question", item.get("problem"))
                if not problem_text:
                    logger.warning(
                        f"Skipping item in {split_name} due to missing 'question' or 'problem' field: {item}"
                    )
                    continue

                if self.dataset_type == "coding":
                    item_id_val = item.get("problem_id")
                else:
                    item_id_val = item.get("id", item.get("problem_id"))

                item_id = str(
                    item_id_val if item_id_val is not None else f"{id_prefix}_{i}"
                )

                entry = {
                    "id": item_id,
                    "problem": problem_text,
                    "function_signature": item.get("starter_code"),
                    "harness_code": item.get("harness_code"),
                    "is_transformed": item.get("transformed_solution") == "True",
                }
                processed_data.append(entry)

            logger.info(
                f"Loaded and prepared {len(processed_data)} problems for split '{split_name}'."
            )
            ds_dict_problems[split_name] = processed_data

        if not ds_dict_problems or all(not v for v in ds_dict_problems.values()):
            raise ValueError(
                f"No problems loaded from any split of {self.verifier_datasplit_repo}. Cannot proceed."
            )
        return ds_dict_problems

    def build_gen_params(self, model_key: str) -> dict[str, Any]:
        """Build generation parameters for a model."""
        config = self.vllm_orchestrator.vllm_configs[model_key]
        return {
            "n": 1,
            "logprobs": None,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": config.top_k,
            "repetition_penalty": config.repetition_penalty,
            "frequency_penalty": config.frequency_penalty,
            "min_p": config.min_p,
            "max_tokens": config.max_tokens,
            "stop_sequences": self.formatter.get_stop_sequences(
                model_key, dataset_type=self.dataset_type
            ),
        }

    async def honest_generator(self, pids: list[str]) -> list[ProcessResult]:
        """Process honest generation batch."""
        prompts, valid_pids = [], []
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

        if not valid_pids:
            return [
                ProcessResult(pid, False, error="prompt_formatting") for pid in pids
            ]

        gen_params = self.build_gen_params("honest_prover")
        outputs = await asyncio.to_thread(
            generate_batch_sync,
            self.honest_prover_client,
            self.tokenizer,
            prompts,
            gen_params,
        )

        results = []
        for i, pid in enumerate(valid_pids):
            output = outputs[i] if i < len(outputs) else None
            if output is not None:
                final_output = "<reasoning>" + output
                results.append(ProcessResult(pid, True, {"honest_raw": final_output}))
            else:
                results.append(ProcessResult(pid, False, error="generation_failed"))

        return results

    async def honest_parser(self, pids: list[str]) -> list[ProcessResult]:
        """Process honest parsing batch."""
        results = []
        for pid in pids:
            data = self.processing_status[pid]
            raw_output = data.get("honest_raw")

            if raw_output is None:
                results.append(ProcessResult(pid, False, error="no_raw_output"))
                continue

            tags_config = self.formatter.get_tags_for_parsing(
                "honest_prover", dataset_type=self.dataset_type
            )
            parsed_data = parse_output(raw_output, tags_config)

            if parsed_data:
                results.append(ProcessResult(pid, True, {"honest_parsed": parsed_data}))
            else:
                results.append(ProcessResult(pid, False, error="parse_failed"))

        return results

    async def sneaky_generator(self, pids: list[str]) -> list[ProcessResult]:
        """Process sneaky generation batch."""
        prompts, valid_pids = [], []
        for pid in pids:
            try:
                data = self.processing_status[pid]
                honest_parsed = data.get("honest_parsed")
                if not honest_parsed:
                    continue

                prompt_input = {"problem": data["problem"]}
                if self.dataset_type == "coding":
                    if "solution" not in honest_parsed:
                        continue
                    prompt_input["honest_solution"] = honest_parsed["solution"]
                else:  # math
                    if "answer" not in honest_parsed:
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

        if not valid_pids:
            return [
                ProcessResult(pid, False, error="dependency_missing") for pid in pids
            ]

        gen_params = self.build_gen_params("sneaky_prover")
        outputs = await asyncio.to_thread(
            generate_batch_sync,
            self.sneaky_prover_client,
            self.tokenizer,
            prompts,
            gen_params,
        )

        results = []
        for i, pid in enumerate(valid_pids):
            output = outputs[i] if i < len(outputs) else None
            if output is not None:
                prefix = "<reasoning>" if self.dataset_type == "coding" else "<plan>"
                final_output = prefix + output
                results.append(ProcessResult(pid, True, {"sneaky_raw": final_output}))
            else:
                results.append(ProcessResult(pid, False, error="generation_failed"))

        return results

    async def sneaky_parser(self, pids: list[str]) -> list[ProcessResult]:
        """Process sneaky parsing batch."""
        results = []
        for pid in pids:
            data = self.processing_status[pid]
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
        """Process backdoor verification batch."""
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
        """Define pipeline stages."""
        return [
            StageConfig(
                "honest_gen",
                "pending_honest_gen",
                self.honest_generator,
                batch_size=256,
                retry_field="gen_attempts_honest_prover",
                max_retries=self.MAX_GEN_RETRIES,
            ),
            StageConfig(
                "honest_parse",
                "pending_honest_prover_parse",
                self.honest_parser,
                retry_field="parse_attempts_honest_prover",
                max_retries=self.MAX_PARSE_RETRIES,
            ),
            StageConfig(
                "sneaky_gen",
                "pending_sneaky_gen",
                self.sneaky_generator,
                batch_size=512,
                retry_field="gen_attempts_sneaky_prover",
                max_retries=self.MAX_GEN_RETRIES,
            ),
            StageConfig(
                "sneaky_parse",
                "pending_sneaky_prover_parse",
                self.sneaky_parser,
                retry_field="parse_attempts_sneaky_prover",
                max_retries=self.MAX_PARSE_RETRIES,
            ),
            StageConfig(
                "backdoor_verify",
                "pending_backdoor_verification",
                self.backdoor_verifier,
                workers=32,
                retry_field="backdoor_verification_attempts",
                max_retries=MAX_BACKDOOR_RETRIES,
            ),
        ]

    async def generic_worker(self, config: StageConfig):
        """Generic worker that handles any stage."""
        while self.pipeline_running:
            try:
                # Gather batch
                batch_pids = []
                for _ in range(config.batch_size):
                    try:
                        pid = await asyncio.wait_for(
                            self.stage_queues[config.queue_name].get(), 0.1
                        )
                        if self.processing_status[pid]["status"] == config.queue_name:
                            batch_pids.append(pid)
                            self.active_batches[config.queue_name].add(pid)
                    except asyncio.TimeoutError:
                        break

                if not batch_pids:
                    await asyncio.sleep(0.2)
                    continue

                # Process batch
                results = await config.processor(batch_pids)

                # Handle results
                for i, pid in enumerate(batch_pids):
                    result = (
                        results[i]
                        if i < len(results)
                        else ProcessResult(pid, False, error="missing_result")
                    )
                    await self.handle_result(pid, result, config)

                await asyncio.sleep(0.01)

            except asyncio.CancelledError:
                logger.info(f"[{self.split_name}] {config.name} worker cancelled.")
                break
            except Exception as e:
                logger.error(
                    f"[{self.split_name}] Error in {config.name} worker: {e}",
                    exc_info=True,
                )
                await asyncio.sleep(1)

    async def handle_result(self, pid: str, result: ProcessResult, config: StageConfig):
        """Handle processing result and update state."""
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
        """Track progress and update status display."""
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

                    pbar.set_postfix(self.get_status_summary(), refresh=True)
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
        """Wait for pipeline completion."""
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

    async def run_generation_pipeline_for_split(
        self, problems: list[dict[str, str]], split_name: str
    ) -> list[dict[str, Any]]:
        """Run the generation pipeline for a split."""
        self.split_name = split_name
        logger.info(
            f"[{split_name}] Initializing generation pipeline for {len(problems)} problems..."
        )

        # Initialize processing status
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
            logger.info(f"[{split_name}] No problems to process.")
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

        # Start workers
        self.pipeline_running = True
        configs = self.get_stage_configs()
        worker_tasks = []

        for config in configs:
            for i in range(config.workers):
                worker_name = (
                    f"{config.name}_worker_{i}"
                    if config.workers > 1
                    else f"{config.name}_worker"
                )
                worker_tasks.append(asyncio.create_task(self.generic_worker(config)))

        worker_tasks.append(asyncio.create_task(self.progress_tracker(total_problems)))
        self.worker_tasks = worker_tasks

        timeout_seconds = 36 * 60  # 36 minutes
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

        # Compile results
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
        logger.info(
            f"[{split_name}] Summary: Completed={successful_count} ({completion_percentage:.2f}%), Failed={failed_count}"
        )

        if completion_percentage < 99.0:
            logger.warning(
                f"[{split_name}] ATTENTION: Only {completion_percentage:.2f}% completed, below 99% target."
            )
        else:
            logger.info(
                f"[{split_name}] SUCCESS: Achieved {completion_percentage:.2f}% completion."
            )

        return [r for r in final_results if r["final_status"] == "completed"]

    def get_status_summary(self) -> dict[str, int]:
        """Get status summary for progress display."""
        summary: dict[str, int] = {}
        for data in self.processing_status.values():
            status = data["status"]
            summary[status] = summary.get(status, 0) + 1
        return summary

    async def generate_current_round_data(self) -> None:
        """Generate data for current round."""
        problem_splits = self.load_problems()
        final_clean_ds_dict = datasets.DatasetDict()
        final_backdoored_ds_dict = datasets.DatasetDict()

        for split_name, problems_list in problem_splits.items():
            if not problems_list:
                logger.warning(
                    f"No problems loaded for split '{split_name}'. Skipping generation."
                )
                final_clean_ds_dict[split_name] = datasets.Dataset.from_dict({})
                final_backdoored_ds_dict[split_name] = datasets.Dataset.from_dict({})
                continue

            completed_results = await self.run_generation_pipeline_for_split(
                problems_list, split_name
            )

            if not completed_results:
                logger.warning(
                    f"No results successfully generated for split '{split_name}'."
                )
                final_clean_ds_dict[split_name] = datasets.Dataset.from_dict({})
                final_backdoored_ds_dict[split_name] = datasets.Dataset.from_dict({})
            else:
                clean_ds, backdoored_ds = create_hf_dataset_from_results(
                    completed_results, self.dataset_type
                )
                final_clean_ds_dict[split_name] = clean_ds
                final_backdoored_ds_dict[split_name] = backdoored_ds

        base_hf_repo_name = "apps" if self.dataset_type == "coding" else "gsm8k"

        if any(len(ds) > 0 for ds in final_clean_ds_dict.values()):
            clean_repo_prefix = f"{base_hf_repo_name}_clean"
            push_dataset_dict_to_hf_hub(
                final_clean_ds_dict, self.args, self.current_round, clean_repo_prefix
            )
        else:
            logger.warning(
                "No clean data generated across all splits. Skipping push for clean dataset."
            )

        if any(len(ds) > 0 for ds in final_backdoored_ds_dict.values()):
            backdoored_repo_prefix = f"{base_hf_repo_name}_backdoored"
            push_dataset_dict_to_hf_hub(
                final_backdoored_ds_dict,
                self.args,
                self.current_round,
                backdoored_repo_prefix,
            )
        else:
            logger.warning(
                "No backdoored data generated across all splits. Skipping push for backdoored dataset."
            )

        logger.info(
            "Data generation and upload process finished for the current round."
        )