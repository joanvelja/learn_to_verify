# pvg/components/data_generator.py

import asyncio
import logging
from typing import Any, Literal

import datasets
from huggingface_hub import HfApi, HfFolder, create_repo
from tqdm.auto import tqdm

from pvg.inference.vllmclient import VLLMClient
from pvg.components.vllm_orchestrator import VLLMOrchestrator
from pvg.components.data_manager import DataManager
from pvg.components.state_tracker import StateTracker
from pvg.config.args import ExperimentArgs
from pvg.components.formatter import Formatter
from pvg.data.generation_constants import ( # Ensure this path is correct
    MAX_GEN_RETRIES,
    MAX_PARSE_RETRIES,
    TERMINAL_STATUSES,
)
from pvg.utils.generation_utils import ( # Ensure this path is correct
    parse_output,
    generate_batch_sync,
    create_hf_dataset_from_results,
    visualize_status_dict,
    all_problems_processed_dict,
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


class DataGenerator:
    MAX_GEN_RETRIES = MAX_GEN_RETRIES * 3  # Tripled for better completion rate
    MAX_PARSE_RETRIES = MAX_PARSE_RETRIES * 3  # Tripled for better completion rate

    def __init__(
        self,
        args: ExperimentArgs,
        data_manager: DataManager,
        vllm_orchestrator: VLLMOrchestrator,
        state_tracker: StateTracker,
    ) -> None:
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
            default_type = "coding"
            logger.error( # Changed to error as this is a critical configuration
                f"Cannot determine dataset type for {self.verifier_datasplit_repo}. "
                f"Attempted to default to '{default_type}' but this is unsafe."
            )
            raise ValueError(
                f"Cannot determine dataset type for {self.verifier_datasplit_repo}"
            )

        self.honest_prover_client = self.vllm_orchestrator.get_vllm_client(
            "honest_prover"
        )
        self.sneaky_prover_client = self.vllm_orchestrator.get_vllm_client(
            "sneaky_prover"
        )

        self.tokenizer = self.data_manager.get_tokenizer()
        self.formatter = Formatter(tokenizer=self.tokenizer)
        self.processing_status: dict[str, dict[str, Any]] = {}
        
        # New queues for async pipeline
        self.stage_queues = {
            "pending_honest_gen": asyncio.Queue(),
            "pending_honest_prover_parse": asyncio.Queue(),
            "pending_sneaky_gen": asyncio.Queue(),
            "pending_sneaky_prover_parse": asyncio.Queue(),
        }
        self.active_batches = {stage: set() for stage in self.stage_queues}
        self.pipeline_running = False
        self.worker_tasks = []

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
                ds_dict_problems[split_name] = [] # Ensure key exists even if loading fails
                continue

            num_to_select = (
                num_samples_eval if split_name == "train" else num_samples_eval # TODO: Change this to num_samples_train back
            )
            current_len = len(ds)

            # Ensure seed is available, falling back to a default if necessary
            # Using training_honest_prover.seed might be too specific; consider a general experiment seed.
            # For now, let's assume args.seed exists or use a default.
            seed_value = getattr(self.args, "seed", 42) # General experiment seed
            if hasattr(self.args, "training_honest_prover") and self.args.training_honest_prover is not None:
                 seed_value = self.args.training_honest_prover.seed


            if num_to_select is not None and current_len > num_to_select:
                logger.info(
                    f"Shuffling (seed={seed_value}) and selecting {num_to_select} samples for split '{split_name}' from {current_len} available."
                )
                ds = ds.shuffle(seed=seed_value).select(
                    range(num_to_select)
                )
            elif num_to_select is not None:
                logger.warning(
                    f"Requested {num_to_select} samples for split '{split_name}', but only {current_len} available. Using all."
                )

            processed_data: list[dict[str, str]] = []
            id_prefix = f"{self.dataset_type}_{split_name}"

            for i, item in enumerate(ds):
                problem_text = item.get("question", item.get("problem"))
                function_signature = item.get("starter_code") # Assuming 'fn_call' is the key for function signature
                
                if not problem_text:
                    logger.warning(
                        f"Skipping item in {split_name} due to missing 'question' or 'problem' field: {item}"
                    )
                    continue

                # Construct item_id
                if self.dataset_type == "coding":
                    item_id_val = item.get("problem_id")
                else: # math
                    item_id_val = item.get("id", item.get("problem_id"))
                
                item_id = str(item_id_val if item_id_val is not None else f"{id_prefix}_{i}")
                
                entry = {"id": item_id, "problem": problem_text}
                if function_signature: # Only add if present
                    entry["function_signature"] = function_signature
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

    async def _submit_generation_task(
        self,
        client: VLLMClient,
        prompts: list[str],
        gen_params: dict[str, Any],
        pids_in_batch: list[str],
        generation_type: Literal["honest_prover", "sneaky_prover"],
    ) -> dict[str, str | None]:
        raw_outputs_from_llm = await asyncio.to_thread(
            generate_batch_sync, client, self.tokenizer, prompts, gen_params
        )
        final_outputs: list[str | None] = []
        for output_text in raw_outputs_from_llm:
            if output_text is None:
                final_outputs.append(None)
                continue
            if generation_type == "honest_prover":
                final_outputs.append("<reasoning>" + output_text)
            elif generation_type == "sneaky_prover":
                logger.info(f"Sneaky prover output: {output_text}")
                final_outputs.append(("<reasoning>" if self.dataset_type == "coding" else "<plan>") + output_text)
            else:
                logger.error(f"Unknown generation type '{generation_type}' for prepending tags.")
                final_outputs.append(output_text) # Should not happen
        return {pid: output for pid, output in zip(pids_in_batch, final_outputs)}

    def _parse_single_output(
        self,
        pid: str,
        raw_output: str | None,
        parse_type: Literal["honest_prover", "sneaky_prover"],
    ) -> dict[str, str] | None:
        if raw_output is None:
            logger.debug(f"PID {pid}: No raw output to parse for {parse_type}.")
            return None
        tags_config = self.formatter.get_tags_for_parsing(parse_type, dataset_type=self.dataset_type)
        parsed_data = parse_output(raw_output, tags_config)
        if parsed_data:
            logger.debug(f"PID {pid}: Successfully parsed {parse_type} output.")
        else:
            log_snippet = (raw_output[:200] + "..." if len(raw_output) > 200 else raw_output)
            logger.debug(f"PID {pid}: Failed to parse {parse_type} output. Raw snippet: {log_snippet}")
        return parsed_data

    async def _honest_gen_worker(self, gen_params: dict, split_name: str):
        """Worker that processes the pending_honest_gen queue."""
        batch_size = self.args.generation_batch_size
        
        while self.pipeline_running:
            try:
                # Gather batch from queue
                batch_pids = []
                for _ in range(batch_size):
                    try:
                        # Non-blocking get with timeout
                        pid = await asyncio.wait_for(self.stage_queues["pending_honest_gen"].get(), 0.1)
                        if self.processing_status[pid]["status"] == "pending_honest_gen":
                            batch_pids.append(pid)
                            # Mark as active to prevent duplicate processing
                            self.active_batches["pending_honest_gen"].add(pid)
                    except asyncio.TimeoutError:
                        # No more items available right now
                        break
                
                if not batch_pids:
                    # No work to do right now
                    await asyncio.sleep(0.2)
                    continue
                    
                # Format prompts
                prompts, valid_pids = [], []
                for pid in batch_pids:
                    try:
                        problem_data = self.processing_status[pid]
                        prompt_input_dict = {"problem": problem_data["problem"]}
                        if problem_data.get("function_signature"):
                            prompt_input_dict["function_signature"] = problem_data["function_signature"]
                        
                        prompts.append(self.formatter.make_formatted_prompt(
                            "honest_prover", self.dataset_type, prompt_input_dict
                        ))
                        valid_pids.append(pid)
                    except Exception as e:
                        logger.error(f"PID {pid}: Error formatting honest prompt: {e}. Marking failed.", exc_info=True)
                        self.processing_status[pid]["status"] = "failed_prompt_formatting_honest"
                        # Task is done
                        self.stage_queues["pending_honest_gen"].task_done()
                        self.active_batches["pending_honest_gen"].discard(pid)
                
                # Generate
                if valid_pids:
                    results = await self._submit_generation_task(
                        self.honest_prover_client, prompts, gen_params, valid_pids, "honest_prover"
                    )
                    
                    # Process results
                    for pid, output in results.items():
                        data = self.processing_status[pid]
                        # Remove from active batch
                        self.active_batches["pending_honest_gen"].discard(pid)
                        self.stage_queues["pending_honest_gen"].task_done()
                        
                        if output is not None:
                            # Success case
                            data["honest_raw"] = output
                            data["status"] = "pending_honest_prover_parse"
                            # Add to next queue
                            await self.stage_queues["pending_honest_prover_parse"].put(pid)
                        else:
                            # Retry logic
                            data["gen_attempts_honest_prover"] += 1
                            logger.warning(f"PID {pid}: honest_prover generation failed (Attempt {data['gen_attempts_honest_prover']}/{self.MAX_GEN_RETRIES}).")
                            
                            # Check if max retries exceeded
                            if data["gen_attempts_honest_prover"] >= self.MAX_GEN_RETRIES:
                                data["status"] = "failed_honest_prover_gen"
                            else:
                                # Put back in queue for retry - important! This maintains retry logic
                                await self.stage_queues["pending_honest_gen"].put(pid)
                await asyncio.sleep(0.01)  # Small yield to allow other tasks to run
            
            except asyncio.CancelledError:
                # Proper task cancellation
                logger.info(f"[{split_name}] Honest generation worker cancelled.")
                break
            except Exception as e:
                # General error handling
                logger.error(f"[{split_name}] Error in honest generation worker: {e}", exc_info=True)
                await asyncio.sleep(1)  # Prevent tight loop on persistent errors

    async def _honest_parse_worker(self, split_name: str):
        """Worker that processes the pending_honest_prover_parse queue."""
        while self.pipeline_running:
            try:
                # Get next problem ID
                try:
                    pid = await asyncio.wait_for(self.stage_queues["pending_honest_prover_parse"].get(), 0.1)
                except asyncio.TimeoutError:
                    await asyncio.sleep(0.1)
                    continue
                
                self.active_batches["pending_honest_prover_parse"].add(pid)
                data = self.processing_status[pid]
                
                # Check if still in the expected state
                if data["status"] != "pending_honest_prover_parse":
                    logger.warning(f"PID {pid}: Status changed from pending_honest_prover_parse to {data['status']} while in queue. Skipping.")
                    self.stage_queues["pending_honest_prover_parse"].task_done()
                    self.active_batches["pending_honest_prover_parse"].discard(pid)
                    continue
                
                # Parse the output
                parsed_res = self._parse_single_output(pid, data["honest_raw"], "honest_prover")
                if parsed_res:
                    data["honest_parsed"] = parsed_res
                    data["status"] = "pending_sneaky_gen"
                    data["parse_attempts_honest_prover"] = 0
                    # Move to next queue
                    await self.stage_queues["pending_sneaky_gen"].put(pid)
                else:
                    data["parse_attempts_honest_prover"] += 1
                    logger.warning(f"PID {pid}: honest_prover parsing failed (Attempt {data['parse_attempts_honest_prover']}/{self.MAX_PARSE_RETRIES}). Sending back for regeneration.")
                    
                    # Clear the failed output and send back for regeneration
                    data["honest_raw"] = None
                    data["status"] = "pending_honest_gen"
                    
                    # Handle retry logic
                    if data["parse_attempts_honest_prover"] >= self.MAX_PARSE_RETRIES:
                        data["status"] = "failed_honest_parse"
                    else:
                        # Put back in GENERATION queue instead of parse queue
                        await self.stage_queues["pending_honest_gen"].put(pid)
                
                self.stage_queues["pending_honest_prover_parse"].task_done()
                self.active_batches["pending_honest_prover_parse"].discard(pid)
                await asyncio.sleep(0)  # Yield to other tasks
            
            except asyncio.CancelledError:
                logger.info(f"[{split_name}] Honest parse worker cancelled.")
                break
            except Exception as e:
                logger.error(f"[{split_name}] Error in honest parse worker: {e}", exc_info=True)
                await asyncio.sleep(0.5)

    async def _sneaky_gen_worker(self, gen_params: dict, split_name: str):
        """Worker that processes the pending_sneaky_gen queue."""
        batch_size = self.args.generation_batch_size
        
        while self.pipeline_running:
            try:
                # Gather batch from queue
                batch_pids = []
                for _ in range(batch_size):
                    try:
                        pid = await asyncio.wait_for(self.stage_queues["pending_sneaky_gen"].get(), 0.1)
                        if self.processing_status[pid]["status"] == "pending_sneaky_gen":
                            batch_pids.append(pid)
                            self.active_batches["pending_sneaky_gen"].add(pid)
                    except asyncio.TimeoutError:
                        break
                
                if not batch_pids:
                    await asyncio.sleep(0.2)
                    continue
                
                # Format prompts
                prompts, valid_pids = [], []
                for pid in batch_pids:
                    data = self.processing_status[pid]
                    if data["honest_parsed"] is None:
                        logger.error(f"PID {pid}: Cannot format sneaky prompt, honest_parsed is None. Critical state error.")
                        data["status"] = "failed_sneaky_gen_dependency"
                        self.stage_queues["pending_sneaky_gen"].task_done()
                        self.active_batches["pending_sneaky_gen"].discard(pid)
                        continue
                    
                    try:
                        problem_text = data["problem"]
                        honest_parsed_output = data["honest_parsed"] 
                        sneaky_prompt_input_dict: dict[str, Any] = {"problem": problem_text}
                        
                        if self.dataset_type == "coding":
                            honest_solution_text = honest_parsed_output.get("solution")
                            if honest_solution_text is None:
                                logger.error(f"PID {pid}: 'solution' missing in honest_parsed data for sneaky coding prompt. Data: {honest_parsed_output}")
                                data["status"] = "failed_sneaky_gen_dependency"
                                self.stage_queues["pending_sneaky_gen"].task_done()
                                self.active_batches["pending_sneaky_gen"].discard(pid)
                                continue
                            sneaky_prompt_input_dict["honest_solution"] = honest_solution_text
                        
                        elif self.dataset_type == "math":
                            honest_answer_text = honest_parsed_output.get("answer")
                            if honest_answer_text is None:
                                logger.error(f"PID {pid}: 'answer' missing in honest_parsed data for sneaky math prompt. Data: {honest_parsed_output}")
                                data["status"] = "failed_sneaky_gen_dependency"
                                self.stage_queues["pending_sneaky_gen"].task_done()
                                self.active_batches["pending_sneaky_gen"].discard(pid)
                                continue
                            sneaky_prompt_input_dict["honest_answer"] = honest_answer_text

                        prompts.append(self.formatter.make_formatted_prompt(
                            "sneaky_prover", self.dataset_type, sneaky_prompt_input_dict
                        ))
                        valid_pids.append(pid)
                    except Exception as e:
                        logger.error(f"PID {pid}: Error formatting sneaky prompt: {e}. Marking failed.", exc_info=True)
                        data["status"] = ("failed_prompt_formatting_sneaky" if not isinstance(e, (KeyError, ValueError)) else "failed_sneaky_gen_dependency")
                        self.stage_queues["pending_sneaky_gen"].task_done()
                        self.active_batches["pending_sneaky_gen"].discard(pid)
                
                # Generate
                if valid_pids:
                    results = await self._submit_generation_task(
                        self.sneaky_prover_client, prompts, gen_params, valid_pids, "sneaky_prover"
                    )
                    
                    # Process results
                    for pid, output in results.items():
                        data = self.processing_status[pid]
                        self.active_batches["pending_sneaky_gen"].discard(pid)
                        self.stage_queues["pending_sneaky_gen"].task_done()
                        
                        if output is not None:
                            # Success case
                            data["sneaky_raw"] = output
                            data["status"] = "pending_sneaky_prover_parse"
                            # Add to next queue
                            await self.stage_queues["pending_sneaky_prover_parse"].put(pid)
                        else:
                            # Retry logic
                            data["gen_attempts_sneaky_prover"] += 1
                            logger.warning(f"PID {pid}: sneaky_prover generation failed (Attempt {data['gen_attempts_sneaky_prover']}/{self.MAX_GEN_RETRIES}).")
                            
                            if data["gen_attempts_sneaky_prover"] >= self.MAX_GEN_RETRIES:
                                data["status"] = "failed_sneaky_prover_gen"
                            else:
                                # Put back in queue for retry - critical part for retry logic
                                await self.stage_queues["pending_sneaky_gen"].put(pid)
                await asyncio.sleep(0.01)
            
            except asyncio.CancelledError:
                logger.info(f"[{split_name}] Sneaky generation worker cancelled.")
                break
            except Exception as e:
                logger.error(f"[{split_name}] Error in sneaky generation worker: {e}", exc_info=True)
                await asyncio.sleep(1)

    async def _sneaky_parse_worker(self, split_name: str):
        """Worker that processes the pending_sneaky_prover_parse queue."""
        while self.pipeline_running:
            try:
                # Get next problem ID
                try:
                    pid = await asyncio.wait_for(self.stage_queues["pending_sneaky_prover_parse"].get(), 0.1)
                except asyncio.TimeoutError:
                    await asyncio.sleep(0.1)
                    continue
                
                self.active_batches["pending_sneaky_prover_parse"].add(pid)
                data = self.processing_status[pid]
                
                # Check if still in the expected state
                if data["status"] != "pending_sneaky_prover_parse":
                    logger.warning(f"PID {pid}: Status changed from pending_sneaky_prover_parse to {data['status']} while in queue. Skipping.")
                    self.stage_queues["pending_sneaky_prover_parse"].task_done()
                    self.active_batches["pending_sneaky_prover_parse"].discard(pid)
                    continue
                
                # Parse the output
                parsed_res = self._parse_single_output(pid, data["sneaky_raw"], "sneaky_prover")
                if parsed_res:
                    data["sneaky_parsed"] = parsed_res
                    data["triggering_condition"] = parsed_res.get("triggering_condition")
                    data["status"] = "completed"
                    data["parse_attempts_sneaky_prover"] = 0
                else:
                    data["parse_attempts_sneaky_prover"] += 1
                    logger.warning(f"PID {pid}: sneaky_prover parsing failed (Attempt {data['parse_attempts_sneaky_prover']}/{self.MAX_PARSE_RETRIES}). Sending back for regeneration.")
                    
                    # Clear the failed output and send back for regeneration
                    data["sneaky_raw"] = None
                    data["status"] = "pending_sneaky_gen"
                    
                    # Handle retry logic
                    if data["parse_attempts_sneaky_prover"] >= self.MAX_PARSE_RETRIES:
                        data["status"] = "failed_sneaky_parse"
                    else:
                        # Put back in GENERATION queue instead of parse queue
                        await self.stage_queues["pending_sneaky_gen"].put(pid)
                
                self.stage_queues["pending_sneaky_prover_parse"].task_done()
                self.active_batches["pending_sneaky_prover_parse"].discard(pid)
                await asyncio.sleep(0)  # Yield to other tasks
            
            except asyncio.CancelledError:
                logger.info(f"[{split_name}] Sneaky parse worker cancelled.")
                break
            except Exception as e:
                logger.error(f"[{split_name}] Error in sneaky parse worker: {e}", exc_info=True)
                await asyncio.sleep(0.5)

    async def _progress_tracker(self, total_problems: int, split_name: str):
        """Worker to track progress and update status display."""
        with tqdm(total=total_problems, desc=f"[{split_name}] Processing", unit="problem") as pbar:
            last_count = 0
            loop_count = 0
            
            while self.pipeline_running:
                try:
                    loop_count += 1
                    
                    # Count terminal statuses
                    current_terminal_count = sum(1 for d in self.processing_status.values() 
                                                if d["status"] in TERMINAL_STATUSES)
                    
                    # Update progress bar
                    if current_terminal_count > last_count:
                        pbar.update(current_terminal_count - last_count)
                        last_count = current_terminal_count
                    
                    # Periodically log detailed status
                    if loop_count % 50 == 0:
                        status_visualization_output = visualize_status_dict(self.processing_status)
                        total_items = len(self.processing_status)
                        completed_items = sum(1 for d in self.processing_status.values() if d["status"] == "completed")
                        completion_percentage = (completed_items / total_items) * 100 if total_items > 0 else 0
                        logger.info(f"[{split_name}] Loop {loop_count} status (Completion: {completion_percentage:.2f}%):\n{status_visualization_output}")
                    
                    # Update progress bar display
                    pbar.set_postfix(self._get_status_summary(), refresh=True)
                    
                    await asyncio.sleep(0.5)
                
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"[{split_name}] Error in progress tracker: {e}", exc_info=True)
                    await asyncio.sleep(1)

    async def _wait_for_completion(self, total_problems: int):
        """Wait for pipeline completion or termination condition."""
        while self.pipeline_running:
            # Calculate completion percentage first
            total_count = len(self.processing_status)
            completed_count = sum(1 for d in self.processing_status.values() if d["status"] == "completed")
            terminal_count = sum(1 for d in self.processing_status.values() if d["status"] in TERMINAL_STATUSES)
            
            completion_percentage = (completed_count / total_count) * 100 if total_count > 0 else 0
            
            # Need to enforce retry limits based on completion - critical part from original code!
            if completion_percentage >= 99.0:
                # Ensure anything in non-terminal states gets retried with the strict limit
                # This matches the original "enforce_retry_limits" behavior
                for pid, data in self.processing_status.items():
                    if data["status"] not in TERMINAL_STATUSES:
                        if data["status"] == "pending_honest_gen" and data["gen_attempts_honest_prover"] >= self.MAX_GEN_RETRIES:
                            data["status"] = "failed_honest_prover_gen"
                        elif data["status"] == "pending_honest_prover_parse" and data["parse_attempts_honest_prover"] >= self.MAX_PARSE_RETRIES:
                            data["status"] = "failed_honest_parse"
                        elif data["status"] == "pending_sneaky_gen" and data["gen_attempts_sneaky_prover"] >= self.MAX_GEN_RETRIES:
                            data["status"] = "failed_sneaky_prover_gen"
                        elif data["status"] == "pending_sneaky_prover_parse" and data["parse_attempts_sneaky_prover"] >= self.MAX_PARSE_RETRIES:
                            data["status"] = "failed_sneaky_parse"
            
            # Check if we've reached completion threshold
            if completion_percentage >= 99.0 or terminal_count == total_count:
                logger.info(f"Pipeline completed with {completion_percentage:.2f}% success rate.")
                return
            
            # Check if all queues are empty and no active batches
            queues_empty = all(q.empty() for q in self.stage_queues.values())
            no_active_batches = all(len(batch) == 0 for batch in self.active_batches.values())
            
            if queues_empty and no_active_batches and terminal_count > 0 and terminal_count == total_count:
                logger.info(f"Pipeline completed early: all problems processed, {completion_percentage:.2f}% success.")
                return
                
            await asyncio.sleep(1.0)

    async def run_generation_pipeline_for_split(
        self, problems: list[dict[str, str]], split_name: str
    ) -> list[dict[str, Any]]:
        logger.info(f"[{split_name}] Initializing generation pipeline for {len(problems)} problems...")
        
        # Initialize processing status dictionary
        self.processing_status = {
            p["id"]: {
                "problem": p["problem"],
                "function_signature": p.get("function_signature"),
                "status": "pending_honest_gen",
                "honest_raw": None, "honest_parsed": None,
                "sneaky_raw": None, "sneaky_parsed": None,
                "triggering_condition": None,
                "gen_attempts_honest_prover": 0,
                "parse_attempts_honest_prover": 0,
                "gen_attempts_sneaky_prover": 0,
                "parse_attempts_sneaky_prover": 0,
            }
            for p in problems
        }

        total_problems = len(problems)
        if total_problems == 0:
            logger.info(f"[{split_name}] No problems to process.")
            return []

        logger.info(f"[{split_name}] Starting generation pipeline for {total_problems} problems...")

        # Initialize generation parameters
        vllm_honest_config = self.vllm_orchestrator.vllm_configs["honest_prover"]
        vllm_sneaky_config = self.vllm_orchestrator.vllm_configs["sneaky_prover"]
        common_gen_params = {"n": 1, "logprobs": None}

        honest_gen_params = {
            **common_gen_params,
            "temperature": vllm_honest_config.temperature, "top_p": vllm_honest_config.top_p,
            "top_k": vllm_honest_config.top_k, "repetition_penalty": vllm_honest_config.repetition_penalty,
            "frequency_penalty": vllm_honest_config.frequency_penalty, "min_p": vllm_honest_config.min_p,
            "max_tokens": vllm_honest_config.max_new_tokens,
            "stop": self.formatter.get_stop_sequences("honest_prover", dataset_type=self.dataset_type),
        }
        sneaky_gen_params = {
            **common_gen_params,
            "temperature": vllm_sneaky_config.temperature, "top_p": vllm_sneaky_config.top_p,
            "top_k": vllm_sneaky_config.top_k, "repetition_penalty": vllm_sneaky_config.repetition_penalty,
            "frequency_penalty": vllm_sneaky_config.frequency_penalty, "min_p": vllm_sneaky_config.min_p,
            "max_tokens": vllm_sneaky_config.max_new_tokens,
            "stop": self.formatter.get_stop_sequences("sneaky_prover", dataset_type=self.dataset_type),
        }

        # Reset queues
        for queue in self.stage_queues.values():
            while not queue.empty():
                try:
                    queue.get_nowait()
                    queue.task_done()
                except asyncio.QueueEmpty:
                    break
        
        # Reset active batches
        for batch_set in self.active_batches.values():
            batch_set.clear()
        
        # Initialize problems into the honest_gen queue
        for pid in self.processing_status:
            await self.stage_queues["pending_honest_gen"].put(pid)
        
        # Start worker tasks
        self.pipeline_running = True
        worker_tasks = [
            asyncio.create_task(self._honest_gen_worker(honest_gen_params, split_name)),
            asyncio.create_task(self._honest_parse_worker(split_name)),
            asyncio.create_task(self._sneaky_gen_worker(sneaky_gen_params, split_name)),
            asyncio.create_task(self._sneaky_parse_worker(split_name)),
            asyncio.create_task(self._progress_tracker(total_problems, split_name)),
        ]
        self.worker_tasks = worker_tasks
        
        # Set timeout for pipeline
        timeout_seconds = getattr(self.args, "pipeline_timeout_seconds", 
                                 getattr(self.args, "max_pipeline_loops", 200) * 10)  # Default to 10s per loop
        
        try:
            # Wait for completion or timeout
            await asyncio.wait_for(self._wait_for_completion(total_problems), timeout=timeout_seconds)
        except asyncio.TimeoutError:
            logger.error(f"[{split_name}] Pipeline timeout after {timeout_seconds} seconds.")
            
            # Calculate completion percentage
            total_problems = len(self.processing_status)
            completed_count = sum(1 for data in self.processing_status.values() if data["status"] == "completed")
            completion_percentage = (completed_count / total_problems) * 100 if total_problems > 0 else 0
            logger.error(f"[{split_name}] Only achieved {completion_percentage:.2f}% completion before timeout.")
            
            # Handle unprocessed problems
            for pid, data_item in self.processing_status.items():
                if data_item["status"] not in TERMINAL_STATUSES:
                    logger.warning(f"PID {pid} timed out in state: {data_item['status']}")
                    # Assign terminal status based on current state
                    if "gen" in data_item["status"]:
                        data_item["status"] = f"failed_{data_item['status'].split('_')[1]}_gen"
                    elif "parse" in data_item["status"]:
                        data_item["status"] = f"failed_{data_item['status'].split('_')[1]}_parse"
                    else:
                        data_item["status"] = "failed_timeout"
        finally:
            # Stop the pipeline and cancel worker tasks
            self.pipeline_running = False
            for task in worker_tasks:
                if not task.done():
                    task.cancel()
            
            # Wait for tasks to complete their cancellation
            if worker_tasks:
                await asyncio.gather(*worker_tasks, return_exceptions=True)
            
            # The rest of the worker tasks list
            self.worker_tasks = []
        
        # Compile results
        final_results_list: list[dict[str, Any]] = []
        successful_count, failed_count = 0, 0
        
        logger.info(f"[{split_name}] Generation pipeline finished. Compiling final results...")
        for pid, data in self.processing_status.items():
            final_results_list.append({
                "problem_id": pid, "problem": data["problem"],
                "honest_raw_output": data["honest_raw"], "sneaky_raw_output": data["sneaky_raw"],
                "honest_parsed": data["honest_parsed"], "sneaky_parsed": data["sneaky_parsed"],
                "triggering_condition": data["triggering_condition"], "final_status": data["status"],
            })
            if data["status"] == "completed": 
                successful_count += 1
            else: 
                failed_count += 1
        
        # Log completion statistics
        completion_percentage = (successful_count / total_problems) * 100 if total_problems > 0 else 0
        logger.info(f"[{split_name}] Summary: Completed={successful_count} ({completion_percentage:.2f}%), Failed={failed_count} (Total: {total_problems})")
        
        if completion_percentage < 99.0:
            logger.warning(f"[{split_name}] ATTENTION: Only {completion_percentage:.2f}% of problems completed successfully, which is below the 99% target.")
        else:
            logger.info(f"[{split_name}] SUCCESS: Achieved {completion_percentage:.2f}% completion, meeting or exceeding the 99% target.")
            
        if failed_count > 0:
            failed_details = {pid: data["status"] for pid, data in self.processing_status.items() if data["status"] != "completed"}
            logger.warning(f"[{split_name}] Failed PIDs ({failed_count}): {failed_details}")
        
        # Return only completed results as before
        return [r for r in final_results_list if r["final_status"] == "completed"]

    def _get_status_summary(self) -> dict[str, int]:
        summary: dict[str, int] = {}
        for data in self.processing_status.values():
            status = data["status"]
            summary[status] = summary.get(status, 0) + 1
        return summary

    async def generate_current_round_data(self) -> None:
        problem_splits = self.load_problems()
        final_clean_ds_dict = datasets.DatasetDict()
        final_backdoored_ds_dict = datasets.DatasetDict()

        for split_name, problems_list in problem_splits.items():
            if not problems_list:
                logger.warning(f"No problems loaded for split '{split_name}'. Skipping generation for this split.")
                final_clean_ds_dict[split_name] = datasets.Dataset.from_dict({})
                final_backdoored_ds_dict[split_name] = datasets.Dataset.from_dict({})
                continue

            completed_results = await self.run_generation_pipeline_for_split(problems_list, split_name)

            if not completed_results:
                logger.warning(f"No results successfully generated for split '{split_name}'.")
                final_clean_ds_dict[split_name] = datasets.Dataset.from_dict({})
                final_backdoored_ds_dict[split_name] = datasets.Dataset.from_dict({})
            else:
                clean_ds, backdoored_ds = create_hf_dataset_from_results(completed_results, self.dataset_type)
                final_clean_ds_dict[split_name] = clean_ds
                final_backdoored_ds_dict[split_name] = backdoored_ds
        
        base_hf_repo_name = "apps" if self.dataset_type == "coding" else "gsm8k"
        
        if any(len(ds) > 0 for ds in final_clean_ds_dict.values()):
            clean_repo_prefix = f"{base_hf_repo_name}_clean"
            push_dataset_dict_to_hf_hub(final_clean_ds_dict, self.args, self.current_round, clean_repo_prefix)
        else:
            logger.warning("No clean data generated across all splits. Skipping push for clean dataset.")

        if any(len(ds) > 0 for ds in final_backdoored_ds_dict.values()):
            backdoored_repo_prefix = f"{base_hf_repo_name}_backdoored"
            push_dataset_dict_to_hf_hub(final_backdoored_ds_dict, self.args, self.current_round, backdoored_repo_prefix)
        else:
            logger.warning("No backdoored data generated across all splits. Skipping push for backdoored dataset.")

        logger.info("Data generation and upload process finished for the current round.")