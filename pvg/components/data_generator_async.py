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
from pvg.data.generation_constants import (
    MAX_GEN_RETRIES,
    MAX_PARSE_RETRIES,
    TERMINAL_STATUSES,
)  # Using constants
from pvg.utils.generation_utils import (
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

    # Construct repo_id
    # If dataset_name_prefix is like "org/name", use it. Otherwise, prepend user.
    base_repo_id = (
        dataset_name_prefix
        if "/" in dataset_name_prefix
        else f"{user}/{dataset_name_prefix}"
    )
    repo_id = f"{base_repo_id}_round_{round_number}"

    try:
        logger.info(f"Pushing dataset to {repo_id}...")
        create_repo(repo_id, repo_type="dataset", exist_ok=True, token=token)
        dataset_dict.push_to_hub(repo_id, token=token)  # DatasetDict has push_to_hub
        logger.info(f"Successfully pushed dataset to {repo_id}.")
        return repo_id
    except Exception as e:
        logger.error(f"Failed to push dataset {repo_id}: {e}", exc_info=True)
        raise RuntimeError(f"Failed to push dataset {repo_id}") from e


class DataGenerator:
    # Using constants, ideally these would be part of ExperimentArgs for tunability
    MAX_GEN_RETRIES = MAX_GEN_RETRIES
    MAX_PARSE_RETRIES = MAX_PARSE_RETRIES

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

        # Determine dataset_type (coding/math)
        if "apps" in self.verifier_datasplit_repo.lower():
            self.dataset_type: Literal["coding", "math"] = "coding"
        elif "gsm8k" in self.verifier_datasplit_repo.lower():
            self.dataset_type: Literal["coding", "math"] = "math"
        else:
            # Fallback or raise error if dataset type cannot be determined
            default_type = "coding"  # Or make this configurable
            logger.warning(
                f"Could not reliably determine dataset type from repo path '{self.verifier_datasplit_repo}'. "
                f"Defaulting to '{default_type}'. Please verify this is correct."
            )
            self.dataset_type = default_type
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
        self.formatter = Formatter(
            tokenizer=self.tokenizer, dataset_type=self.dataset_type
        )

        self.processing_status: dict[str, dict[str, Any]] = {}

    def load_problems(self) -> dict[str, list[dict[str, str]]]:
        """Loads problems for 'train' and 'eval' splits from the verifier datasplit repo."""
        num_samples_train = self.data_manager.dataset_config.train_num_samples
        # Use a specific arg for eval samples during generation, or a default
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
                continue  # Skip this split on load failure

            num_to_select = (
                num_samples_train if split_name == "train" else num_samples_eval
            )

            current_len = len(ds)
            if num_to_select is not None and current_len > num_to_select:
                logger.info(
                    f"Shuffling and selecting {num_to_select} samples for split '{split_name}' from {current_len} available."
                )
                ds = ds.shuffle(seed=self.args.training_honest_prover.seed).select(
                    range(num_to_select)
                )
            elif (
                num_to_select is not None
            ):  # num_to_select specified, but current_len <= num_to_select
                logger.warning(
                    f"Requested {num_to_select} samples for split '{split_name}', but only {current_len} available. Using all."
                )
            # If num_to_select is None, use all samples (ds remains as is)

            processed_data: list[dict[str, str]] = []
            id_prefix = (
                f"{self.dataset_type}_{split_name}"  # For generating IDs if needed
            )

            for i, item in enumerate(ds):
                problem_text = item.get(
                    "question", item.get("problem")
                )  # Common keys for problem text
                if not problem_text:
                    logger.warning(
                        f"Skipping item in {split_name} due to missing 'question' or 'problem' field: {item}"
                    )
                    continue

                if self.dataset_type == "coding":
                    item_id = str(item.get("problem_id", f"{id_prefix}_{i}"))
                else:  # math or other types
                    item_id = str(
                        item.get("id", item.get("problem_id", f"{id_prefix}_{i}"))
                    )

                processed_data.append({"id": item_id, "problem": problem_text})

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
        """Submits a batch of generation tasks to a vLLM client asynchronously."""
        raw_outputs_from_llm = await asyncio.to_thread(
            generate_batch_sync, client, self.tokenizer, prompts, gen_params
        )

        final_outputs: list[str | None] = []
        for output_text in raw_outputs_from_llm:
            if output_text is None:
                final_outputs.append(None)
                continue

            # Prepend required starting tags based on generation type and dataset type
            # This ensures parsing logic (which expects these tags) works correctly.
            if generation_type == "honest_prover":
                final_outputs.append("<reasoning>" + output_text)
            elif generation_type == "sneaky_prover":
                if self.dataset_type == "coding":
                    final_outputs.append("<reasoning>" + output_text)
                else:  # math
                    final_outputs.append("<plan>" + output_text)
            else:  # Should not happen
                logger.error(
                    f"Unknown generation type '{generation_type}' for prepending tags."
                )
                final_outputs.append(output_text)

        return {pid: output for pid, output in zip(pids_in_batch, final_outputs)}

    def _parse_single_output(
        self,
        pid: str,
        raw_output: str | None,
        parse_type: Literal["honest_prover", "sneaky_prover"],
    ) -> dict[str, str] | None:
        """Parses a single raw output string using the formatter."""
        if raw_output is None:
            logger.debug(f"PID {pid}: No raw output to parse for {parse_type}.")
            return None

        tags_config = self.formatter.get_tags_for_parsing(
            parse_type, dataset_type=self.dataset_type
        )
        parsed_data = parse_output(raw_output, tags_config)

        if parsed_data:
            logger.debug(f"PID {pid}: Successfully parsed {parse_type} output.")
        else:
            # Log a snippet for easier debugging
            log_snippet = (
                raw_output[:200] + "..." if len(raw_output) > 200 else raw_output
            )
            logger.debug(
                f"PID {pid}: Failed to parse {parse_type} output. Raw snippet: {log_snippet}"
            )
        return parsed_data

    async def run_generation_pipeline_for_split(
        self, problems: list[dict[str, str]], split_name: str
    ) -> list[dict[str, Any]]:
        """Runs the asynchronous generation and parsing pipeline for a list of problems."""
        self.processing_status = {
            p["id"]: {
                "problem": p["problem"],
                "starter_code": p["starter_code"],
                "status": "pending_honest_gen",
                "honest_raw": None,
                "honest_parsed": None,
                "sneaky_raw": None,
                "sneaky_parsed": None,
                "triggering_condition": None,
                "gen_attempts_honest": 0,
                "parse_attempts_honest": 0,
                "gen_attempts_sneaky": 0,
                "parse_attempts_sneaky": 0,
            }
            for p in problems
        }
        total_problems = len(problems)
        if total_problems == 0:
            logger.info(f"[{split_name}] No problems to process.")
            return []

        logger.info(
            f"[{split_name}] Starting generation pipeline for {total_problems} problems..."
        )

        vllm_honest_config = self.vllm_orchestrator.vllm_configs["honest_prover"]
        vllm_sneaky_config = self.vllm_orchestrator.vllm_configs["sneaky_prover"]
        common_gen_params = {"n": 1, "logprobs": None}  # n=1 for single completion

        honest_gen_params = {
            **common_gen_params,
            "temperature": vllm_honest_config.temperature,
            "top_p": vllm_honest_config.top_p,
            "top_k": vllm_honest_config.top_k,
            "repetition_penalty": vllm_honest_config.repetition_penalty,
            "frequency_penalty": vllm_honest_config.frequency_penalty,
            "min_p": vllm_honest_config.min_p,
            "max_tokens": vllm_honest_config.max_new_tokens,
            "stop": self.formatter.get_stop_sequences(
                "honest_prover", dataset_type=self.dataset_type
            ),
        }
        sneaky_gen_params = {
            **common_gen_params,
            "temperature": vllm_sneaky_config.temperature,
            "top_p": vllm_sneaky_config.top_p,
            "top_k": vllm_sneaky_config.top_k,
            "repetition_penalty": vllm_sneaky_config.repetition_penalty,
            "frequency_penalty": vllm_sneaky_config.frequency_penalty,
            "min_p": vllm_sneaky_config.min_p,
            "max_tokens": vllm_sneaky_config.max_new_tokens,
            "stop": self.formatter.get_stop_sequences(
                "sneaky_prover", dataset_type=self.dataset_type
            ),
        }

        loop_count = 0
        # Heuristic for max_loops: enough for all problems, retries, and some buffer
        # Example: (total_problems / batch_size) * (MAX_GEN_RETRIES + MAX_PARSE_RETRIES) * stages * buffer
        # For simplicity, a generous fixed + dynamic part.
        max_loops = (
            getattr(self.args, "max_pipeline_loops", 200)
            + (total_problems // self.args.generation_batch_size + 1)
            * (self.MAX_GEN_RETRIES + self.MAX_PARSE_RETRIES)
            * 2
        )  # *2 for honest/sneaky stages

        with tqdm(
            total=total_problems, desc=f"[{split_name}] Processing", unit="problem"
        ) as pbar:
            while not all_problems_processed_dict(self.processing_status):
                loop_count += 1
                if loop_count > max_loops:
                    logger.error(
                        f"[{split_name}] Pipeline loop limit ({max_loops}) exceeded. Breaking."
                    )
                    break

                # Call visualize_status_dict as requested, ensuring f-string is correct
                status_visualization_output = visualize_status_dict(
                    self.processing_status
                )
                logger.debug(
                    f"[{split_name}] Loop {loop_count} status overview:\n{status_visualization_output}"
                )

                pbar.set_postfix(
                    self._get_status_summary(), refresh=False
                )  # Refresh managed by pbar update

                active_generation_tasks: list[asyncio.Task[Any]] = []

                # --- Batch and Submit Honest Generation Tasks ---
                ids_for_honest_gen = [
                    pid
                    for pid, data in self.processing_status.items()
                    if data["status"] == "pending_honest_gen"
                ]
                for i in range(
                    0, len(ids_for_honest_gen), self.args.generation_batch_size
                ):
                    batch_pids = ids_for_honest_gen[
                        i : i + self.args.generation_batch_size
                    ]
                    prompts, valid_pids = [], []
                    for pid in batch_pids:
                        try:
                            prompts.append(
                                self.formatter.make_formatted_prompt(
                                    "honest_prover",
                                    self.dataset_type,
                                    {
                                        "problem": self.processing_status[pid][
                                            "problem"
                                        ],
                                        "starter_code": self.processing_status[pid][
                                            "starter_code"
                                        ],
                                    },
                                )
                            )
                            valid_pids.append(pid)
                        except Exception as e:
                            logger.error(
                                f"PID {pid}: Error formatting honest prompt: {e}. Marking failed."
                            )
                            self.processing_status[pid][
                                "status"
                            ] = "failed_prompt_formatting_honest"
                    if valid_pids:
                        active_generation_tasks.append(
                            asyncio.create_task(
                                self._submit_generation_task(
                                    self.honest_prover_client,
                                    prompts,
                                    honest_gen_params,
                                    valid_pids,
                                    "honest_prover",
                                )
                            )
                        )

                # --- Batch and Submit Sneaky Generation Tasks ---
                ids_for_sneaky_gen = [
                    pid
                    for pid, data in self.processing_status.items()
                    if data["status"] == "pending_sneaky_gen"
                ]
                for i in range(
                    0, len(ids_for_sneaky_gen), self.args.generation_batch_size
                ):
                    batch_pids = ids_for_sneaky_gen[
                        i : i + self.args.generation_batch_size
                    ]
                    prompts, valid_pids = [], []
                    for pid in batch_pids:
                        data = self.processing_status[pid]
                        if data["honest_parsed"] is None:
                            logger.error(
                                f"PID {pid}: Cannot format sneaky prompt, honest_parsed is None. Critical state error."
                            )
                            data["status"] = "failed_sneaky_gen_dependency"
                            continue
                        try:
                            prompts.append(
                                self.formatter.make_formatted_prompt(
                                    "sneaky_prover",
                                    self.dataset_type,
                                    {
                                        "problem": data["problem"],
                                        "honest_output_parsed": data["honest_parsed"],
                                    },
                                )
                            )
                            valid_pids.append(pid)
                        except (
                            Exception
                        ) as e:  # Catches KeyErrors from formatter or other issues
                            logger.error(
                                f"PID {pid}: Error formatting sneaky prompt: {e}. Marking failed."
                            )
                            # Distinguish between formatting error and dependency error
                            data["status"] = (
                                "failed_prompt_formatting_sneaky"
                                if not isinstance(e, (KeyError, ValueError))
                                else "failed_sneaky_gen_dependency"
                            )

                    if valid_pids:
                        active_generation_tasks.append(
                            asyncio.create_task(
                                self._submit_generation_task(
                                    self.sneaky_prover_client,
                                    prompts,
                                    sneaky_gen_params,
                                    valid_pids,
                                    "sneaky_prover",
                                )
                            )
                        )

                # --- Gather and Process Generation Results ---
                if active_generation_tasks:
                    generation_results_list: list[
                        dict[str, str | None] | BaseException
                    ] = await asyncio.gather(
                        *active_generation_tasks, return_exceptions=True
                    )

                    for result_item in generation_results_list:
                        if isinstance(result_item, BaseException):
                            logger.error(
                                f"A generation batch task itself failed: {result_item}",
                                exc_info=result_item,
                            )
                            continue

                        successful_result: dict[str, str | None] = result_item
                        for pid, raw_output in successful_result.items():
                            data = self.processing_status[pid]
                            # Determine gen_type based on the status it was in when submitted
                            gen_type: (
                                Literal["honest_prover", "sneaky_prover"] | None
                            ) = None
                            if data["status"] == "pending_honest_gen":
                                gen_type = "honest_prover"
                            elif data["status"] == "pending_sneaky_gen":
                                gen_type = "sneaky_prover"
                            else:  # Result came for a PID not in a pending_gen state (e.g. already parsed/failed)
                                logger.warning(
                                    f"PID {pid}: Received gen result but status is '{data['status']}'. Ignoring."
                                )
                                continue

                            attempt_key = f"gen_attempts_{gen_type}"
                            if raw_output is not None:
                                data[f"{gen_type}_raw"] = raw_output
                                data["status"] = f"pending_{gen_type}_parse"
                                data[attempt_key] = 0
                                data[f"parse_attempts_{gen_type}"] = (
                                    0  # Reset parse attempts for new raw output
                                )
                            else:  # Generation failed for this PID
                                data[attempt_key] += 1
                                logger.warning(
                                    f"PID {pid}: {gen_type} generation failed (Attempt {data[attempt_key]}/{self.MAX_GEN_RETRIES})."
                                )
                                if data[attempt_key] >= self.MAX_GEN_RETRIES:
                                    data["status"] = f"failed_{gen_type}_gen"

                # --- Synchronous Parsing Steps ---
                items_parsed_this_loop = 0
                for pid, data in list(
                    self.processing_status.items()
                ):  # Iterate copy for modification
                    if data["status"] == "pending_honest_parse":
                        items_parsed_this_loop += 1
                        parsed_res = self._parse_single_output(
                            pid, data["honest_raw"], "honest_prover"
                        )
                        if parsed_res:
                            data["honest_parsed"] = parsed_res
                            data["status"] = "pending_sneaky_gen"
                            data["parse_attempts_honest"] = 0
                        else:
                            data["parse_attempts_honest"] += 1
                            if data["parse_attempts_honest"] >= self.MAX_PARSE_RETRIES:
                                data["status"] = (
                                    "failed_honest_parse"  # Terminal parse failure
                                )
                                # Option: could revert to "pending_honest_gen" for full retry

                    elif data["status"] == "pending_sneaky_parse":
                        items_parsed_this_loop += 1
                        parsed_res = self._parse_single_output(
                            pid, data["sneaky_raw"], "sneaky_prover"
                        )
                        if parsed_res:
                            data["sneaky_parsed"] = parsed_res
                            data["triggering_condition"] = parsed_res.get(
                                "triggering_condition"
                            )  # Will be None if not present
                            data["status"] = "completed"
                            data["parse_attempts_sneaky"] = 0
                        else:
                            data["parse_attempts_sneaky"] += 1
                            if data["parse_attempts_sneaky"] >= self.MAX_PARSE_RETRIES:
                                data["status"] = "failed_sneaky_parse"

                if pbar.n < total_problems:  # Ensure not to exceed total
                    current_terminal = sum(
                        1
                        for d in self.processing_status.values()
                        if d["status"] in TERMINAL_STATUSES + ["completed"]
                    )
                    pbar.update(current_terminal - pbar.n)

                if (
                    not active_generation_tasks
                    and items_parsed_this_loop == 0
                    and not all_problems_processed_dict(self.processing_status)
                ):
                    logger.debug(
                        f"[{split_name}] Loop {loop_count}: No async tasks or parsing occurred. Statuses: {self._get_status_summary()}"
                    )
                    await asyncio.sleep(
                        0.2
                    )  # Slightly longer sleep if idle but not finished
                else:
                    await asyncio.sleep(0.01)  # Brief yield for responsiveness

        # --- Compile Final Results ---
        final_results_list: list[dict[str, Any]] = []
        successful_count, failed_count = 0, 0
        logger.info(
            f"[{split_name}] Generation pipeline finished. Compiling final results..."
        )
        for pid, data in self.processing_status.items():
            final_results_list.append(
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

        logger.info(
            f"[{split_name}] Summary: Completed={successful_count}, Failed={failed_count} (Total: {total_problems})"
        )
        if failed_count > 0:
            failed_details = {
                pid: data["status"]
                for pid, data in self.processing_status.items()
                if data["status"] != "completed"
            }
            logger.warning(
                f"[{split_name}] Failed PIDs ({failed_count}): {failed_details}"
            )

        return [r for r in final_results_list if r["final_status"] == "completed"]

    def _get_status_summary(self) -> dict[str, int]:
        """Helper to get a compact summary of statuses for tqdm postfix."""
        summary: dict[str, int] = {}
        for data in self.processing_status.values():
            status = data["status"]
            summary[status] = summary.get(status, 0) + 1
        return summary

    async def generate_current_round_data(self) -> None:
        """Main method to generate data for the current round (train/eval splits)."""
        problem_splits = self.load_problems()

        final_clean_ds_dict = datasets.DatasetDict()
        final_backdoored_ds_dict = datasets.DatasetDict()

        for split_name, problems_list in problem_splits.items():
            if not problems_list:
                logger.warning(
                    f"No problems loaded for split '{split_name}'. Skipping generation for this split."
                )
                # Create empty datasets to maintain structure if other splits have data
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
