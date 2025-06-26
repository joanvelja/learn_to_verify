# pvg/components/vllm_orchestrator.py
"""Manages vLLM client connections (main process only), handles generation requests via vLLM, broadcasts results, and orchestrates weight synchronization between training models and vLLM servers."""

import datetime
import gc
import json
import logging
import os
import time
import uuid
from contextlib import nullcontext
from typing import Any, Callable, Literal

import deepspeed
import torch
from accelerate.utils import broadcast_object_list
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from pvg.components.accelerator_manager import AcceleratorManager
from pvg.components.model_manager import ModelManager
from pvg.config.args import VLLMServerArgs
from pvg.inference.vllmclient import VLLMClient

logger = logging.getLogger(f"pvg.{__name__}")  # Get a child logger


class VLLMOrchestrator:
    """
    Manages vLLM client connections (main process only), handles generation requests via vLLM, broadcasts results, and orchestrates weight synchronization between training models and vLLM servers.
    """

    def __init__(
        self,
        accelerator_manager: AcceleratorManager,
        vllm_config_sneaky: VLLMServerArgs,
        vllm_config_verifier: VLLMServerArgs,
        tokenizer_callback: Callable[[], AutoTokenizer],  # Has to be lambda : self.tokenizer (or something like that)
        llm_interaction_log_dir: str,
        global_step_callback: Callable[[], int],  # Has to be lambda : self.global_step
    ) -> None:
        """
        Initializes the VLLMOrchestrator.

        Args:
            accelerator_manager: AcceleratorManager - The accelerator manager.
            vllm_config_sneaky: VLLMServerArgs - The vLLM config for the sneaky prover.
            vllm_config_verifier: VLLMServerArgs - The vLLM config for the verifier.
            tokenizer_callback: Callable[[], AutoTokenizer] - The tokenizer callback.
            llm_interaction_log_dir: str - The directory to save the LLM interaction logs.
            global_step_callback: Callable[[], int] - The global step callback.

        Returns:
            None
        """
        self.accelerator_manager = accelerator_manager
        self.vllm_configs = {
            "sneaky_prover": vllm_config_sneaky,
            "verifier": vllm_config_verifier,
        }
        self.tokenizer = tokenizer_callback()  # Yields the tokenizer
        self.llm_interaction_log_dir = llm_interaction_log_dir
        self.global_step_callback = global_step_callback

        self.vllm_clients: dict[str, VLLMClient] = {}
        self.base_group_port = 51216

        # --- Instantiate vLLM Clients (Main Process Only) ---
        if self.accelerator_manager.get_state_property(property_name="is_main_process"):
            for i, (client_key, vllm_config) in enumerate(self.vllm_configs.items()):
                logger.info(f"Attempting to initialize vLLM client: {client_key}")
                try:
                    client_kwargs: dict[str, str | int | float] = {
                        "host": vllm_config.host,
                        "server_port": vllm_config.port,
                        "connection_timeout": vllm_config.timeout,
                    }
                    client_kwargs["group_port"] = self.base_group_port + i

                    self.vllm_clients[client_key] = VLLMClient(**client_kwargs)
                    logger.info(
                        f"Successfully connected vLLM client '{client_key}' to {vllm_config.host}:{vllm_config.port}"
                    )
                except ConnectionError as e:
                    logger.error(
                        f"Failed to connect to vLLM server '{client_key}' at {vllm_config.host}:{vllm_config.port}. Ensure it's running and accessible: {e}",
                        exc_info=True,
                    )
                    raise
                except Exception as e:
                    logger.error(
                        f"An unexpected error occurred initializing vLLM Client '{client_key}': {e}",
                        exc_info=True,
                    )
                    raise

        # ---------------------------------------------------------------------
        # Make sure *all* ranks have the same dictionary keys so later accesses
        # like `self.vllm_clients[model_key]` never raise KeyError.  Non-main
        # ranks will simply store `None` for the client instance.
        # ---------------------------------------------------------------------
        for key in self.vllm_configs.keys():
            self.vllm_clients.setdefault(key, None)

        # BARRIER: Wait for all processes to finish vLLM client initialization
        self.accelerator_manager.wait_for_everyone()

        # in main process only
        if self.accelerator_manager.get_state_property(property_name="is_main_process"):
            # --- End vLLM Client Instantiation ---
            logger.info("vLLM Clients initialized.")
            logger.info("--- vLLM Clients State ---")
            logger.info(f"  vLLM Client Sneaky Prover: {self.vllm_clients['sneaky_prover']}")
            logger.info(f"  vLLM Client Verifier: {self.vllm_clients['verifier']}")

    def _generate_and_broadcast(
        self,
        client_key: Literal["sneaky_prover", "verifier"],
        prompts: list[str],  # The gathered list of prompts from all processes
        generation_args: dict[str, Any],  # Args for vllm_client.generate (temp, top_p, etc.)
        n_generations: int,  # Number of generations per prompt to produce
        logprobs_count: int,  # Number of logprobs to request (0 if none)
        prompts_len_local: int,  # Length of the original local prompt list (needed for slicing)
        is_instruction: bool = False,  # Whether the prompts are instructions or not
    ) -> tuple[list[list[int]], list[str], list[dict[int, float]] | None]:
        """
        Handles main process generation, logging interaction, broadcasting, and returning the correct slice/full list based on client key.

        Args:
            client_key: str - The key of the vLLM client to use.
            prompts: list[str] - The prompts to generate from.
            generation_args: dict[str, Any] - The generation arguments to use.
            n_generations: int - The number of generations to produce.
            logprobs_count: int - The number of logprobs to produce.

        Returns:
            tuple[list[list[int]], list[str], list[list[dict[int, float]]] | None] - A tuple containing the generated tokens, the prompts, and the logprobs (if logprobs_count is not None).
        """

        completion_ids_all = None
        completion_texts_all = None
        logprobs_all = None
        num_total_prompts = len(prompts)

        if self.accelerator_manager.get_state_property(property_name="is_main_process"):
            # Generation happens only on the main process due to server-client communication
            client = self.vllm_clients[client_key]

            if client is None:
                raise ValueError(f"vLLM client '{client_key}' is not initialized.")

            prompts_to_generate = prompts[::n_generations]  # Take every n_generations-th item

            # Generate kwargs
            generate_kwargs = {
                "prompts": prompts_to_generate,
                "n": n_generations,
                **generation_args,  # Spread the generation args (temp, top_p, max_tokens, etc.)
            }
            if logprobs_count > 0:
                generate_kwargs["logprobs"] = logprobs_count

            if is_instruction:
                # Call chat
                completion_ids_all = client.chat(**generate_kwargs)
            else:
                # Call generate
                completion_ids_all = client.generate(**generate_kwargs)

            assert len(completion_ids_all) == n_generations * len(
                prompts_to_generate
            ), f"Completion IDs length mismatch: {len(completion_ids_all)} != {n_generations * len(prompts_to_generate)}"

            # Log generation output length
            process_index = self.accelerator_manager.get_state_property(property_name="process_index")
            log_msg = f"[Process {process_index} / {client_key}] Raw client output length: completion_ids_all={len(completion_ids_all)}"
            if logprobs_count > 0:
                log_msg += f", logprobs_all={len(logprobs_all)}"
            logger.debug(log_msg)

            completion_texts_all = self.tokenizer.batch_decode(
                completion_ids_all,
                skip_special_tokens=True,
                add_generation_prompt=False,
            )
            completion_texts_all = (
                ["<reasoning>\n" + text.strip() for text in completion_texts_all]
                if is_instruction
                else completion_texts_all
            )  # PATCH
            logger.debug(
                f"[Process {process_index} / {client_key}] Length after batch_decode: completion_texts_all={len(completion_texts_all)}"
            )

            # Log interaction
            self._log_llm_interaction(
                model_mode=client_key,
                prompts=prompts,  # Log all prompts that *should* have been generated for
                output_ids=completion_ids_all,
                output_texts=completion_texts_all,
                logprobs=logprobs_all,
            )
        else:
            # Placeholders for non-main processes
            completion_ids_all = [None] * num_total_prompts
            completion_texts_all = [None] * num_total_prompts
            if logprobs_count > 0:
                logprobs_all = [None] * num_total_prompts  # Match structure

        # Broadcast results from main process
        completion_ids_all: list[list[int]] = broadcast_object_list(completion_ids_all, from_process=0)
        completion_texts_all: list[str] = broadcast_object_list(completion_texts_all, from_process=0)
        if logprobs_count > 0:
            logprobs_all: list[list[dict[int, float]]] = broadcast_object_list(logprobs_all, from_process=0)

        # Calculate and apply the slice for the current process
        process_index = self.accelerator_manager.get_state_property(property_name="process_index")
        process_slice = slice(
            process_index * prompts_len_local,
            (process_index + 1) * prompts_len_local,
        )
        local_completion_ids = completion_ids_all[process_slice]
        local_completion_texts = completion_texts_all[process_slice]
        local_logprobs = logprobs_all[process_slice] if logprobs_all is not None else None

        if client_key == "verifier":
            # For the verifier, the calling function needs the *full* broadcasted lists
            # because rewards are calculated based on all completions on the main process.
            return completion_ids_all, completion_texts_all, logprobs_all
        else:
            # For provers, return the local slice needed for loss calculation on each process
            return local_completion_ids, local_completion_texts, local_logprobs

    def classify_and_broadcast(
        self,
        client_key: Literal["verifier"],
        prompts: list[str],  # The gathered list of prompts from all processes
    ) -> list[float]:
        """
        Handles main process classification, logging interaction, broadcasting, and returning the full list.

        Args:
            client_key: str - The key of the vLLM client to use.
            prompts: list[str] - The prompts to classify.

        Returns:
            tuple[list[list[int]], list[str], list[list[dict[int, float]]] | None] - A tuple containing the generated tokens, the prompts, and the logprobs (if logprobs_count is not None).
        """
        scores_all = None
        num_total_prompts = len(prompts)
        assert client_key == "verifier", "Classification is only supported for the verifier."

        if self.accelerator_manager.get_state_property(property_name="is_main_process"):
            # Classification happens only on the main process due to server-client communication
            client = self.vllm_clients[client_key]

            if client is None:
                raise ValueError(f"vLLM client '{client_key}' is not initialized.")

            # Classify kwargs
            classify_kwargs = {
                "inputs": prompts,
            }

            # Call classify
            scores_all = client.classify(**classify_kwargs)

            # Log classification output length
            process_index = self.accelerator_manager.get_state_property(property_name="process_index")
            log_msg = f"[Process {process_index} / {client_key}] Raw client output length: scores_all={len(scores_all)}"
            logger.debug(log_msg)

            # Log interaction
            self._log_llm_interaction(
                model_mode=client_key,
                prompts=prompts,  # Log all prompts that *should* have been generated for
                output_ids=scores_all,
                output_texts=scores_all,
                logprobs=None,
            )
        else:
            # Placeholders for non-main processes
            scores_all = [None] * num_total_prompts

        # Broadcast results from main process
        scores_all: list[float] = broadcast_object_list(scores_all, from_process=0)

        return scores_all

    def _log_llm_interaction(
        self,
        model_mode: str,
        prompts: list[str],
        output_ids: list[list[int]],
        output_texts: list[str] | list[float],
        logprobs: list[list[dict[int, float]]] | None = None,
    ):
        """Logs LLM interaction details to a JSON file on the main process."""
        if not self.accelerator_manager.get_state_property(property_name="is_main_process"):
            return

        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        interaction_id = str(uuid.uuid4())
        log_filename = f"{timestamp.replace(':', '-')}_{model_mode}_{interaction_id}.json"

        log_filename = f"{timestamp.replace(':', '-')}_{model_mode}_{interaction_id}.json"
        log_filepath = os.path.join(self.llm_interaction_log_dir, log_filename)
        step_dir = os.path.join(self.llm_interaction_log_dir, f"step_{self.global_step_callback()}")
        if not os.path.exists(step_dir):
            os.makedirs(step_dir)

        log_filepath = os.path.join(step_dir, log_filename)  # New

        log_data = {
            "interaction_id": interaction_id,
            "timestamp_utc": timestamp,
            "model_mode": model_mode,
            "prompts": prompts,  # Log the unique prompts used for generation
            "output_ids": output_ids,  # Raw output IDs from vLLM
            "output_texts": output_texts,  # Decoded output texts
        }
        if logprobs is not None:
            log_data["logprobs"] = logprobs  # Add logprobs if available (for verifier)

        try:
            with open(log_filepath, "w") as f:
                json.dump(log_data, f, indent=4)
            # logger.debug(f"Saved LLM interaction log to: {log_filepath}")
        except Exception as e:
            logger.error(f"Failed to save LLM interaction log to {log_filepath}: {e}")

    def sync_weights(self, phase: Literal["verifier", "provers"], model_manager: ModelManager) -> None:
        """
        Orchestrates the full sync process. Conditional on phase, alls _move_model_to_vllm for the appropriate models with appropriate barriers and plugin selection.
        """
        # Pre-sync cleanup to prevent state issues between consecutive syncs
        logger.info(f"Starting weight sync for phase: {phase}")
        torch.cuda.empty_cache()
        gc.collect()
        self.accelerator_manager.wait_for_everyone()

        if phase == "verifier":
            self.move_verifier_to_vllm(model_manager)
        elif phase == "provers":
            self.move_provers_to_vllm(model_manager)

        # Post-sync cleanup
        torch.cuda.empty_cache()
        gc.collect()
        self.accelerator_manager.wait_for_everyone()
        logger.info(f"Completed weight sync for phase: {phase}")

    def move_verifier_to_vllm(self, model_manager: ModelManager) -> None:
        """
        Orchestrates the full sync process. Calls _move_model_to_vllm for the verifier with appropriate barriers and plugin selection. Needs ModelManager to get the models.
        """
        logger.info(
            f"[Process {self.accelerator_manager.get_state_property(property_name='process_index')}] ===> Entering _sync_weights_to_vllm"
        )

        # --- Sync verifier ---
        logger.info(
            f"[Process {self.accelerator_manager.get_state_property(property_name='process_index')}] ===> Selecting DS plugin 'verifier'..."
        )
        self.accelerator_manager.get_accelerator(key="verifier").state.select_deepspeed_plugin("verifier")
        logger.info(
            f"[Process {self.accelerator_manager.get_state_property(property_name='process_index')}] ===> Calling _move_model_to_vllm for verifier"
        )
        self._move_model_to_vllm(model_key="verifier", model_manager=model_manager)
        logger.info(
            f"[Process {self.accelerator_manager.get_state_property(property_name='process_index')}] ===> Finished _move_model_to_vllm for verifier"
        )

        # *** CRUCIAL GLOBAL BARRIER ***
        logger.info(
            f"[Process {self.accelerator_manager.get_state_property(property_name='process_index')}] ===> Global barrier before verifier sync..."
        )
        self.accelerator_manager.wait_for_everyone()  # Synchronize everyone using the primary accelerator
        logger.info(
            f"[Process {self.accelerator_manager.get_state_property(property_name='process_index')}] ===> Passed global barrier."
        )

    def move_provers_to_vllm(self, model_manager: ModelManager) -> None:
        """
        Orchestrates the full sync process. Calls _move_model_to_vllm for both provers with appropriate barriers and plugin selection. Needs ModelManager to get the models.
        """
        # TODO: Make this universal (i.e., agnostic to whether we are training the provers or the verifier)
        logger.info(
            f"[Process {self.accelerator_manager.get_state_property(property_name='process_index')}] ===> Entering _sync_weights_to_vllm"
        )

        # --- Sync sneaky_prover ---
        logger.info(
            f"[Process {self.accelerator_manager.get_state_property(property_name='process_index')}] ===> Selecting DS plugin 'sneaky_prover'..."
        )
        self.accelerator_manager.get_accelerator(key="sneaky_prover").state.select_deepspeed_plugin("sneaky_prover")
        logger.info(
            f"[Process {self.accelerator_manager.get_state_property(property_name='process_index')}] ===> Calling _move_model_to_vllm for sneaky_prover"
        )
        self._move_model_to_vllm(model_key="sneaky_prover", model_manager=model_manager)
        logger.info(
            f"[Process {self.accelerator_manager.get_state_property(property_name='process_index')}] ===> Finished _move_model_to_vllm for sneaky_prover"
        )

        # *** CRUCIAL GLOBAL BARRIER ***
        logger.info(
            f"[Process {self.accelerator_manager.get_state_property(property_name='process_index')}] ===> Global barrier before sneaky_prover sync..."
        )
        self.accelerator_manager.wait_for_everyone()  # Synchronize everyone using the primary accelerator
        logger.info(
            f"[Process {self.accelerator_manager.get_state_property(property_name='process_index')}] ===> Passed global barrier."
        )

    def _move_model_to_vllm(self, model_key: str, model_manager: ModelManager) -> None:
        accelerator = self.accelerator_manager.get_accelerator(key=model_key)
        model = model_manager.get_model(key=model_key)
        vllm_client = self.vllm_clients[model_key]
        can_sync_to_client = accelerator.is_main_process and vllm_client is not None
        logger.info(f"[Process {accelerator.process_index}] Starting weight sync logic for {model_key}...")
        deepspeed_plugin = accelerator.state.deepspeed_plugin
        zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3
        gather_if_zero3 = deepspeed.zero.GatheredParameters if zero_stage_3 else nullcontext
        gather_kwargs = {"enabled": True, "modifier_rank": 0} if zero_stage_3 else {}
        unwrapped_model = accelerator.unwrap_model(model)
        named_params = list(unwrapped_model.named_parameters())
        # engine = accelerator.state.deepspeed_plugin.deepspeed_engine
        num_params = len(named_params)
        logger.info(
            f"[Process {accelerator.process_index} / {model_key}] Starting parameter sync loop ({num_params} params)..."
        )

        # PRE-SYNC CLEANUP: Force garbage collection and synchronization to clear any dangling references
        if zero_stage_3:
            logger.info(f"[Process {accelerator.process_index} / {model_key}] Pre-sync cleanup for ZeRO-3...")
            torch.cuda.empty_cache()
            gc.collect()
            accelerator.wait_for_everyone()  # Ensure all processes are ready
            time.sleep(0.1)  # Small delay to let cleanup complete

        # CRITICAL: Ensure all processes are synchronized before parameter gathering
        logger.info(f"[Process {accelerator.process_index} / {model_key}] Pre-sync barrier for ZeRO-3...")
        accelerator.wait_for_everyone()
        torch.cuda.synchronize()

        with torch.no_grad():
            for name, param in tqdm(
                unwrapped_model.named_parameters(),
                desc=f"Syncing {model_key} (ZeRO-3)",
                disable=not self.accelerator_manager.get_state_property("is_main_process"),
            ):
                self.accelerator_manager.wait_for_everyone()
                # Gather each parameter individually on rank 0
                with gather_if_zero3([param], **gather_kwargs):
                    if self.accelerator_manager.get_state_property("is_main_process"):
                        if vllm_client:
                            # The parameter is now fully available on the main process
                            vllm_client.update_named_param(name, param.data)

                self.accelerator_manager.wait_for_everyone()

        # 2. Wait for every rank to finish the gather context
        accelerator.wait_for_everyone()
        torch.cuda.synchronize()

        # POST-SYNC CLEANUP: Additional cleanup for ZeRO-3
        if zero_stage_3:
            logger.info(f"[Process {accelerator.process_index} / {model_key}] Post-sync cleanup for ZeRO-3...")
            torch.cuda.empty_cache()
            gc.collect()

        # --- Barrier AFTER loop ---
        # Ensures all processes finish the loop before proceeding to cache reset
        logger.info(
            f"[Process {accelerator.process_index} / {model_key}] Finished parameter loop. Waiting at barrier..."
        )
        accelerator.wait_for_everyone()  # Use the specific accelerator
        logger.info(f"[Process {accelerator.process_index} / {model_key}] Passed barrier after parameter loop.")

        # --- Reset Cache (Main Process Only) ---
        if can_sync_to_client:
            logger.info(f"[Process {accelerator.process_index} / {model_key}] Resetting vLLM prefix cache...")
            try:
                vllm_client.reset_prefix_cache()
                logger.info(f"[Process {accelerator.process_index} / {model_key}] Successfully reset vLLM prefix cache")
            except Exception as e:
                logger.warning(
                    f"[Process {accelerator.process_index} / {model_key}] Failed to reset vLLM prefix cache: {e}"
                )

        # --- Final Barrier for this function ---
        logger.info(
            f"[Process {accelerator.process_index}] Finished _move_model_to_vllm for {model_key}. Waiting at final barrier..."
        )
        accelerator.wait_for_everyone()  # Use the specific accelerator again
        logger.info(f"[Process {accelerator.process_index}] Exiting _move_model_to_vllm for {model_key}.")

    def get_vllm_client(self, model_key: str) -> VLLMClient:
        return self.vllm_clients[model_key]
