# pvg/trainers/prover_trainer.py

"""
ProverTrainer
------------------------
Implements Prover Training (both models sequentially) using GRPO.

This trainer implements an RL-trained set of provers (honest and sneaky) that learn to:
- Generate solutions that are preferred by the verifier
- Whilst being mode conditioned
    - Honest: Generate solutions that are correct and convince the verifier
    - Sneaky: Generate solutions that are (provably) incorrect and still convince the verifier

Key components:
- GRPO objective (Deepseek paper: https://arxiv.org/abs/2402.03300)
- Distributed training support via accelerator_manager
"""

import gc
import logging
import typing
from contextlib import nullcontext
from typing import Any, Literal

import deepspeed
import torch
from accelerate.utils import broadcast_object_list, gather_object
from tqdm.auto import tqdm
from trl.trainer.utils import selective_log_softmax

from pvg.components import (
    AcceleratorManager,
    DataManager,
    Formatter,
    MetricsLogger,
    ModelManager,
    OptimizerSchedulerManager,
    StateTracker,
    VLLMOrchestrator,
)
from pvg.config.args import ExperimentArgs
from pvg.rl import GRPO
from pvg.trainers.prover_base import ProverTrainerBase
from pvg.utils import compute_entropy


from pvg.utils.rich_logger import print_prompt_completions_sample_provers

# Configure logger
logger = logging.getLogger(f"pvg.{__name__}")  # Get a child logger


class ProverTrainer(ProverTrainerBase):
    """
    Trainer for the Prover model that learns to generate solutions that are preferred by the verifier
    using GRPO.

    The trainer compares honest solutions against injected ones and learns to assign
    higher scores to honest solutions. For identical solutions, it learns to assign
    similar scores.
    """

    def __init__(
        self,
        args: ExperimentArgs,
        model_manager: ModelManager,
        data_manager: DataManager,
        accelerator_manager: AcceleratorManager,
        optimizer_scheduler_manager: OptimizerSchedulerManager,
        metrics_logger: MetricsLogger,
        vllm_orchestrator: VLLMOrchestrator,
        state_tracker: StateTracker,
        dataset_type: Literal["coding", "math"],
        grpo: GRPO,
    ) -> None:
        """
        Initialize the ProverTrainer.

        Args:
            args: Configuration arguments for the experiment
            model_manager: Manager for model loading and preparation
            data_manager: Manager for dataset and dataloader handling
            accelerator_manager: Manager for distributed training acceleration
            optimizer_scheduler_manager: Manager for optimizers and schedulers
            metrics_logger: Logger for training and evaluation metrics
            vllm_orchestrator: Orchestrator for VLLM inference
            state_tracker: Tracker for training state
            dataset_type: Type of dataset to train on
        """
        super().__init__(
            args,
            model_manager,
            data_manager,
            accelerator_manager,
            optimizer_scheduler_manager,
            metrics_logger,
            vllm_orchestrator,
            state_tracker,
        )
        self.tokenizer = self.model_manager.get_tokenizer()
        self.train_dataloader = self.data_manager.dataloaders["provers"][
            "honest_prover"
        ]["train_dataloader"]
        self.eval_dataloader = self.data_manager.dataloaders["provers"][
            "honest_prover"
        ][
            "eval_dataloader"
        ]  # No need to duplicate dataloaders for sneaky prover due to sequential training
        self.dataset_type = dataset_type
        self.rl_config = self.model_manager.rl_config
        self.total_steps = 0

        # Ensure tokenizer has pad_token set
        if self.tokenizer.pad_token is None:
            logger.info("Setting tokenizer pad_token to eos_token")
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.formatter = Formatter(tokenizer=self.tokenizer)
        self.grpo = GRPO(self.rl_config)

        self.is_main = self.accelerator_manager.get_state_property(
            property_name="is_main_process"
        )

        self._buffered_inputs = [None] * self.args.gradient_accumulation_steps
        self._has_logged_prover_samples_this_eval = False

    def _prepare_batch(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare a batch for model input by running the vLLM pipeline.

        Args:
            batch: dictionary containing batch data

        Returns:
            dictionary containing:
                # TODO: Fill what we return here
        """
        # Prepares inputs for model training/evaluation by managing completion generation and batch handling.
        # During training:
        #   - Receives the local generation batch (Per-GPU batch size × steps per generation)
        #     from the modified training dataloader instead of the standard local batch
        #   - Generates completions once for the entire generation batch and splits it into batches of size
        #     `per_device_train_batch_size`
        #   - Buffers these completions and returns the appropriate slice for the current accumulation step
        #   - Optimizes by regenerating completions only periodically (every steps_per_generation * num_iterations)
        # During evaluation:
        #   - The input is treated as a standard local batch (no accumulation, no multiple iterations)
        #   - Completions are generated for each batch without buffering or reuse
        # Returns a single local batch in both cases.

        is_training = (
            self.model_manager.get_model("honest_prover").training
            and self.model_manager.get_model("sneaky_prover").training
        )  # Make sure both models always have the same mode

        if is_training:
            buffer_index = self.total_steps % self.args.gradient_accumulation_steps
            buffered_inputs = self._buffered_inputs[buffer_index]
            if (
                self.total_steps % self.rl_config.num_iterations == 0
                or buffered_inputs is None
            ):
                # buffered_inputs=None can occur when resuming from a checkpoint
                inputs = self._generate_and_score_completions(batch)
                self._buffered_inputs[buffer_index] = inputs
            else:
                inputs = buffered_inputs
                self.total_steps += 1
        else:
            # In evaluation, we don't reuse completions across multiple updates, so we don't need to buffer inputs.
            inputs = self._generate_and_score_completions(batch)
        return inputs

    def _extract_scores(self, outputs: torch.Tensor) -> torch.Tensor:
        """
        Extract scalar scores from model outputs.

        Args:
            outputs: Model output tensor or tuple

        Returns:
            Tensor of scalar scores
        """
        scores = outputs[0] if isinstance(outputs, tuple) else outputs
        if scores.dim() > 1 and scores.shape[1] == 1:
            scores = scores.squeeze(-1)
        return scores

    def _get_code_snippets(
        self, batch: dict[str, torch.Tensor], debug: bool = False
    ) -> tuple[list[str] | None, list[str] | None]:
        """
        Extract code snippets from batch for debugging purposes.
        Only executes if debug=True to save computational resources.

        Args:
            batch: dictionary containing batch data
            debug: Whether to extract code snippets

        Returns:
            tuple of honest and injected code snippets, or (None, None) if debug=False
        """
        if not debug:
            return None, None

        # Decode input IDs to text
        honest_prompts = self.tokenizer.batch_decode(
            batch["honest_input_ids"], skip_special_tokens=True
        )
        injected_prompts = self.tokenizer.batch_decode(
            batch["injected_input_ids"], skip_special_tokens=True
        )

        # Extract code snippets between <solution> tags
        honest_code_snippets = [
            prompt.split("<solution>")[1].split("</solution>")[0]
            for prompt in honest_prompts
        ]
        injected_code_snippets = [
            prompt.split("<solution>")[1].split("</solution>")[0]
            for prompt in injected_prompts
        ]

        return honest_code_snippets, injected_code_snippets

    def _generate_and_score_completions(
        self, batch: dict[str, torch.Tensor | Any]
    ) -> dict[str, dict[str, Any | list[Any] | None]]:
        """Generate completions and score them."""

        # devices = {
        #     "honest_prover": self.accelerator_manager.get_state_property(
        #         property_name="device", key="honest_prover"
        #     ),
        #     "sneaky_prover": self.accelerator_manager.get_state_property(
        #         property_name="device", key="sneaky_prover"
        #     ),
        # }
        raw_prompts_local = [
            x["question"] for x in batch
        ]  # List of strings, no devicing # TODO: This is not dataset specific!
        logger.info(f"Raw prompts local: {raw_prompts_local}")

        assert (
            len(raw_prompts_local) == self.args.per_device_train_batch_size
        ), f"Raw prompts local length mismatch: {len(raw_prompts_local)} != {self.args.per_device_train_batch_size}"

        num_local_raw_prompts = len(
            raw_prompts_local
        )  # These are not unique! They are repeated num_generations times [::self.args.num_generations]

        # --- HONEST PROVER ---
        hp_model_key = "honest_prover"
        hp_device = self.accelerator_manager.get_state_property(
            "device", key=hp_model_key
        )

        hp_prompt_texts_local = [
            self.formatter.make_formatted_prompt(
                model_key=hp_model_key,
                dataset_type=self.dataset_type,
                template_args={"problem": rp},
            )
            for rp in raw_prompts_local
        ]

        tokenized_hp_prompts = self.tokenizer(
            hp_prompt_texts_local,
            return_tensors="pt",
            padding="longest",
            padding_side="left",
            add_special_tokens=False,
        ).to(hp_device)
        hp_prompt_ids_local = tokenized_hp_prompts.input_ids
        hp_prompt_mask_local = tokenized_hp_prompts.attention_mask

        all_hp_prompt_texts_gathered_for_vllm = None
        if self.accelerator_manager.get_state_property("is_main_process"):
            logger.info("Gathering honest prover prompts from all processes")
            gathered_nested = gather_object(
                hp_prompt_texts_local
            )  # gather_object is from accelerate
            all_hp_prompt_texts_gathered_for_vllm = [
                item for sublist in gathered_nested for item in sublist
            ]

        honest_gen_args = self.args

        hp_completion_ids_local_lol, hp_completion_texts_local, _ = (
            self.vllm_orchestrator._generate_and_broadcast(
                client_key=hp_model_key,
                prompts=all_hp_prompt_texts_gathered_for_vllm,
                generation_args=honest_gen_args,
                n_generations=self.rl_config.num_generations,
                logprobs_count=0,
                raw_prompts_len_local=num_local_raw_prompts,
            )
        )
        honest_solutions_local = [
            self.formatter.extract_solution(text, hp_model_key, self.dataset_type)
            for text in hp_completion_texts_local
        ]  # Stores (success, solution) tuples

        # --- SNEAKY PROVER ---
        sp_model_key = "sneaky_prover"
        sp_device = self.accelerator_manager.get_state_property(
            "device", key=sp_model_key
        )

        # `honest_solutions_local` will have `num_local_raw_prompts * num_generations` items.
        # We need to pair each raw_prompt with its corresponding `num_generations` honest solutions.
        sp_prompt_texts_local = []
        if len(honest_solutions_local) != num_local_raw_prompts:
            raise ValueError(
                "Mismatch in honest solutions for sneaky prover prompt prep."
            )

        # Make the sneaky prover prompt texts
        sp_prompt_texts_local = [
            self.formatter.make_formatted_prompt(
                model_key=sp_model_key,
                dataset_type=self.dataset_type,
                template_args={
                    "problem": raw_prompts_local[i],
                    "honest_solution": honest_solutions_local[i],
                },
            )
            for i in range(num_local_raw_prompts)
        ]

        tokenized_sp_prompts = self.tokenizer(
            sp_prompt_texts_local,
            return_tensors="pt",
            padding="longest",
            padding_side="left",
            add_special_tokens=False,
        ).to(sp_device)
        sp_prompt_ids_local = tokenized_sp_prompts.input_ids
        sp_prompt_mask_local = tokenized_sp_prompts.attention_mask

        all_sp_prompt_texts_gathered_for_vllm = None
        if self.accelerator_manager.get_state_property("is_main_process"):
            gathered_nested = gather_object(sp_prompt_texts_local)
            all_sp_prompt_texts_gathered_for_vllm = [
                item for sublist in gathered_nested for item in sublist
            ]

        sneaky_gen_args = self.args.vllm_sneaky_prover

        sp_completion_ids_local_lol, sp_completion_texts_local, _ = (
            self.vllm_orchestrator._generate_and_broadcast(
                client_key=sp_model_key,
                prompts=all_sp_prompt_texts_gathered_for_vllm,
                generation_args=sneaky_gen_args,
                n_generations=self.rl_config.num_generations,
                logprobs_count=0,
                raw_prompts_len_local=num_local_raw_prompts,
            )
        )
        sneaky_solutions_local = [
            self.formatter.extract_solution(
                completion_text=text,
                model_key=sp_model_key,
                dataset_type=self.dataset_type,
            )
            for text in sp_completion_texts_local
        ]  # Stores (success, solution) tuples

        # Ensure sneaky_solutions_local has the same length as honest_solutions_local
        if len(sneaky_solutions_local) != len(honest_solutions_local):
            raise ValueError(
                "Mismatch between number of sneaky and honest solutions generated."
            )

        ## For every prover we now have:
        # - completion_ids
        # - completion_texts
        # - solutions

        # --- VERIFIER REWARD GENERATION ---
        verifier_model_key = "verifier"
        verifier_mode = (
            self.args.training_verifier.verifier_mode
        )  # Literal["regressor", "classifier"]
        # If regressor, call vllm_orchestrator.classify_and_broadcast --> yields a number
        # If classifier, call vllm_orchestrator.generate_and_broadcast --> yields a completion string

        verifier_prompt_texts_local_pairs = []
        # honest_solutions_local and sneaky_solutions_local both have num_local_raw_prompts
        for i in range(
            len(honest_solutions_local)
        ):  # Iterate through each generated solution pair
            # Determine the original raw_prompt index
            raw_prompt_item = raw_prompts_local[i]

            # Pair 1: problem + honest_solution
            verifier_prompt_texts_local_pairs.append(
                self.formatter.make_formatted_prompt(
                    model_key=verifier_model_key,
                    dataset_type=self.dataset_type,
                    template_args={
                        "problem": raw_prompt_item,
                        "solution": honest_solutions_local[i],
                    },
                )
            )
            # Pair 2: problem + sneaky_solution
            verifier_prompt_texts_local_pairs.append(
                self.formatter.make_formatted_prompt(
                    model_key=verifier_model_key,
                    dataset_type=self.dataset_type,
                    template_args={
                        "problem": raw_prompt_item,
                        "solution": sneaky_solutions_local[i],
                    },
                )
            )
        # Total verifier prompts locally = 2 * num_local_raw_prompts

        all_verifier_prompt_texts_gathered_for_vllm = None
        if self.accelerator_manager.get_state_property("is_main_process"):
            gathered_nested = gather_object(verifier_prompt_texts_local_pairs)
            all_verifier_prompt_texts_gathered_for_vllm = [
                item for sublist in gathered_nested for item in sublist
            ]

        verifier_gen_args = self.args.vllm_verifier

        if verifier_mode != "regressor":
            # _generate_and_broadcast for verifier returns *all* results to *all* processes
            _, verifier_completion_texts_all, _ = (
                self.vllm_orchestrator._generate_and_broadcast(
                    client_key=verifier_model_key,
                    prompts=all_verifier_prompt_texts_gathered_for_vllm,
                    generation_args=verifier_gen_args,
                    n_generations=1,
                    logprobs_count=0,
                    raw_prompts_len_local=num_local_raw_prompts
                    * 2,  # Due to 2 pairs of prompts
                )
            )
        else:
            # Use classify_and_broadcast
            verifier_scores_all = self.vllm_orchestrator.classify_and_broadcast(
                client_key=verifier_model_key,
                prompts=all_verifier_prompt_texts_gathered_for_vllm,
                raw_prompts_len_local=num_local_raw_prompts
                * 2,  # Due to 2 pairs of prompts
            )

        rewards_all_flat_list = None

        if self.accelerator_manager.get_state_property("is_main_process"):
            rewards_all_flat_list = (
                [
                    self.formatter.extract_verifier_reward(verifier_output_text=text)
                    for text in verifier_completion_texts_all
                ]
                if verifier_mode != "regressor"
                else verifier_scores_all
            )  # Score is either binary or float (gen/reg)
        else:
            # Placeholder on non-main processes before broadcast
            expected_len = (
                num_local_raw_prompts
                * 2
                * self.accelerator_manager.get_state_property("num_processes")
            )
            logger.info(
                f"[Process {self.accelerator_manager.get_state_property('process_index')}] Non-main process creating placeholder for {expected_len} rewards, waiting for broadcast."
            )
            rewards_all_flat_list = [None] * expected_len
        # Placeholder for broadcast_object_list, assuming it handles None on non-main
        rewards_all_flat_list = broadcast_object_list(
            rewards_all_flat_list, from_process=0
        )  # Broadcast the globally extracted rewards

        assert len(
            rewards_all_flat_list
        ) == num_local_raw_prompts * 2 * self.accelerator_manager.get_state_property(
            "num_processes"
        ), f"Rewards all flat list length mismatch: {len(rewards_all_flat_list)} != {num_local_raw_prompts * 2 * self.accelerator_manager.get_state_property('num_processes')}"

        assert len(rewards_all_flat_list) == len(
            all_verifier_prompt_texts_gathered_for_vllm
        ), (
            f"[Process {self.accelerator_manager.get_state_property('process_index')}] Mismatch after broadcast: "
            f"len(rewards_all)={len(rewards_all_flat_list)} vs len(all_verifier_prompts)={len(all_verifier_prompt_texts_gathered_for_vllm)}"
        )
        logger.info(
            f"[Process {self.accelerator_manager.get_state_property('process_index')}] Assertion passed: Length of rewards_all ({len(rewards_all_flat_list)}) matches all_verifier_prompts."
        )

        rewards_all_tensor = torch.tensor(
            rewards_all_flat_list,
            dtype=torch.float32,
            device=self.accelerator_manager.get_state_property(
                "device", key="honest_prover"
            ),
        )

        if torch.isnan(rewards_all_tensor).any():
            nan_mask = torch.isnan(rewards_all_tensor)
            rewards_all_tensor[nan_mask] = (
                self.rl_config.nan_reward_value
            )  # Set to -0.1 iirc
            if self.accelerator_manager.get_state_property("is_main_process"):
                logger.warning(
                    f"Replaced {nan_mask.sum().item()} NaN values in rewards tensor with {self.rl_config.nan_reward_value}."
                )

        # Rewards are for [H_sol1, S_sol1, H_sol2, S_sol2, ...] globally
        global_rewards_a = rewards_all_tensor[::2]  # Rewards for honest solutions
        global_rewards_b = rewards_all_tensor[1::2]  # Rewards for sneaky solutions

        logger.info("Completed reward extraction. Calculating advantages...")

        # Perform Global GRPO advantage calculation for Prover A
        advantages_a_global = self.grpo.calculate_advantages(
            global_rewards=global_rewards_a,
        )

        # Perform Global GRPO advantage calculation for Prover B (Similarly)
        advantages_b_global = self.grpo.calculate_advantages(
            global_rewards=global_rewards_b,
        )

        logger.info(
            "Completed advantage calculation. Slicing advantages and rewards..."
        )
        logger.info(f"Advantages a: {advantages_a_global}")
        logger.info(f"Advantages b: {advantages_b_global}")

        # --- Local Slicing - Advantages and Rewards ---
        # Calculate the *correct* local slice indices
        # The slice should correspond to the number of *original* prompts processed
        # by this rank for *one* prover. len(raw_prompts) holds this.
        start_index_prover = (
            self.accelerator_manager.get_state_property("process_index")
            * num_local_raw_prompts
        )
        end_index_prover = start_index_prover + num_local_raw_prompts
        local_slice_prover = slice(start_index_prover, end_index_prover)

        # Slice the *global* advantages and rewards to get the local part for provers
        local_advantages_a = advantages_a_global[local_slice_prover]
        local_advantages_b = advantages_b_global[local_slice_prover]
        local_rewards_a = global_rewards_a[local_slice_prover]
        local_rewards_b = global_rewards_b[local_slice_prover]

        logger.info("Completed slicing. Preparing model inputs and logging metrics...")

        # --- Logging Prover Samples (during evaluation for the first batch) ---
        is_eval_context = not self.model_manager.get_model("honest_prover").training
        if (
            is_eval_context
            and self.is_main
            and not self._has_logged_prover_samples_this_eval
        ):

            num_samples_to_log = 3

            problem_ids_for_log = [
                item.get("problem_id", f"p_{i}") for i, item in enumerate(batch)
            ]

            honest_solutions_text = [
                sol[1] if sol[0] else "Extraction Failed"
                for sol in honest_solutions_local
            ]
            sneaky_solutions_text = [
                sol[1] if sol[0] else "Extraction Failed"
                for sol in sneaky_solutions_local
            ]

            print_prompt_completions_sample_provers(
                problem_ids=problem_ids_for_log,
                honest_solutions=honest_solutions_text,
                sneaky_solutions=sneaky_solutions_text,
                honest_scores=local_rewards_a,
                sneaky_scores=local_rewards_b,
                correctness_scores=torch.ones_like(
                    local_rewards_a
                ),  # Placeholder for correctness scores
                step=self.state_tracker.step,
                num_samples=num_samples_to_log,
            )
            self._has_logged_prover_samples_this_eval = True  # Set flag
        # --- End Logging Block ---

        # --- POST-PROCESSING FOR TRAINING (Data for Policy Models) ---
        # Honest Prover Data
        hp_padded_completion_ids = self.formatter.tensorize_and_pad_completions(
            hp_completion_ids_local_lol, hp_device
        )
        hp_prompt_completion_ids = torch.cat(
            [hp_prompt_ids_local, hp_padded_completion_ids], dim=1
        )
        hp_completion_mask, hp_is_eos, hp_logits_to_keep = (
            self.formatter.create_completion_masks(hp_padded_completion_ids)
        )
        hp_prompt_completion_mask = torch.cat(
            [hp_prompt_mask_local, hp_completion_mask], dim=1
        )

        with (
            torch.no_grad()
        ):  # Log prob calculation should not affect gradients of policy/ref models here
            hp_old_logps, hp_ref_logps = self._calculate_log_probabilities(
                model_key=hp_model_key,
                input_ids=hp_prompt_completion_ids,
                attention_mask=hp_prompt_completion_mask,
                logits_to_keep=hp_logits_to_keep,
            )

        self.metrics_logger.store_metrics(
            mode="train",
            model_key=hp_model_key,
            metrics={
                "completion_mask": hp_completion_mask,
                "is_eos": hp_is_eos,
                "rewards": local_rewards_a,
                "advantages": local_advantages_a,
                "old_per_token_logps": hp_old_logps,
                "ref_per_token_logps": hp_ref_logps,
                "logits_to_keep": hp_logits_to_keep,
                "prompt_completion_ids": hp_prompt_completion_ids,
                "prompt_completion_mask": hp_prompt_completion_mask,
            },
        )

        # Sneaky Prover Data
        sp_padded_completion_ids = self.formatter.tensorize_and_pad_completions(
            sp_completion_ids_local_lol, sp_device
        )
        # sp_prompt_ids_local is (num_local_raw_prompts * num_generations, seq_len)
        # sp_padded_completion_ids is (num_local_raw_prompts * num_generations, compl_len)
        if sp_prompt_ids_local.shape[0] != sp_padded_completion_ids.shape[0]:
            raise ValueError(
                f"Shape mismatch for SP: prompts {sp_prompt_ids_local.shape}, completions {sp_padded_completion_ids.shape}"
            )

        sp_prompt_completion_ids = torch.cat(
            [sp_prompt_ids_local, sp_padded_completion_ids], dim=1
        )
        sp_completion_mask, sp_is_eos, sp_logits_to_keep = (
            self.formatter.create_completion_masks(sp_padded_completion_ids)
        )
        sp_prompt_completion_mask = torch.cat(
            [sp_prompt_mask_local, sp_completion_mask], dim=1
        )

        with torch.no_grad():
            sp_old_logps, sp_ref_logps = self._calculate_log_probabilities(
                model_key=sp_model_key,
                input_ids=sp_prompt_completion_ids,
                attention_mask=sp_prompt_completion_mask,
                logits_to_keep=sp_logits_to_keep,
            )

        self.metrics_logger.store_metrics(
            mode="train",
            model_key=sp_model_key,
            metrics={
                "completion_mask": sp_completion_mask,
                "is_eos": sp_is_eos,
                "rewards": local_rewards_b,
                "advantages": local_advantages_b,
                "old_per_token_logps": sp_old_logps,
                "ref_per_token_logps": sp_ref_logps,
                "logits_to_keep": sp_logits_to_keep,
                "prompt_completion_ids": sp_prompt_completion_ids,
                "prompt_completion_mask": sp_prompt_completion_mask,
            },
        )

        return {
            "honest_prover": {
                "prompt_ids": hp_prompt_ids_local,
                "prompt_mask": hp_prompt_mask_local,
                "completion_ids": hp_padded_completion_ids,  # Padded completions only
                "completion_mask": hp_completion_mask,  # Mask for padded_completions
                "advantages": local_advantages_a,
                "old_per_token_logps": hp_old_logps,  # Logps for completion tokens
                "ref_per_token_logps": hp_ref_logps,  # Ref logps for completion tokens
                "logits_to_keep": hp_logits_to_keep,  # Length of completions for slicing logits
                "prompt_completion_ids": hp_prompt_completion_ids,  # Full sequence for training
                "prompt_completion_mask": hp_prompt_completion_mask,  # Mask for full sequence
            },
            "sneaky_prover": {
                "prompt_ids": sp_prompt_ids_local,
                "prompt_mask": sp_prompt_mask_local,
                "completion_ids": sp_padded_completion_ids,
                "completion_mask": sp_completion_mask,
                "advantages": local_advantages_b,
                "old_per_token_logps": sp_old_logps,
                "ref_per_token_logps": sp_ref_logps,
                "logits_to_keep": sp_logits_to_keep,
                "prompt_completion_ids": sp_prompt_completion_ids,
                "prompt_completion_mask": sp_prompt_completion_mask,
            },
        }

    def _calculate_log_probabilities(
        self,
        model_key: Literal["honest_prover", "sneaky_prover"],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        logits_to_keep: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """
        Calculates old (policy) and reference log probabilities for a given model key.
        Ensures reference model is in eval mode.

        Args:
            model_key: The key identifying the model ("honest_prover" or "sneaky_prover").
            container_data: The dictionary slice from the container for the given model key,
                            containing keys like "prompt_completion_ids", "prompt_completion_mask", "logits_to_keep".

        Returns:
            A tuple containing (old_per_token_logps, ref_per_token_logps). Each can be None.
        """
        old_per_token_logps = None
        ref_per_token_logps = None
        policy_model = (
            self.honest_prover_model
            if model_key == "honest_prover"
            else self.sneaky_prover_model
        )
        ref_model = self.model_manager.get_ref_model(model_key)

        # Calculate old log probabilities if needed (num_iterations > 1)
        if self.rl_config.num_iterations > 1:
            # Assuming policy_model is already in the correct mode (train/eval)
            old_per_token_logps = self._get_per_token_logps(
                policy_model, input_ids, attention_mask, logits_to_keep
            )

        # Calculate reference log probabilities if needed (beta > 0)
        if self.rl_config.beta > 0.0:
            if ref_model is None:
                raise ValueError(
                    f"Reference model for {model_key} required but not loaded (beta > 0)."
                )
            # Ensure reference model is in eval mode
            ref_model.eval()
            ref_per_token_logps = self._get_per_token_logps(
                ref_model, input_ids, attention_mask, logits_to_keep
            )

        return old_per_token_logps, ref_per_token_logps

    def _get_per_token_logps(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        logits_to_keep: int,
    ) -> torch.Tensor:
        # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            logits_to_keep=logits_to_keep + 1,
        ).logits
        logits = logits[
            :, :-1, :
        ]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred

        input_ids = input_ids[:, -logits_to_keep:]
        # For transformers<=4.48, logits_to_keep argument isn't supported, so here we drop logits ourselves.
        # See https://github.com/huggingface/trl/issues/2770
        logits = logits[:, -logits_to_keep:]
        # Divide logits by sampling temperature.
        # See https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo#policy-training-implementation-details
        logits = logits / self.args.training_honest_prover.temperature
        return selective_log_softmax(
            logits, input_ids
        )  # compute logprobs for the input tokens

    def _full_lm_head_params(self, unwrapped_model, zero3):
        # returns weight,bias tensors that are guaranteed to be full on every rank
        params = [unwrapped_model.lm_head.weight]
        if unwrapped_model.lm_head.bias is not None:
            params.append(unwrapped_model.lm_head.bias)

        if zero3:
            # modifier_rank=None => gather on *all* ranks (not only rank‑0)
            ctx = deepspeed.zero.GatheredParameters(params, modifier_rank=None)
        else:
            ctx = nullcontext()
        return ctx  # to be used as a "with" context‑manager

    def _compute_loss(
        self,
        completion_mask: torch.Tensor,
        old_per_token_logps: torch.Tensor,
        per_token_logps: torch.Tensor,
        advantages: torch.Tensor,
        per_token_kl: torch.Tensor | None,
        mode: Literal["train", "eval"],
        model_key: Literal["honest_prover", "sneaky_prover"],
    ):
        # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's computation (see
        # _generate_and_score_completions) and use per_token_logps.detach() instead.
        old_per_token_logps = (
            old_per_token_logps
            if self.rl_config.num_iterations > 1
            else per_token_logps.detach()
        )
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(
            coef_1, 1 - self.rl_config.epsilon_low, 1 + self.rl_config.epsilon_high
        )
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if per_token_kl is not None:  # Equivalent to if self.rl_config.beta != 0.0
            per_token_loss = per_token_loss + self.rl_config.beta * per_token_kl

        loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(
            min=1.0
        )

        if self.rl_config.beta != 0.0:
            mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
            self.metrics_logger.store_metric(
                mode=mode,
                model_key=model_key,
                metric_name="kl",
                value=mean_kl,
            )

        # Compute the clip ratio
        is_clipped = (
            (coef_1 < 1 - self.rl_config.epsilon_low) & (advantages.unsqueeze(1) < 0)
        ) | ((coef_1 > 1 + self.rl_config.epsilon_high) & (advantages.unsqueeze(1) > 0))
        clip_ratio = (is_clipped * completion_mask).sum() / completion_mask.sum()
        self.metrics_logger.store_metric(
            mode=mode,
            model_key=model_key,
            metric_name="clip_ratio",
            value=clip_ratio,
        )
        return loss

    def _get_last_hidden_state(
        self,
        model: torch.nn.Module,
        model_key: Literal["honest_prover", "sneaky_prover"],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        logits_to_keep: int | None = None,
    ) -> torch.Tensor:
        unwrapped_model = self.accelerator_manager.unwrap_model(model, key=model_key)
        # unwrap the model to access the model.model
        last_hidden_state = unwrapped_model.model(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        last_hidden_state = last_hidden_state[:, :-1, :]  # (B, L-1, H)
        if logits_to_keep is not None:
            last_hidden_state = last_hidden_state[
                :, -logits_to_keep:, :
            ]  # (B, logits_to_keep, H)
        return last_hidden_state

    def compute_liger_loss(
        self,
        model,
        model_key: Literal["honest_prover", "sneaky_prover"],
        mode: Literal["train", "eval"],
        **kwargs,
    ):
        zero3 = (
            self.accelerator_manager.get_state_property(
                "deepspeed_plugin", key=model_key
            )
            is not None
            and self.accelerator_manager.get_state_property(
                "deepspeed_plugin", key=model_key
            ).zero_stage
            == 3
        )

        unwrapped_model = self.accelerator_manager.unwrap_model(model, key=model_key)

        with self._full_lm_head_params(unwrapped_model, zero3):
            weight = unwrapped_model.lm_head.weight  # now full on every rank
            bias = unwrapped_model.lm_head.bias

            loss, metrics = self.model_manager.liger_grpo_loss(
                lin_weight=weight,
                bias=bias,
                _input=kwargs["last_hidden_state"],
                selected_token_ids=kwargs["completion_ids"],
                attention_mask=kwargs["completion_mask"],
                advantages=kwargs["advantages"],
                ref_per_token_logps=kwargs["ref_per_token_logps"],
                old_per_token_logps=kwargs["old_per_token_logps"],
            )  # type: ignore

        # Extract metrics from the liger_grpo_loss output
        # KL divergence is the first metric when beta is non-zero
        mean_kl = metrics[0] if self.rl_config.beta != 0.0 else None
        clip_ratio = metrics[-1]

        if self.rl_config.beta != 0.0:
            self.metrics_logger.store_metric(
                mode=mode,
                model_key=model_key,
                metric_name="kl",
                value=mean_kl,
            )
        self.metrics_logger.store_metric(
            mode=mode,
            model_key=model_key,
            metric_name="clip_ratio",
            value=clip_ratio,
        )
        return loss

    def train(
        self, num_steps_or_epochs: int = 1
    ):  # Renamed from num_steps_or_epochs for clarity
        """
        Main training loop for Prover models using GRPO.
        """
        # self.total_steps needs to be initialized if not resuming, or loaded from checkpoint.
        # Let's assume StateTracker or orchestrator handles epoch/global step tracking.
        # For this example, we'll use a local `current_micro_batch_step`.

        current_micro_batch_step = 0

        for epoch in range(num_steps_or_epochs):
            logger.info(
                f"Starting Prover Training Epoch {epoch + 1}/{num_steps_or_epochs}"
            )

            if self.is_main:
                progress_bar = tqdm(
                    self.train_dataloader, desc=f"Prover Epoch {epoch + 1}"
                )
            else:
                progress_bar = self.train_dataloader

            for batch_idx, raw_batch_data in enumerate(progress_bar):
                # self.total_steps is used by _prepare_batch for buffering logic
                self.total_steps = current_micro_batch_step

                # 1. Generate/Retrieve completions and scores
                # `raw_batch_data` is a list of dicts from AppsDataset, e.g. [{'question': '...'}, ...]
                processed_batch_for_training = self._prepare_batch(raw_batch_data)

                # 2. Perform training update for each prover
                for prover_key in ["honest_prover", "sneaky_prover"]:  # type: ignore
                    prover_specific_data = processed_batch_for_training[prover_key]
                    policy_model = self.model_manager.get_model(
                        prover_key, prepared=True
                    )
                    optimizer = self.optimizer_scheduler_manager.get_optimizer(
                        prover_key
                    )

                    device = self.accelerator_manager.get_state_property(
                        "device", key=prover_key
                    )
                    mode = "train" if policy_model.training else "eval"

                    prompt_ids, prompt_mask = (
                        prover_specific_data["prompt_ids"].to(device),
                        prover_specific_data["prompt_mask"].to(device),
                    )
                    completion_ids, completion_mask = (
                        prover_specific_data["completion_ids"].to(device),
                        prover_specific_data["completion_mask"].to(device),
                    )
                    input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
                    attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
                    logits_to_keep = completion_ids.size(1)

                    # Get full logits to compute entropy
                    unwrapped_model = self.accelerator_manager.unwrap_model(
                        policy_model, key=prover_key
                    )
                    outputs = unwrapped_model(
                        input_ids=input_ids, attention_mask=attention_mask
                    )
                    logits = outputs.logits[
                        :, -logits_to_keep - 1 : -1, :
                    ]  # Get logits for completion tokens

                    # Calculate entropy over the logits
                    per_token_entropy = compute_entropy(
                        logits, completion_mask, reduce=False
                    )
                    # Store entropy in metrics
                    self.metrics_logger.store_entropy(
                        prover_key, "train", per_token_entropy
                    )

                    per_token_logps = self._get_per_token_logps(
                        policy_model, input_ids, attention_mask, logits_to_keep
                    )

                    # Compute the KL divergence between the model and the reference model
                    if self.rl_config.beta != 0.0:
                        ref_per_token_logps = prover_specific_data[
                            "ref_per_token_logps"
                        ]
                        per_token_kl = (
                            torch.exp(ref_per_token_logps - per_token_logps)
                            - (ref_per_token_logps - per_token_logps)
                            - 1
                        )

                    if self.args.training_honest_prover.apply_liger_kernel:
                        last_hidden_state = self._get_last_hidden_state(
                            model=policy_model,
                            model_key=prover_key,
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            logits_to_keep=logits_to_keep,
                        )

                        loss = self.compute_liger_loss(
                            model=policy_model,
                            completion_ids=completion_ids,
                            completion_mask=completion_mask,
                            last_hidden_state=last_hidden_state,
                            advantages=prover_specific_data["advantages"],
                            ref_per_token_logps=(
                                ref_per_token_logps.to(device)
                                if ref_per_token_logps is not None
                                else None
                            ),
                            old_per_token_logps=prover_specific_data[
                                "old_per_token_logps"
                            ].to(device),
                            model_key=prover_key,
                            mode=mode,
                        )
                    else:
                        loss = self._compute_loss(
                            completion_mask=completion_mask,
                            old_per_token_logps=prover_specific_data[
                                "old_per_token_logps"
                            ].to(device),
                            per_token_logps=per_token_logps,
                            advantages=prover_specific_data["advantages"],
                            per_token_kl=(
                                per_token_kl.to(device)
                                if per_token_kl is not None
                                else None
                            ),
                            mode=mode,
                            model_key=prover_key,
                        )

                    # Normalize loss by gradient accumulation steps
                    loss = loss / self.args.gradient_accumulation_steps

                    self.accelerator_manager.backward(loss, key=prover_key)

                    self.metrics_logger.store_metric(
                        mode="train",
                        model_key=prover_key,
                        metric_name="loss",
                        value=loss.item() * self.args.gradient_accumulation_steps,
                    )

                # Gradient accumulation optimizer step
                is_last_step_in_batch = (current_micro_batch_step + 1) == len(
                    self.train_dataloader
                )
                is_accumulation_boundary = (
                    current_micro_batch_step + 1
                ) % self.args.gradient_accumulation_steps == 0
                is_sync_step = is_last_step_in_batch or is_accumulation_boundary

                if is_sync_step:
                    for prover_key in ["honest_prover", "sneaky_prover"]:
                        optimizer = self.optimizer_scheduler_manager.get_optimizer(
                            prover_key
                        )
                        scheduler = self.optimizer_scheduler_manager.get_scheduler(
                            prover_key
                        )

                        # Optional: Gradient Clipping (needs model parameters)
                        if self.args.training_honest_prover.max_grad_norm is not None:
                            self.accelerator_manager.clip_grad_norm_(
                                self.model_manager.get_model(
                                    prover_key, prepared=True
                                ).parameters(),
                                self.args.training_honest_prover.max_grad_norm,
                                key=prover_key,
                            )

                        optimizer.step()
                        if scheduler:
                            scheduler.step()
                        optimizer.zero_grad()

                    logger.debug(
                        f"Prover optimizer step at micro_batch_step {current_micro_batch_step}"
                    )

                    # Log step metrics after optimizer step
                    self.metrics_logger.log_step_metrics(
                        phase=self.state_tracker.phase, mode="train"
                    )

                    # Evaluate every 100 steps
                    if current_micro_batch_step % 100 == 0:
                        self.evaluate()

                if (
                    self.is_main
                ):  # self.is_main from ProverTrainerBase or AcceleratorManager
                    progress_bar.set_postfix(
                        hp_loss=self.metrics_logger.get_latest_metric(
                            "train", "honest_prover", "loss"
                        ),
                        sp_loss=self.metrics_logger.get_latest_metric(
                            "train", "sneaky_prover", "loss"
                        ),
                    )

                current_micro_batch_step += 1
                self.state_tracker.increment_step()  # Global step tracking

            # End of epoch
            if self.is_main:
                progress_bar.close()

            # Evaluation at end of epoch
            self.evaluate()

        logger.info("Prover training finished.")
        # Potentially sync weights to VLLM if provers are used for generation after training
        self.vllm_orchestrator.sync_weights(
            phase="provers", model_manager=self.model_manager
        )

        logger.info("Attempting to push models to hub after training...")
        model_keys_to_push = ["honest_prover", "sneaky_prover"]
        for model_key in model_keys_to_push:
            try:
                model_to_push = self.model_manager.get_model(model_key, prepared=True)
                if model_to_push is not None:
                    self._push_model_to_hub(
                        model_to_push=model_to_push, model_key=model_key
                    )
                else:
                    logger.warning(
                        f"Model with key '{model_key}' not found in ModelManager. Skipping push."
                    )
            except Exception as e:
                logger.error(
                    f"An error occurred while preparing to push {model_key} to hub: {str(e)}"
                )

        logger.info("Prover training finished.")

    def evaluate(self) -> dict[str, float] | None:
        """
        Evaluate the provers.

        Returns:
            dictionary of evaluation metrics if on the main process, otherwise None.
        """
        self._has_logged_prover_samples_this_eval = (
            False  # Reset flag at the start of evaluation
        )
        logger.info("Starting Prover Evaluation...")
        mode = "eval"

        # Initialize accumulators for metrics for each prover
        # These will store lists of per-batch metrics, to be aggregated later
        batch_metrics_accumulator = {
            prover_key: {
                "loss": [],
                "kl": [],
                "clip_ratio": [],
                "entropy": [],
                "count": 0,  # Number of batches processed
            }
            for prover_key in ["honest_prover", "sneaky_prover"]
        }

        if self.is_main:
            progress_bar = tqdm(
                self.eval_dataloader,
                desc="Prover Evaluating",
                total=len(self.eval_dataloader),
            )
        else:
            progress_bar = self.eval_dataloader

        try:
            with torch.no_grad():  # No gradients needed for evaluation
                for raw_batch_data in progress_bar:
                    # 1. Generate/Retrieve completions and scores
                    # _prepare_batch handles the is_training flag internally for generation behavior
                    processed_batch_for_evaluation = self._prepare_batch(raw_batch_data)

                    for prover_key_literal in ["honest_prover", "sneaky_prover"]:
                        prover_key = typing.cast(
                            Literal["honest_prover", "sneaky_prover"],
                            prover_key_literal,
                        )

                        policy_model = self.model_manager.get_model(
                            prover_key, prepared=True
                        )
                        policy_model.eval()  # Ensure model is in eval mode

                        prover_specific_data = processed_batch_for_evaluation[
                            prover_key
                        ]
                        device = self.accelerator_manager.get_state_property(
                            "device", key=prover_key
                        )

                        # Move data to device
                        prompt_ids = prover_specific_data["prompt_ids"].to(device)
                        prompt_mask = prover_specific_data["prompt_mask"].to(device)
                        completion_ids = prover_specific_data["completion_ids"].to(
                            device
                        )
                        completion_mask = prover_specific_data["completion_mask"].to(
                            device
                        )
                        advantages = prover_specific_data["advantages"].to(device)
                        old_per_token_logps = prover_specific_data[
                            "old_per_token_logps"
                        ].to(device)
                        ref_per_token_logps = (
                            prover_specific_data["ref_per_token_logps"].to(device)
                            if self.rl_config.beta > 0.0
                            else None
                        )

                        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
                        attention_mask = torch.cat(
                            [prompt_mask, completion_mask], dim=1
                        )
                        logits_to_keep = completion_ids.size(1)

                        # Get full logits to compute entropy
                        unwrapped_model = self.accelerator_manager.unwrap_model(
                            policy_model, key=prover_key
                        )
                        outputs = unwrapped_model(
                            input_ids=input_ids, attention_mask=attention_mask
                        )
                        logits = outputs.logits[
                            :,
                            -logits_to_keep - 1 : -1,
                            :,  # Get logits for completion tokens
                        ]

                        # Calculate entropy over the logits
                        per_token_entropy = compute_entropy(
                            logits, completion_mask, reduce=False
                        )
                        mean_entropy_batch = (
                            per_token_entropy * completion_mask
                        ).sum() / completion_mask.sum().clamp(min=1.0)
                        batch_metrics_accumulator[prover_key]["entropy"].append(
                            mean_entropy_batch.item()
                        )
                        # Also log it via metrics_logger immediately if desired, or wait for aggregation
                        self.metrics_logger.store_entropy(
                            prover_key, mode, per_token_entropy
                        )

                        # Calculate current per-token log probabilities
                        per_token_logps = self._get_per_token_logps(
                            policy_model, input_ids, attention_mask, logits_to_keep
                        )

                        per_token_kl_eval = None
                        if (
                            self.rl_config.beta > 0.0
                            and ref_per_token_logps is not None
                        ):
                            per_token_kl_eval = (
                                torch.exp(per_token_logps - ref_per_token_logps)
                                - (per_token_logps - ref_per_token_logps)
                                - 1
                            )

                        loss: torch.Tensor
                        if self.args.training_honest_prover.apply_liger_kernel:
                            last_hidden_state = self._get_last_hidden_state(
                                model=policy_model,
                                model_key=prover_key,
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                logits_to_keep=logits_to_keep,
                            )
                            loss = self.compute_liger_loss(
                                model=policy_model,
                                completion_ids=completion_ids,  # LIGER needs completion_ids, not full input_ids
                                completion_mask=completion_mask,
                                last_hidden_state=last_hidden_state,
                                advantages=advantages,
                                ref_per_token_logps=(
                                    ref_per_token_logps.to(device)
                                    if ref_per_token_logps is not None
                                    else None
                                ),
                                old_per_token_logps=old_per_token_logps.to(device),
                                model_key=prover_key,
                                mode=mode,  # compute_liger_loss will log clip_ratio and KL itself
                            )
                        else:
                            loss = self._compute_loss(
                                completion_mask=completion_mask,
                                old_per_token_logps=old_per_token_logps.to(
                                    device
                                ),  # For eval, old==current if num_iterations=1
                                per_token_logps=per_token_logps,
                                advantages=advantages,  # Advantages might not be super relevant for eval loss value but needed by func
                                per_token_kl=(
                                    per_token_kl_eval.to(device)
                                    if per_token_kl_eval is not None
                                    else None
                                ),
                                mode=mode,  # _compute_loss will log clip_ratio and KL itself
                                model_key=prover_key,
                            )

                        batch_metrics_accumulator[prover_key]["loss"].append(
                            loss.item()
                        )
                        # KL and clip_ratio are logged by _compute_loss and compute_liger_loss
                        # So we can retrieve them from metrics_logger later or re-calculate for aggregation here.
                        # For simplicity, let's assume they are logged and we'll fetch from MetricsLogger or re-average.
                        # If we want to aggregate KL specifically for eval summary:
                        if per_token_kl_eval is not None:
                            mean_kl_batch = (
                                per_token_kl_eval * completion_mask
                            ).sum() / completion_mask.sum().clamp(min=1.0)
                            batch_metrics_accumulator[prover_key]["kl"].append(
                                mean_kl_batch.item()
                            )

                        # Clip ratio is also logged internally by the loss functions.
                        # We can retrieve the latest from MetricsLogger for progress bar, and average at the end.

                    batch_metrics_accumulator["honest_prover"][
                        "count"
                    ] += 1  # Assuming one entry per outer batch loop for count
                    batch_metrics_accumulator["sneaky_prover"]["count"] += 1

                    if self.is_main and isinstance(progress_bar, tqdm):
                        postfix_metrics = {}
                        for pk in ["honest_prover", "sneaky_prover"]:
                            latest_loss = (
                                batch_metrics_accumulator[pk]["loss"][-1]
                                if batch_metrics_accumulator[pk]["loss"]
                                else float("nan")
                            )
                            postfix_metrics[f"{pk[:2]}_loss"] = f"{latest_loss:.4f}"
                            if batch_metrics_accumulator[pk]["kl"]:
                                latest_kl = batch_metrics_accumulator[pk]["kl"][-1]
                                postfix_metrics[f"{pk[:2]}_kl"] = f"{latest_kl:.4f}"
                        progress_bar.set_postfix(postfix_metrics)

            # Aggregate and log metrics
            final_metrics: dict[str, float] = {}
            if self.is_main:
                for prover_key_literal in ["honest_prover", "sneaky_prover"]:
                    prover_key = typing.cast(
                        Literal["honest_prover", "sneaky_prover"], prover_key_literal
                    )
                    num_batches = batch_metrics_accumulator[prover_key]["count"]
                    if num_batches > 0:
                        avg_loss = (
                            sum(batch_metrics_accumulator[prover_key]["loss"])
                            / num_batches
                        )
                        final_metrics[f"eval_{prover_key}_loss"] = avg_loss
                        self.metrics_logger.store_metric(
                            mode=mode,
                            model_key=prover_key,
                            metric_name="loss",
                            value=avg_loss,
                        )

                        if batch_metrics_accumulator[prover_key]["kl"]:
                            avg_kl = (
                                sum(batch_metrics_accumulator[prover_key]["kl"])
                                / num_batches
                            )
                            final_metrics[f"eval_{prover_key}_kl"] = avg_kl
                            # _compute_loss already logs 'kl', this would be redundant unless we want a separate overall eval_kl
                            # self.metrics_logger.store_metric(
                            #     mode=mode, model_key=prover_key, metric_name="kl_divergence", value=avg_kl
                            # )

                        avg_entropy = (
                            sum(batch_metrics_accumulator[prover_key]["entropy"])
                            / num_batches
                        )
                        final_metrics[f"eval_{prover_key}_entropy"] = avg_entropy
                        self.metrics_logger.store_metric(
                            mode=mode,
                            model_key=prover_key,
                            metric_name="entropy",
                            value=avg_entropy,
                        )

                        # Note: clip_ratio is logged by _compute_loss and compute_liger_loss.
                        # We can log an average if needed by fetching all stored values from metrics_logger for the eval steps.
                        # For now, we rely on the per-step logging from those functions.

                logger.info(f"Prover Evaluation finished. Metrics: {final_metrics}")
                self.metrics_logger.log_step_metrics(
                    phase=self.state_tracker.phase, mode=mode
                )

                if isinstance(progress_bar, tqdm):
                    progress_bar.close()
                return final_metrics
            else:
                # Non-main processes can also log step metrics if the logger supports it,
                # or simply wait for the main process.
                # Ensure all processes reach this point if there are distributed operations not shown here.
                # self.metrics_logger.log_step_metrics(phase=self.state_tracker.phase, mode=mode) # If needed
                if isinstance(progress_bar, tqdm):  # Should not happen if not is_main
                    progress_bar.close()
                return None

        except Exception as e:
            logger.error(f"Error during prover evaluation: {str(e)}")
            if self.is_main and isinstance(progress_bar, tqdm):
                progress_bar.close()
            # Attempt to clean up resources
            torch.cuda.empty_cache()
            gc.collect()
            raise

    def _push_model_to_hub(
        self, model_to_push: torch.nn.Module, model_key: str
    ) -> None:
        """
        Push the trained model to the Hugging Face model hub with proper error handling.

        Args:
            model_to_push: The model instance to push.
            model_key: The key of the model (e.g., 'honest_prover', 'sneaky_prover').
        """

        hub_model_name = f"jvelja/{model_key}_round_{self.state_tracker.round}"

        try:
            logger.info(f"Attempting to push {model_key} to {hub_model_name}")
            model_to_push.push_to_hub(repo_id=hub_model_name)
            logger.info(
                f"{model_key} model successfully pushed to the hub as {hub_model_name}."
            )
            self.tokenizer.push_to_hub(repo_id=hub_model_name)
            logger.info(f"Tokenizer also pushed to {hub_model_name} for {model_key}.")
        except Exception as e:
            logger.error(
                f"Failed to push {model_key} or its tokenizer to hub ({hub_model_name}): {str(e)}"
            )
            logger.info(f"Continuing without pushing {model_key} to hub.")
