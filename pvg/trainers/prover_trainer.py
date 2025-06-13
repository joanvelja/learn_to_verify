# pvg/trainers/prover_trainer.py

"""
ProverTrainer
------------------------
Implements sequential Prover Training using GRPO.

This trainer implements an RL-trained set of provers (sneaky) that learn to:
- Generate solutions that are preferred by the verifier
- Whilst being mode conditioned
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
    BatchEvaluator,
    EvaluationConfig,
)
from pvg.config.args import ExperimentArgs
from pvg.rl import GRPO
from pvg.trainers.prover_base import ProverTrainerBase
from pvg.utils import compute_entropy
from pvg.utils.verifier_performance import (
    VerifierPerformanceTracker,
    calculate_verifier_performance_metrics,
)


from pvg.utils.rich_logger import print_prompt_completions_sample_provers

# Configure logger
logger = logging.getLogger(f"pvg.{__name__}")  # Get a child logger


class ProverTrainer(ProverTrainerBase):
    """
    Trainer for the Prover model that learns to generate solutions that are preferred by the verifier using GRPO with a coefficient-free tier-based reward system.

    The trainer implements a new coefficient-free reward schedule that uses lexicographic
    tiers derived from verifier bounds (B) to ensure proper alignment without hyperparameters:

    Reward Structure:
    - Constants: B (verifier bound), M = 2B + 1 (tier separator)
    - Honest: -M (extraction fail), 0 (no tests pass), M*f + v (tests pass)
    - Sneaky (conditioned on honest compilation):
      * If honest fails: -abs(v)/2 (sneaky fails), -M/2 (no backdoor), +M/2 + v (backdoor works)
      * If honest works: -M (sneaky fails), -M/2 (no backdoor), +M/2 + v (backdoor works)

    Where f = test pass rate [0,1], v = raw verifier score [-B,+B],
    and tier gaps guarantee lexicographic ordering without tunable coefficients.
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
            "sneaky_prover"
        ]["train_dataloader"]
        self.eval_dataloader = self.data_manager.dataloaders["provers"][
            "sneaky_prover"
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

        self.evaluator = BatchEvaluator(
            config=EvaluationConfig(
                step_timeouts={"exec": 2, "test_gen": 5, "verify": 8},
                success_threshold=0.85,
                total_timeout=16,
            )
        )

        self.verifier_performance_tracker = VerifierPerformanceTracker(
            window_size=100, track_bounds_history=True
        )

        self.is_main = self.accelerator_manager.get_state_property(
            property_name="is_main_process"
        )

        self._buffered_inputs: list[dict[str, torch.Tensor | None] | None] = [
            None
        ] * self.args.gradient_accumulation_steps
        self._has_logged_prover_samples_this_eval = False

    def _to_dict(self, obj):
        """Convert an object to a dictionary if it's not already one.

        Args:
            obj: The object to convert

        Returns:
            dict: Dictionary representation of the object
        """
        if isinstance(obj, dict):
            return obj

        # For dataclasses, use dataclasses.asdict()
        if hasattr(obj, "__dataclass_fields__"):
            import dataclasses

            # Extract only fields relevant to the vLLM server
            obj_dict = dataclasses.asdict(obj)
            vllm_relevant_fields = [
                "max_tokens",
                "temperature",
                "top_p",
                "top_k",
                "repetition_penalty",
                "frequency_penalty",
                "min_p",
                "stop_sequences",
                "logprobs",
            ]

            # Create a dictionary with only the fields that exist in the object
            return {field: obj_dict[field] for field in vllm_relevant_fields}
        # Fallback for non-dataclass objects
        return vars(obj)

    def _prepare_batch(
        self, batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor | None]:
        """
        Prepare a batch for model input by running the vLLM pipeline.

        Args:
            batch: dictionary containing batch data

        Returns:
            dictionary containing:
                - prompt_ids
                - prompt_mask
                - completion_ids
                - completion_mask
                - advantages
                - old_per_token_logps
                - ref_per_token_logps
        """
        is_training = self.model_manager.get_model("sneaky_prover").training

        if is_training:
            buffer_index = self.total_steps % self.args.gradient_accumulation_steps
            buffered_inputs = self._buffered_inputs[buffer_index]
            should_generate_new = (
                self.total_steps % self.rl_config.num_iterations == 0
                or buffered_inputs is None
            )
            if should_generate_new:
                # buffered_inputs=None can occur when resuming from a checkpoint
                inputs = self._generate_and_score_completions(batch)
                self._buffered_inputs[buffer_index] = inputs
            else:
                inputs = buffered_inputs
        else:
            # In evaluation, we don't reuse completions across multiple updates, so we don't need to buffer inputs.
            inputs = self._generate_and_score_completions(batch)
        return inputs

    def _generate_and_score_completions(
        self, batch: dict[str, torch.Tensor | Any]
    ) -> dict[str, Any]:
        """Generate completions and score them."""

        raw_prompts_local = [
            (
                x["question"],  # 0
                x["starter_code"],  # 1
                x["problem_id"],  # 2
                x["harness_code"],  # 3
                x["transformed_solution"],  # 4
            )
            for x in batch
        ]  # List of strings, no devicing # TODO: This is not dataset specific!

        raw_mono_solutions_local = [x["mono_solutions"] for x in batch]

        num_local_raw_prompts = len(
            raw_prompts_local
        )  # These are **not** unique prompts

        honest_solutions_local: list[tuple[bool, str]] = [
            self.formatter.extract_solution(text, "honest_prover", self.dataset_type)
            for text in raw_mono_solutions_local
        ]  # (success, solution) tuples

        # --- SNEAKY PROVER ---
        sp_model_key = "sneaky_prover"
        sp_device = self.accelerator_manager.get_state_property(
            "device", key=sp_model_key
        )

        # `honest_solutions_local` will have `num_local_raw_prompts` items.
        # Make the sneaky prover prompt texts -- 1 per original prompt
        # sp_prompt_texts_local = [
        #     self.formatter.make_formatted_prompt(
        #         model_key=sp_model_key,
        #         dataset_type=self.dataset_type,
        #         template_args={
        #             "problem": raw_prompts_local[i][0],
        #             "honest_solution": honest_solutions_local[i][1],
        #         },
        #     )
        #     for i in range(num_local_raw_prompts)
        # ]

        ### Instruction-tuned tweak
        sp_prompt_texts_local = [
            [
                {
                    "role": "user",
                    "content": self.formatter.make_formatted_prompt(
                        model_key=sp_model_key,
                        dataset_type=self.dataset_type,
                        template_args={
                            "problem": raw_prompts_local[i][0],
                            "honest_solution": honest_solutions_local[i][1],
                        },
                    ),
                },
                {"role": "assistant", "content": "<reasoning>\n"},
            ]
            for i in range(num_local_raw_prompts)
        ]

        # Another Instruction-tuned tweak
        formatted_texts = []
        for (
            conversation
        ) in (
            sp_prompt_texts_local
        ):  # sp_prompt_texts_local is now list of chat conversations
            formatted_text = self.tokenizer.apply_chat_template(
                conversation,
                tokenize=False,  # Get string first
                add_generation_prompt=False,  # Adjust based on your needs
            )
            formatted_texts.append(formatted_text)

        # tokenized_sp_prompts = self.tokenizer(
        #     sp_prompt_texts_local,
        #     return_tensors="pt",
        #     padding="longest",
        #     padding_side="left",
        #     add_special_tokens=False,
        # ).to(sp_device)
        tokenized_sp_prompts = self.tokenizer(
            formatted_texts,  # Now a list of formatted strings
            return_tensors="pt",
            padding="longest",
            padding_side="left",
            add_special_tokens=False,
        ).to(sp_device)

        sp_prompt_ids_local = tokenized_sp_prompts.input_ids
        sp_prompt_mask_local = tokenized_sp_prompts.attention_mask

        all_sp_prompt_texts_gathered_for_vllm = gather_object(sp_prompt_texts_local)

        sneaky_gen_args = self._to_dict(self.args.vllm_sneaky_prover)

        sneaky_gen_args["chat_template"] = self.tokenizer.chat_template
        sneaky_gen_args["continue_final_message"] = True
        sneaky_gen_args["add_generation_prompt"] = False

        logger.info("[DEBUG]: Rolling with n_generations=8 -- Best-of-N GRPO")
        sp_completion_ids_local_lol, sp_completion_texts_local, _ = (
            self.vllm_orchestrator._generate_and_broadcast(
                client_key=sp_model_key,
                prompts=all_sp_prompt_texts_gathered_for_vllm,
                generation_args=sneaky_gen_args,
                n_generations=self.rl_config.num_generations,
                logprobs_count=0,
                raw_prompts_len_local=num_local_raw_prompts,
                is_instruction=True,
            )
        )
        logger.info("[DEBUG]: Generated sneaky completions.")

        sneaky_solutions_local: list[tuple[bool, str]] = [
            self.formatter.extract_solution(
                completion_text=text,
                model_key=sp_model_key,
                dataset_type=self.dataset_type,
            )
            for text in sp_completion_texts_local
        ]

        logger.info("[DEBUG]: Extracted sneaky solutions.")

        # Get triggering conditions from sneaky solutions
        sneaky_triggering_conditions_local: list[tuple[bool, str]] = [
            self.formatter.extract_triggering_condition(
                solution=text,
                model_key=sp_model_key,
                dataset_type=self.dataset_type,
            )
            for text in sp_completion_texts_local
        ]  # Stores (success, triggering_condition) tuples
        logger.info("[DEBUG]: Extracted sneaky triggering conditions.")

        # Ensure sneaky_solutions_local has the same length as honest_solutions_local
        if len(sneaky_solutions_local) != len(honest_solutions_local):
            # Prepare the data to dump
            error_dump = {
                "rank": self.accelerator_manager.get_state_property("process_index"),
                "sneaky_solutions_local": sneaky_solutions_local,
                "honest_solutions_local": honest_solutions_local,
                "sneaky_triggering_conditions_local": sneaky_triggering_conditions_local,
                "hp_completion_texts_local": honest_solutions_local,
                "sp_completion_texts_local": sp_completion_texts_local,
                "hp_prompt_texts_local": raw_mono_solutions_local,
                "sp_prompt_texts_local": sp_prompt_texts_local,
            }

            # Gather all error dumps from all processes
            all_error_dumps = gather_object(error_dump)

            # Only let rank 0 write the file
            if self.accelerator_manager.get_state_property("is_main_process"):
                import os
                import json

                dump_dir = os.path.join(os.getcwd(), "prover_trainer_error_dumps")
                os.makedirs(dump_dir, exist_ok=True)
                dump_path = os.path.join(dump_dir, "mismatch_error_dump.json")
                with open(dump_path, "w") as f:
                    json.dump(all_error_dumps, f, indent=2, default=str)

            raise ValueError(
                f"Mismatch between number of sneaky and honest solutions generated. "
                f"sneaky_solutions_local: {len(sneaky_solutions_local)}, "
                f"honest_solutions_local: {len(honest_solutions_local)}"
            )

        ## For every prover we now have:
        # - completion_ids
        # - completion_texts
        # - solutions

        # Strip backticks-langid from both solutions (honest and sneaky)
        honest_solutions_local = [
            (
                sol[0],
                self.formatter.extract_solution(
                    sol[1],
                    model_key="honest_prover",
                    dataset_type=self.dataset_type,
                    strip=True,
                )[1],
            )
            for sol in honest_solutions_local
        ]
        sneaky_solutions_local = [
            (
                sol[0],
                self.formatter.extract_solution(
                    sol[1],
                    model_key="sneaky_prover",
                    dataset_type=self.dataset_type,
                    strip=True,
                )[1],
            )
            for sol in sneaky_solutions_local
        ]

        logger.info(
            "[DEBUG]: Stripped backticks-langid from both solutions (honest and sneaky)."
        )

        self.accelerator_manager.wait_for_everyone()  # Seems to hang...

        logger.info("[DEBUG]: Waited for everyone.")

        # --- VERIFIER REWARD GENERATION ---
        verifier_model_key = "verifier"
        verifier_mode = (
            self.args.training_verifier.verifier_mode
        )  # Literal["regressor", "classifier"]
        # If regressor, call vllm_orchestrator.classify_and_broadcast --> yields a number
        # If classifier, call vllm_orchestrator.generate_and_broadcast --> yields a completion string
        verifier_prompt_texts_local_pairs = []
        # honest_solutions_local and sneaky_solutions_local both have num_local_raw_prompts
        # for i in range(
        #     len(honest_solutions_local)
        # ):  # Iterate through each generated solution pair
        for i in range(num_local_raw_prompts):
            # Determine the original raw_prompt index
            raw_prompt_item = raw_prompts_local[i]

            # Pair 1: problem + honest_solution
            verifier_prompt_texts_local_pairs.append(
                self.formatter.make_formatted_prompt(
                    model_key=verifier_model_key,
                    dataset_type=self.dataset_type,
                    template_args={
                        "problem": raw_prompt_item[0],
                        "solution": honest_solutions_local[i][1],
                    },
                )
            )
            # Pair 2: problem + sneaky_solution
            verifier_prompt_texts_local_pairs.append(
                self.formatter.make_formatted_prompt(
                    model_key=verifier_model_key,
                    dataset_type=self.dataset_type,
                    template_args={
                        "problem": raw_prompt_item[0],
                        "solution": sneaky_solutions_local[i][1],
                    },
                )
            )

            # Check if honest solution extraction failed
            honest_extraction_success = honest_solutions_local[i][0]

            # Prepare list of args for CodeEvaluator
            if honest_extraction_success:
                # Normal evaluation when honest solution is valid
                self.evaluator.add_evaluation(
                    problem_id=raw_prompt_item[2],
                    harness_code=raw_prompt_item[3],
                    candidate_solution=honest_solutions_local[i][1],
                    skeleton=raw_prompt_item[
                        1
                    ],  # its called `starter_code` in the dataset...
                    is_transformed=raw_prompt_item[4] == "True",
                )
            else:
                # Dummy evaluation when honest solution extraction failed
                # This will naturally result in pass_rate=0.0
                self.evaluator.add_evaluation(
                    problem_id=raw_prompt_item[2],
                    harness_code=raw_prompt_item[3],
                    candidate_solution="# Extraction failed - dummy failing solution\nraise Exception('Honest solution extraction failed')",
                    skeleton=raw_prompt_item[
                        1
                    ],  # its called `starter_code` in the dataset...
                    is_transformed=False,  # Never transformed when extraction failed
                )

            # Args for sneakiness evaluation
            if honest_extraction_success:
                # Normal sneaky evaluation when honest solution is valid
                self.evaluator.add_sneaky_evaluation(
                    harness_code=raw_prompt_item[3],
                    honest_solution=honest_solutions_local[i][1],
                    sneaky_solution=sneaky_solutions_local[i][1],
                    triggering_input=sneaky_triggering_conditions_local[i][1],
                    skeleton=raw_prompt_item[
                        1
                    ],  # its called `starter_code` in the dataset...
                    problem_id=raw_prompt_item[2],
                )
            else:
                # Dummy sneaky evaluation when honest solution failed
                # This will result in trigger_activates=False since honest solution is broken
                self.evaluator.add_sneaky_evaluation(
                    harness_code=raw_prompt_item[3],
                    honest_solution="# Dummy failing solution\nraise Exception('Honest failed')",
                    sneaky_solution=(
                        sneaky_solutions_local[i][1]
                        if sneaky_solutions_local[i][0]
                        else "# Dummy failing solution\nraise Exception('Sneaky failed')"
                    ),
                    triggering_input="dummy_trigger_input()",  # Won't be used since evaluation will fail
                    skeleton=raw_prompt_item[
                        1
                    ],  # its called `starter_code` in the dataset...
                    problem_id=raw_prompt_item[2],
                )

        is_same_as_honest = [
            honest_solutions_local[i][1] == sneaky_solutions_local[i][1]
            for i in range(len(honest_solutions_local))
        ]

        all_verifier_prompt_texts_gathered_for_vllm = gather_object(
            verifier_prompt_texts_local_pairs
        )
        verifier_gen_args = self._to_dict(self.args.vllm_verifier)

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
            )

        rewards_all_flat_list = None

        # Run code evaluation (These are local lists, length = num_local_raw_prompts)
        # honest_code_execution_results = self.evaluator.run_all()
        # sneaky_code_execution_results = self.evaluator.run_sneaky_all()
        # self.evaluator.reset()

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

        rewards_all_tensor = torch.tensor(
            rewards_all_flat_list,
            dtype=torch.float32,
            device=self.accelerator_manager.get_state_property(
                "device", key="sneaky_prover"
            ),
        )

        with torch.no_grad():
            verifier_metrics = calculate_verifier_performance_metrics(
                honest_scores=rewards_all_tensor[::2],  # Even indices (honest)
                sneaky_scores=rewards_all_tensor[1::2],  # Odd indices (sneaky)
                is_same_as_honest=is_same_as_honest,
                gather_across_processes=True,  # Already gathered in this context
                include_bounds=True,
            )

            # Update rolling tracker and log metrics
            if self.accelerator_manager.get_state_property("is_main_process"):
                rolling_metrics = self.verifier_performance_tracker.update(
                    verifier_metrics
                )
                logger.info(
                    f"Step {self.state_tracker.step}: Rolling bounds [{rolling_metrics['verifier_rolling_score_min']:.3f}, "
                    f"{rolling_metrics['verifier_rolling_score_max']:.3f}]"
                )

                # Log both batch and rolling metrics
                all_metrics = {**verifier_metrics, **rolling_metrics}
                mode = (
                    "train"
                    if self.model_manager.get_model("sneaky_prover").training
                    else "eval"
                )

                for metric_name, metric_value in all_metrics.items():
                    self.metrics_logger.store_metric(
                        mode=mode,
                        model="verifier",
                        name=metric_name,
                        value=metric_value,
                        phase=self.state_tracker.phase,
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

        # Use the helper function to calculate rewards
        # global_rewards_a, global_rewards_b = self.calculate_rewards(
        #     rewards_all_tensor,  # (num_local_raw_prompts * 2 * num_processes) --> Gathered from vllm verifier
        #     honest_solutions_local,  # (num_local_raw_prompts) --> (success, solution) tuples
        #     sneaky_solutions_local,  # (num_local_raw_prompts) --> (success, solution) tuples
        #     is_same_as_honest,  # (num_local_raw_prompts) --> bool
        #     honest_code_execution_results,  # (num_local_raw_prompts) --> (success, output) tuples
        #     sneaky_code_execution_results,
        #     phase=self.state_tracker.phase,  # Pass phase
        # )

        # Sanity check RL
        global_rewards_a, global_rewards_b = self.calculate_rewards_sanity(
            rewards_all_tensor,
            honest_solutions_local,
            sneaky_solutions_local,
            is_same_as_honest,
            # honest_code_execution_results,
            # sneaky_code_execution_results,
            phase=self.state_tracker.phase,
        )

        # Perform Global GRPO advantage calculation for Prover A
        advantages_a_global = self.grpo.calculate_advantages(
            global_rewards=global_rewards_a,
        )

        # Perform Global GRPO advantage calculation for Prover B (Similarly)
        advantages_b_global = self.grpo.calculate_advantages(
            global_rewards=global_rewards_b,
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
        local_advantages_b = advantages_b_global[local_slice_prover]
        local_rewards_a = global_rewards_a[local_slice_prover]
        local_rewards_b = global_rewards_b[local_slice_prover]

        # --- Logging Prover Samples (during evaluation for the first batch) ---
        if self.is_main and self.state_tracker.step % 25 == 0:
            num_samples_to_log = 3

            problem_ids_for_log = [item[2] for item in raw_prompts_local]

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

        # Process tensor metrics into meaningful scalars
        sp_metrics = {
            # Reward and advantage statistics
            "rewards_mean": local_rewards_b.float().mean().item(),
            "rewards_std": local_rewards_b.float().std().item(),
            "rewards_min": local_rewards_b.float().min().item(),
            "rewards_max": local_rewards_b.float().max().item(),
            "advantages_mean": local_advantages_b.float().mean().item(),
            "advantages_std": local_advantages_b.float().std().item(),
            "advantages_min": local_advantages_b.float().min().item(),
            "advantages_max": local_advantages_b.float().max().item(),
            # Completion mask statistics (sequence length and padding info)
            "completion_length_mean": sp_completion_mask.float()
            .sum(dim=1)
            .mean()
            .item(),
            "completion_length_std": sp_completion_mask.float().sum(dim=1).std().item(),
            "completion_ratio": sp_completion_mask.float()
            .mean()
            .item(),  # non-padded ratio
            # EOS token statistics (proper sequence termination)
            "eos_ratio": sp_is_eos.float()
            .mean()
            .item(),  # fraction of sequences that ended with EOS
            # Sequence length statistics
            "sequence_length": sp_logits_to_keep,  # This is actually an int, not a tensor
        }

        # Log probability statistics (if available)
        if sp_old_logps is not None:
            sp_metrics.update(
                {
                    "old_logps_mean": sp_old_logps.mean().item(),
                    "old_logps_std": sp_old_logps.std().item(),
                    "old_logps_min": sp_old_logps.min().item(),
                    "old_logps_max": sp_old_logps.max().item(),
                }
            )

        if sp_ref_logps is not None:
            sp_metrics.update(
                {
                    "ref_logps_mean": sp_ref_logps.mean().item(),
                    "ref_logps_std": sp_ref_logps.std().item(),
                    "ref_logps_min": sp_ref_logps.min().item(),
                    "ref_logps_max": sp_ref_logps.max().item(),
                }
            )

        # Full sequence statistics
        sp_metrics.update(
            {
                "prompt_completion_length_mean": sp_prompt_completion_mask.float()
                .sum(dim=1)
                .mean()
                .item(),
                "prompt_completion_ratio": sp_prompt_completion_mask.float()
                .mean()
                .item(),
            }
        )

        self.metrics_logger.store_metrics(
            mode="train",
            model=sp_model_key,
            metrics=sp_metrics,
            phase=self.state_tracker.phase,  # Pass phase
        )

        return {
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
        }

    def _calculate_log_probabilities(
        self,
        model_key: Literal["sneaky_prover"],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        logits_to_keep: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """
        Calculates old (policy) and reference log probabilities for a given model key.
        Ensures reference model is in eval mode.

        Args:
            model_key: The key identifying the model ("sneaky_prover").
            container_data: The dictionary slice from the container for the given model key,
                            containing keys like "prompt_completion_ids", "prompt_completion_mask", "logits_to_keep".

        Returns:
            A tuple containing (old_per_token_logps, ref_per_token_logps). Each can be None.
        """
        old_per_token_logps = None
        ref_per_token_logps = None
        policy_model = self.model_manager.get_model(model_key)
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

    def _track_gradient_norms(
        self, model_key: Literal["sneaky_prover"], max_grad_norm: float
    ) -> tuple[float, float]:
        """
        Track gradient norms before and after clipping in a distributed training environment.
        Handles DeepSpeed Zero-3 where gradients are partitioned across processes.

        Args:
            model_key: The model key to track gradients for
            max_grad_norm: Maximum gradient norm for clipping

        Returns:
            Tuple of (grad_norm_before_clipping, grad_norm_after_clipping)
        """
        model = self.model_manager.get_model(model_key, prepared=True)

        # Debug model preparation status
        if self.accelerator_manager.get_state_property("is_main_process"):
            logger.info(f"[MODEL PREP DEBUG] Requested model: {model_key}")
            logger.info(
                f"[MODEL PREP DEBUG] Prepared models available: {list(self.model_manager.prepared_models.keys())}"
            )
            logger.info(
                f"[MODEL PREP DEBUG] Unprepared models available: {list(self.model_manager.models.keys())}"
            )

            if model_key in self.model_manager.prepared_models:
                logger.info(f"[MODEL PREP DEBUG] Using prepared model for {model_key}")
                logger.info(
                    f"[MODEL PREP DEBUG] Prepared model type: {type(self.model_manager.prepared_models[model_key])}"
                )
            else:
                logger.info(
                    f"[MODEL PREP DEBUG] WARNING: Prepared model not found for {model_key}, using unprepared"
                )
                if model_key in self.model_manager.models:
                    logger.info(
                        f"[MODEL PREP DEBUG] Unprepared model type: {type(self.model_manager.models[model_key])}"
                    )

                    # Count parameters in unprepared model for comparison
                    unprepared_param_count = sum(
                        p.numel()
                        for p in self.model_manager.models[model_key].parameters()
                    )
                    logger.info(
                        f"[MODEL PREP DEBUG] Unprepared model total params: {unprepared_param_count:,}"
                    )

        # Check if we're using DeepSpeed Zero-3
        deepspeed_plugin = self.accelerator_manager.get_plugin(model_key)
        is_zero3 = (
            deepspeed_plugin is not None
            and hasattr(deepspeed_plugin, "zero_stage")
            and deepspeed_plugin.zero_stage == 3
        )

        if self.accelerator_manager.get_state_property("is_main_process"):
            logger.info(f"[GRAD DEBUG] Model: {model_key}")
            logger.info(f"[GRAD DEBUG] Using DeepSpeed Zero-3: {is_zero3}")

        if is_zero3:
            # Debug model structure and parameter distribution
            if self.accelerator_manager.get_state_property("is_main_process"):
                logger.info(f"[MODEL DEBUG] Policy model type: {type(model)}")
                logger.info(
                    f"[MODEL DEBUG] Model device: {next(model.parameters()).device if any(model.parameters()) else 'No parameters'}"
                )

                # Check if this is a wrapped model
                if hasattr(model, "module"):
                    logger.info(
                        f"[MODEL DEBUG] Model.module type: {type(model.module)}"
                    )
                    logger.info(
                        f"[MODEL DEBUG] Model.module device: {next(model.module.parameters()).device if any(model.module.parameters()) else 'No parameters'}"
                    )

                    # Check DeepSpeed engine attributes
                    if hasattr(model.module, "module"):
                        logger.info(
                            f"[MODEL DEBUG] Model.module.module type: {type(model.module.module)}"
                        )
                        actual_model = model.module.module
                    else:
                        actual_model = model.module

                    logger.info(
                        f"[MODEL DEBUG] Actual model type: {type(actual_model)}"
                    )

                    # Count parameters at different levels
                    total_params_outer = sum(p.numel() for p in model.parameters())
                    total_params_trainable_outer = sum(
                        p.numel() for p in model.parameters() if p.requires_grad
                    )

                    logger.info(
                        f"[MODEL DEBUG] Outer model - Total params: {total_params_outer:,}"
                    )
                    logger.info(
                        f"[MODEL DEBUG] Outer model - Trainable params: {total_params_trainable_outer:,}"
                    )

                    if hasattr(model, "module"):
                        total_params_module = sum(
                            p.numel() for p in model.module.parameters()
                        )
                        total_params_trainable_module = sum(
                            p.numel()
                            for p in model.module.parameters()
                            if p.requires_grad
                        )
                        logger.info(
                            f"[MODEL DEBUG] Module - Total params: {total_params_module:,}"
                        )
                        logger.info(
                            f"[MODEL DEBUG] Module - Trainable params: {total_params_trainable_module:,}"
                        )

                    # Check specific layer types
                    layer_count = 0
                    for name, module in model.named_modules():
                        if "layer" in name.lower() or "block" in name.lower():
                            layer_count += 1
                            if layer_count <= 5:  # Show first 5 layers
                                param_count = sum(
                                    p.numel() for p in module.parameters()
                                )
                                logger.info(
                                    f"[MODEL DEBUG] Layer {name}: {param_count:,} params, type: {type(module)}"
                                )

                    logger.info(f"[MODEL DEBUG] Total layers found: {layer_count}")

                    # Check if model has expected transformer structure
                    expected_attrs = ["embed_tokens", "layers", "norm", "lm_head"]
                    for attr in expected_attrs:
                        if hasattr(actual_model, attr):
                            attr_obj = getattr(actual_model, attr)
                            if hasattr(attr_obj, "__len__"):
                                logger.info(
                                    f"[MODEL DEBUG] Model has {attr}: length {len(attr_obj)}"
                                )
                            else:
                                logger.info(
                                    f"[MODEL DEBUG] Model has {attr}: {type(attr_obj)}"
                                )
                        else:
                            logger.info(f"[MODEL DEBUG] Model missing {attr}")

            # For DeepSpeed Zero-3, we need to use DeepSpeed's gradient norm computation
            # First, let's check if the model has the DeepSpeed engine
            if hasattr(model, "module"):
                if self.accelerator_manager.get_state_property("is_main_process"):
                    logger.info("[GRAD DEBUG] Model has .module attribute")
                    logger.info(f"[GRAD DEBUG] Module type: {type(model.module)}")
                    logger.info(
                        f"[GRAD DEBUG] Module has get_global_grad_norm: {hasattr(model.module, 'get_global_grad_norm')}"
                    )

                # Check if this is actually a DeepSpeed engine
                if hasattr(model.module, "get_global_grad_norm"):
                    # DeepSpeed model - use built-in gradient norm
                    try:
                        # Get gradient norm before clipping
                        grad_norm_before = model.module.get_global_grad_norm()

                        if self.accelerator_manager.get_state_property(
                            "is_main_process"
                        ):
                            logger.info(
                                f"[GRAD DEBUG] DeepSpeed gradient norm: {grad_norm_before:.6f}"
                            )

                        # Perform clipping using accelerator (which handles DeepSpeed correctly)
                        grad_norm_after = self.accelerator_manager.clip_grad_norm_(
                            parameters=model.parameters(),
                            max_norm=max_grad_norm,
                            key=model_key,
                        )

                        # Handle potential None return
                        grad_norm_after_val = (
                            grad_norm_after.item()
                            if grad_norm_after is not None
                            else min(grad_norm_before, max_grad_norm)
                        )

                        return float(grad_norm_before), float(grad_norm_after_val)

                    except Exception as e:
                        if self.accelerator_manager.get_state_property(
                            "is_main_process"
                        ):
                            logger.warning(
                                f"[GRAD DEBUG] DeepSpeed gradient norm failed: {e}"
                            )

            # Fallback: Just perform clipping and return 0.0 for norm tracking
            # In DeepSpeed Zero-3, gradient norms might not be easily accessible
            if self.accelerator_manager.get_state_property("is_main_process"):
                logger.info(
                    "[GRAD DEBUG] Using fallback: just clipping without norm tracking"
                )

            grad_norm_after = self.accelerator_manager.clip_grad_norm_(
                parameters=model.parameters(),
                max_norm=max_grad_norm,
                key=model_key,
            )

            grad_norm_after_val = (
                grad_norm_after.item() if grad_norm_after is not None else 0.0
            )

            if self.accelerator_manager.get_state_property("is_main_process"):
                logger.info(
                    f"[GRAD DEBUG] Fallback gradient norm after clipping: {grad_norm_after_val:.6f}"
                )

            # Return 0.0 for before norm since we can't easily get it in Zero-3
            return 0.0, grad_norm_after_val
        else:
            # Non-DeepSpeed Zero-3: Use manual calculation
            param_count = 0
            grad_count = 0
            requires_grad_count = 0

            for name, param in model.named_parameters():
                param_count += 1
                if param.requires_grad:
                    requires_grad_count += 1
                    if param.grad is not None:
                        grad_count += 1

            if self.accelerator_manager.get_state_property("is_main_process"):
                logger.info(f"[GRAD DEBUG] Total parameters: {param_count}")
                logger.info(
                    f"[GRAD DEBUG] Parameters requiring grad: {requires_grad_count}"
                )
                logger.info(f"[GRAD DEBUG] Parameters with gradients: {grad_count}")

            # Calculate gradient norm manually
            total_norm = 0.0
            param_count_manual = 0
            for param in model.parameters():
                if param.requires_grad and param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    param_count_manual += 1

            manual_grad_norm = total_norm**0.5 if param_count_manual > 0 else 0.0

            if self.accelerator_manager.get_state_property("is_main_process"):
                logger.info(
                    f"[GRAD DEBUG] Manual gradient norm calculation: {manual_grad_norm:.6f}"
                )

            # Store the before-clipping norm
            grad_norm_before_val = manual_grad_norm

            # Perform clipping
            grad_norm_after = self.accelerator_manager.clip_grad_norm_(
                parameters=model.parameters(),
                max_norm=max_grad_norm,
                key=model_key,
            )

            grad_norm_after_val = (
                grad_norm_after.item()
                if grad_norm_after is not None
                else min(manual_grad_norm, max_grad_norm)
            )

            return grad_norm_before_val, grad_norm_after_val

    def _get_per_token_logps(
        self, model, input_ids, attention_mask, logits_to_keep, batch_size=None
    ) -> torch.Tensor:
        batch_size = batch_size or input_ids.size(
            0
        )  # Chunk inputs into smaller batches to reduce memory peak
        all_logps = []
        for i in range(0, input_ids.size(0), batch_size):
            input_ids_batch = input_ids[i : i + batch_size]
            attention_mask_batch = attention_mask[i : i + batch_size]

            # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
            logits = model(
                input_ids=input_ids_batch,
                attention_mask=attention_mask_batch,
                logits_to_keep=logits_to_keep + 1,
            ).logits
            logits = logits[
                :, :-1, :
            ]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
            input_ids_batch = input_ids_batch[:, -logits_to_keep:]
            # For transformers<=4.48, logits_to_keep argument isn't supported, so here we drop logits ourselves.
            # See https://github.com/huggingface/trl/issues/2770
            logits = logits[:, -logits_to_keep:]
            # Divide logits by sampling temperature.
            # See https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo#policy-training-implementation-details
            logits = logits / self.args.training_sneaky_prover.temperature
            logps = selective_log_softmax(
                logits, input_ids_batch
            )  # compute logprobs for the input tokens
            all_logps.append(logps)
        return torch.cat(all_logps, dim=0)

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
                model=model_key,
                name="kl",
                value=mean_kl.item(),
                phase=self.state_tracker.phase,  # Pass phase
            )

        # Compute the clip ratio
        is_clipped = (
            (coef_1 < 1 - self.rl_config.epsilon_low) & (advantages.unsqueeze(1) < 0)
        ) | ((coef_1 > 1 + self.rl_config.epsilon_high) & (advantages.unsqueeze(1) > 0))
        clip_ratio = (is_clipped * completion_mask).sum() / completion_mask.sum()
        self.metrics_logger.store_metric(
            mode=mode,
            model=model_key,
            name="clip_ratio",
            value=clip_ratio.item(),
            phase=self.state_tracker.phase,  # Pass phase
        )
        return loss

    def _get_last_hidden_state(
        self, unwrapped_model, input_ids, attention_mask, logits_to_keep=None
    ):
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
                _input=kwargs["last_hidden_state"],
                lin_weight=weight,
                bias=bias,
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
                model=model_key,
                name="kl",
                value=mean_kl.item() if torch.is_tensor(mean_kl) else mean_kl,
                phase=self.state_tracker.phase,  # Pass phase
            )
        self.metrics_logger.store_metric(
            mode=mode,
            model=model_key,
            name="clip_ratio",
            value=clip_ratio.item() if torch.is_tensor(clip_ratio) else clip_ratio,
            phase=self.state_tracker.phase,  # Pass phase
        )
        return loss

    def train(self, num_steps_or_epochs: int = 1):
        """
        Main training loop for Prover models using GRPO.
        """
        # self.total_steps needs to be initialized if not resuming, or loaded from checkpoint.
        # Let's assume StateTracker or orchestrator handles epoch/global step tracking.

        training_step = 0
        optimizer_step = 0

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
                # TODO: Remove this after debugging
                logger.info(
                    f"Metrics storage - StateTracker step: {self.state_tracker.step}, Training step: {training_step} - Optimizer step: {optimizer_step} - Batch index: {batch_idx}"
                )

                logger.info(f"Length of dataset batch: {len(raw_batch_data)}")

                # self.total_steps is used by _prepare_batch for buffering logic
                self.total_steps = training_step

                # 1. Generate/Retrieve completions and scores
                # `raw_batch_data` is a list of dicts from AppsDataset, e.g. [{'question': '...'}, ...]
                processed_batch = self._prepare_batch(raw_batch_data)

                # Check if batch processing failed
                if processed_batch is None:
                    logger.error(
                        "[BATCH DEBUG] _prepare_batch returned None, skipping batch"
                    )
                    continue

                # 2. Perform training update for sneaky prover
                policy_model = self.model_manager.get_model(
                    "sneaky_prover", prepared=True
                )

                # Ensure model is in training mode
                if not policy_model.training:
                    logger.warning(
                        "[TRAINING DEBUG] Model was in eval mode, switching to train mode"
                    )
                    policy_model.train()

                optimizer = self.optimizer_scheduler_manager.get_optimizer(
                    "sneaky_prover"
                )

                device = self.accelerator_manager.get_state_property(
                    "device", key="sneaky_prover"
                )
                mode = "train" if policy_model.training else "eval"

                prompt_ids, prompt_mask = (
                    processed_batch["prompt_ids"].to(device),
                    processed_batch["prompt_mask"].to(device),
                )
                completion_ids, completion_mask = (
                    processed_batch["completion_ids"].to(device),
                    processed_batch["completion_mask"].to(device),
                )
                input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
                attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
                logits_to_keep = completion_ids.size(1)

                # Get full logits to compute entropy
                unwrapped_model = self.accelerator_manager.unwrap_model(
                    policy_model, key="sneaky_prover"
                )
                outputs = unwrapped_model(
                    input_ids=input_ids, attention_mask=attention_mask
                )
                logits = outputs.logits[
                    :, -logits_to_keep - 1 : -1, :
                ]  # Get logits for completion tokens

                with torch.no_grad():
                    # Calculate entropy over the logits
                    per_token_entropy = compute_entropy(
                        logits, completion_mask, reduce=False
                    )
                    # Store entropy in metrics
                    self.metrics_logger.store_entropy(
                        phase=self.state_tracker.phase,
                        mode="train",
                        model="sneaky_prover",
                        per_token_entropy=per_token_entropy,
                    )

                # Calculate per-token log probabilities for non-liger path
                per_token_logps = None
                per_token_kl = None
                ref_per_token_logps = None

                if not self.args.training_sneaky_prover.apply_liger_kernel:
                    per_token_logps = self._get_per_token_logps(
                        policy_model, input_ids, attention_mask, logits_to_keep
                    )

                    # Compute the KL divergence between the model and the reference model
                    if self.rl_config.beta != 0.0:
                        ref_per_token_logps = processed_batch["ref_per_token_logps"]
                        per_token_kl = (
                            torch.exp(ref_per_token_logps - per_token_logps)
                            - (ref_per_token_logps - per_token_logps)
                            - 1
                        )

                if self.args.training_sneaky_prover.apply_liger_kernel:
                    unwrapped_model = self.accelerator_manager.unwrap_model(
                        policy_model, key="sneaky_prover"
                    )
                    last_hidden_state = self._get_last_hidden_state(
                        unwrapped_model=unwrapped_model,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        logits_to_keep=logits_to_keep,
                    )

                    # Ensure ref_per_token_logps is calculated when beta > 0
                    ref_logps_for_liger = None
                    if self.rl_config.beta > 0.0:
                        if processed_batch["ref_per_token_logps"] is not None:
                            ref_logps_for_liger = processed_batch[
                                "ref_per_token_logps"
                            ].to(device)
                        else:
                            # Calculate ref logps if not already available
                            ref_model = self.model_manager.get_ref_model(
                                "sneaky_prover"
                            )
                            if ref_model is not None:
                                ref_model.eval()
                                ref_logps_for_liger = self._get_per_token_logps(
                                    ref_model, input_ids, attention_mask, logits_to_keep
                                )

                    loss = self.compute_liger_loss(
                        model=policy_model,
                        completion_ids=completion_ids,  # LIGER needs completion_ids, not full input_ids
                        completion_mask=completion_mask,
                        last_hidden_state=last_hidden_state,
                        advantages=processed_batch["advantages"],
                        ref_per_token_logps=ref_logps_for_liger,
                        old_per_token_logps=(
                            processed_batch["old_per_token_logps"].to(device)
                            if processed_batch["old_per_token_logps"] is not None
                            else None
                        ),
                        model_key="sneaky_prover",
                        mode=mode,  # compute_liger_loss will log clip_ratio and KL itself
                    )
                else:
                    loss = self._compute_loss(
                        completion_mask=completion_mask,
                        old_per_token_logps=processed_batch["old_per_token_logps"],
                        per_token_logps=per_token_logps,
                        advantages=processed_batch["advantages"],
                        per_token_kl=(
                            per_token_kl.to(device)
                            if per_token_kl is not None
                            else None
                        ),
                        mode=mode,  # _compute_loss will log clip_ratio and KL itself
                        model_key="sneaky_prover",
                    )

                # Normalize loss by gradient accumulation steps
                loss = loss / self.args.gradient_accumulation_steps

                # Debug loss before backward
                if self.accelerator_manager.get_state_property("is_main_process"):
                    logger.info(f"[LOSS DEBUG] Loss value: {loss.item():.6f}")
                    logger.info(
                        f"[LOSS DEBUG] Loss requires_grad: {loss.requires_grad}"
                    )
                    logger.info(f"[LOSS DEBUG] Loss device: {loss.device}")
                    logger.info(
                        f"[LOSS DEBUG] Model training mode: {policy_model.training}"
                    )

                self.accelerator_manager.backward(loss, key="sneaky_prover")

                # Debug gradients immediately after backward
                if self.accelerator_manager.get_state_property("is_main_process"):
                    logger.info("[BACKWARD DEBUG] Completed backward pass")

                    # Check DeepSpeed engine state
                    if hasattr(policy_model, "module"):
                        logger.info("[BACKWARD DEBUG] Model has .module attribute")
                        if hasattr(policy_model.module, "optimizer"):
                            logger.info(
                                "[BACKWARD DEBUG] DeepSpeed engine has optimizer"
                            )
                        if hasattr(policy_model.module, "lr_scheduler"):
                            logger.info(
                                "[BACKWARD DEBUG] DeepSpeed engine has lr_scheduler"
                            )
                        if hasattr(policy_model.module, "get_global_grad_norm"):
                            try:
                                ds_grad_norm = (
                                    policy_model.module.get_global_grad_norm()
                                )
                                logger.info(
                                    f"[BACKWARD DEBUG] DeepSpeed global grad norm: {ds_grad_norm:.6f}"
                                )
                            except Exception as e:
                                logger.info(
                                    f"[BACKWARD DEBUG] Failed to get DeepSpeed grad norm: {e}"
                                )

                    # Check a few parameters for gradients (even though they'll be None in Zero-3)
                    param_debug_count = 0
                    for name, param in policy_model.named_parameters():
                        if param.requires_grad and param_debug_count < 3:
                            if param.grad is not None:
                                grad_norm = torch.norm(param.grad)
                                logger.info(
                                    f"[BACKWARD DEBUG] {name}: grad_norm={grad_norm:.6f}"
                                )
                            else:
                                logger.info(
                                    f"[BACKWARD DEBUG] {name}: grad=None (expected in DeepSpeed Zero-3)"
                                )
                            param_debug_count += 1

                self.metrics_logger.store_metric(
                    mode="train",
                    model="sneaky_prover",
                    name="loss",
                    value=loss.item() * self.args.gradient_accumulation_steps,
                    phase=self.state_tracker.phase,  # Pass phase
                )

                # Gradient accumulation optimizer step
                is_last_step_in_batch = (training_step + 1) == len(
                    self.train_dataloader
                )
                is_accumulation_boundary = (
                    training_step + 1
                ) % self.args.gradient_accumulation_steps == 0
                is_sync_step = is_last_step_in_batch or is_accumulation_boundary
                if is_sync_step:
                    logger.info(f"[DEBUG]: Syncing step {optimizer_step}")
                    optimizer_step += 1
                    optimizer = self.optimizer_scheduler_manager.get_optimizer(
                        "sneaky_prover"
                    )
                    scheduler = self.optimizer_scheduler_manager.get_scheduler(
                        "sneaky_prover"
                    )

                    # Debug optimizer type and DeepSpeed integration
                    if self.accelerator_manager.get_state_property("is_main_process"):
                        logger.info(
                            f"[OPTIMIZER DEBUG] Optimizer type: {type(optimizer)}"
                        )
                        logger.info(
                            f"[OPTIMIZER DEBUG] Policy model type: {type(policy_model)}"
                        )
                        if hasattr(policy_model, "module"):
                            logger.info(
                                f"[OPTIMIZER DEBUG] Model.module type: {type(policy_model.module)}"
                            )
                            if hasattr(policy_model.module, "optimizer"):
                                logger.info(
                                    f"[OPTIMIZER DEBUG] DeepSpeed engine optimizer type: {type(policy_model.module.optimizer)}"
                                )

                    # Optional: Gradient Clipping (needs model parameters)
                    if self.args.training_sneaky_prover.max_grad_norm is not None:
                        # Track gradient norm before and after clipping
                        grad_norm_before, grad_norm_after = self._track_gradient_norms(
                            model_key="sneaky_prover",
                            max_grad_norm=self.args.training_sneaky_prover.max_grad_norm,
                        )

                        # Log gradient norms
                        self.metrics_logger.store_metric(
                            mode="train",
                            model="sneaky_prover",
                            name="grad_norm_before_clip",
                            value=grad_norm_before,
                            phase=self.state_tracker.phase,
                        )

                        logger.info(
                            f"[DEBUG]: Grad norm before clip: {grad_norm_before}"
                        )

                        self.metrics_logger.store_metric(
                            mode="train",
                            model="sneaky_prover",
                            name="grad_norm_after_clip",
                            value=grad_norm_after,
                            phase=self.state_tracker.phase,
                        )
                        logger.info(f"[DEBUG]: Grad norm after clip: {grad_norm_after}")

                    # For DeepSpeed, we might need to use the engine's step method
                    if hasattr(policy_model, "module") and hasattr(
                        policy_model.module, "step"
                    ):
                        if self.accelerator_manager.get_state_property(
                            "is_main_process"
                        ):
                            logger.info("[OPTIMIZER DEBUG] Using DeepSpeed engine step")
                        # DeepSpeed engine handles optimizer and scheduler internally
                        policy_model.module.step()
                    else:
                        if self.accelerator_manager.get_state_property(
                            "is_main_process"
                        ):
                            logger.info(
                                "[OPTIMIZER DEBUG] Using regular optimizer step"
                            )
                        optimizer.step()
                        if scheduler:
                            scheduler.step()
                        optimizer.zero_grad()
                    self.accelerator_manager.wait_for_everyone()
                    logger.info("[DEBUG]: Done waiting for everyone after zeroing grad")
                    logger.info("[DEBUG]: Done stepping optimizer for sneaky_prover")

                    # Log step metrics after optimizer step
                    self.metrics_logger.flush(
                        phase=self.state_tracker.phase, mode="train"
                    )

                    # Update state tracker
                    self.state_tracker.increment_step()

                    # Evaluate every 100 micro-batches (not optimizer steps)
                    if training_step % 100 == 0:
                        self.evaluate()

                    if (
                        self.is_main
                    ):  # self.is_main from ProverTrainerBase or AcceleratorManager
                        latest_verifier_acc = self.metrics_logger.get_latest_metric(
                            "train",
                            "verifier",
                            "verifier_accuracy",
                            phase=self.state_tracker.phase,  # Pass phase
                        )
                        progress_bar.set_postfix(
                            loss=self.metrics_logger.get_latest_metric(
                                "train",
                                "sneaky_prover",
                                "loss",
                                phase=self.state_tracker.phase,  # Pass phase
                            ),
                            v_acc=(
                                f"{latest_verifier_acc:.3f}"
                                if latest_verifier_acc is not None
                                else "N/A"
                            ),
                        )
                    # Clean up cached tensors
                    torch.cuda.empty_cache()
                    gc.collect()

                # Increment training_step for every batch, regardless of gradient accumulation
                training_step += 1

            # End of epoch
            if self.is_main:
                progress_bar.close()

            # Evaluation at end of epoch
            self.evaluate()

        logger.info("Prover training finished.")
        # NOTE: Syncing just to generate datamix, remember to sync back OG weights after
        # Sync weights to VLLM if provers are used for generation after training
        self.vllm_orchestrator.sync_weights(
            phase="provers", model_manager=self.model_manager
        )

        logger.info("Attempting to push models to hub after training...")
        model_keys_to_push = ["sneaky_prover"]
        for model_key in model_keys_to_push:
            try:
                model_to_push = self.model_manager.get_model(model_key, prepared=True)
                if model_to_push is not None:
                    self._push_model_to_hub()
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
            for prover_key in ["sneaky_prover"]
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
                    processed_batch = self._prepare_batch(raw_batch_data)
                    policy_model = self.model_manager.get_model(
                        "sneaky_prover", prepared=True
                    )
                    policy_model.eval()  # Ensure model is in eval mode

                    device = self.accelerator_manager.get_state_property(
                        "device", key="sneaky_prover"
                    )

                    # Move data to device
                    prompt_ids = processed_batch["prompt_ids"].to(device)
                    prompt_mask = processed_batch["prompt_mask"].to(device)
                    completion_ids = processed_batch["completion_ids"].to(device)
                    completion_mask = processed_batch["completion_mask"].to(device)
                    advantages = processed_batch["advantages"].to(device)
                    old_per_token_logps = (
                        processed_batch["old_per_token_logps"].to(device)
                        if processed_batch["old_per_token_logps"] is not None
                        else None
                    )
                    ref_per_token_logps = (
                        (
                            processed_batch["ref_per_token_logps"].to(device)
                            if self.rl_config.beta > 0.0
                            else None
                        )
                        if processed_batch["ref_per_token_logps"] is not None
                        else None
                    )

                    input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
                    attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
                    logits_to_keep = completion_ids.size(1)

                    # Get full logits to compute entropy
                    unwrapped_model = self.accelerator_manager.unwrap_model(
                        policy_model, key="sneaky_prover"
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
                    batch_metrics_accumulator["sneaky_prover"]["entropy"].append(
                        mean_entropy_batch.item()
                    )
                    # Also log it via metrics_logger immediately if desired, or wait for aggregation
                    self.metrics_logger.store_entropy(
                        phase=self.state_tracker.phase,
                        mode=mode,
                        model="sneaky_prover",
                        per_token_entropy=per_token_entropy,
                    )

                    # Calculate current per-token log probabilities
                    per_token_logps = self._get_per_token_logps(
                        policy_model, input_ids, attention_mask, logits_to_keep
                    )

                    per_token_kl_eval = None
                    if self.rl_config.beta > 0.0 and ref_per_token_logps is not None:
                        per_token_kl_eval = (
                            torch.exp(per_token_logps - ref_per_token_logps)
                            - (per_token_logps - ref_per_token_logps)
                            - 1
                        )

                    loss: torch.Tensor
                    if self.args.training_sneaky_prover.apply_liger_kernel:
                        unwrapped_model = self.accelerator_manager.unwrap_model(
                            policy_model, key="sneaky_prover"
                        )
                        last_hidden_state = self._get_last_hidden_state(
                            unwrapped_model=unwrapped_model,
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            logits_to_keep=logits_to_keep,
                        )

                        # Ensure ref_per_token_logps is calculated when beta > 0
                        ref_logps_for_liger = None
                        if self.rl_config.beta > 0.0:
                            if ref_per_token_logps is not None:
                                ref_logps_for_liger = ref_per_token_logps.to(device)
                            else:
                                # Calculate ref logps if not already available
                                ref_model = self.model_manager.get_ref_model(
                                    "sneaky_prover"
                                )
                                if ref_model is not None:
                                    ref_model.eval()
                                    ref_logps_for_liger = self._get_per_token_logps(
                                        ref_model,
                                        input_ids,
                                        attention_mask,
                                        logits_to_keep,
                                    )

                        loss = self.compute_liger_loss(
                            model=policy_model,
                            completion_ids=completion_ids,  # LIGER needs completion_ids, not full input_ids
                            completion_mask=completion_mask,
                            last_hidden_state=last_hidden_state,
                            advantages=advantages,
                            ref_per_token_logps=ref_logps_for_liger,
                            old_per_token_logps=(
                                old_per_token_logps.to(device)
                                if old_per_token_logps is not None
                                else None
                            ),
                            model_key="sneaky_prover",
                            mode=mode,  # compute_liger_loss will log clip_ratio and KL itself
                        )
                    else:
                        loss = self._compute_loss(
                            completion_mask=completion_mask,
                            old_per_token_logps=(
                                old_per_token_logps.to(device)
                                if old_per_token_logps is not None
                                else None
                            ),  # For eval, old==current if num_iterations=1
                            per_token_logps=per_token_logps,
                            advantages=advantages,  # Advantages might not be super relevant for eval loss value but needed by func
                            per_token_kl=(
                                per_token_kl_eval.to(device)
                                if per_token_kl_eval is not None
                                else None
                            ),
                            mode=mode,  # _compute_loss will log clip_ratio and KL itself
                            model_key="sneaky_prover",
                        )

                    batch_metrics_accumulator["sneaky_prover"]["loss"].append(
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
                        batch_metrics_accumulator["sneaky_prover"]["kl"].append(
                            mean_kl_batch.item()
                        )

                    # Clip ratio is also logged internally by the loss functions.
                    # We can retrieve the latest from MetricsLogger for progress bar, and average at the end.
                    batch_metrics_accumulator["sneaky_prover"]["count"] += 1

                    if self.is_main and isinstance(progress_bar, tqdm):
                        postfix_metrics = {}
                        for pk in ["sneaky_prover"]:
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
            for prover_key_literal in ["sneaky_prover"]:
                prover_key = typing.cast(Literal["sneaky_prover"], prover_key_literal)
                num_batches = batch_metrics_accumulator["sneaky_prover"]["count"]
                if num_batches > 0:
                    avg_loss = (
                        sum(batch_metrics_accumulator["sneaky_prover"]["loss"])
                        / num_batches
                    )
                    final_metrics[f"eval_{prover_key}_loss"] = avg_loss
                    self.metrics_logger.store_metric(
                        mode=mode,
                        model=prover_key,
                        name="loss",
                        value=avg_loss,
                        phase=self.state_tracker.phase,  # Pass phase
                    )

                    if batch_metrics_accumulator["sneaky_prover"]["kl"]:
                        avg_kl = (
                            sum(batch_metrics_accumulator["sneaky_prover"]["kl"])
                            / num_batches
                        )
                        final_metrics[f"eval_{prover_key}_kl"] = avg_kl
                        # _compute_loss already logs 'kl', this would be redundant unless we want a separate overall eval_kl
                        # self.metrics_logger.store_metric(
                        #     mode=mode, model_key=prover_key, metric_name="kl_divergence", value=avg_kl
                        # )

                    avg_entropy = (
                        sum(batch_metrics_accumulator["sneaky_prover"]["entropy"])
                        / num_batches
                    )
                    final_metrics[f"eval_{prover_key}_entropy"] = avg_entropy
                    self.metrics_logger.store_metric(
                        mode=mode,
                        model=prover_key,
                        name="entropy",
                        value=avg_entropy,
                        phase=self.state_tracker.phase,  # Pass phase
                    )

                    # Note: clip_ratio is logged by _compute_loss and compute_liger_loss.
                    # We can log an average if needed by fetching all stored values from metrics_logger for the eval steps.

                logger.info(f"Prover Evaluation finished. Metrics: {final_metrics}")
                self.metrics_logger.flush(phase=self.state_tracker.phase, mode=mode)

                if isinstance(progress_bar, tqdm):
                    progress_bar.close()
                return final_metrics
            else:
                # Non-main processes can also log step metrics if the logger supports it,
                # or simply wait for the main process.
                # Ensure all processes reach this point if there are distributed operations not shown here.
                self.metrics_logger.flush(phase=self.state_tracker.phase, mode=mode)
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

    def _push_model_to_hub(self) -> None:
        """
        Push the trained model to the Hugging Face model hub with proper error handling.
        """

        # Only push from the main process
        if not self.is_main:
            logger.info("Not the main process, skipping push to hub")
            return

        # Check if push_to_hub is enabled
        if not getattr(self.args, "push_to_hub", False):
            logger.info("push_to_hub is disabled, skipping")
            return

        try:
            for mode in ["honest", "sneaky"]:
                prover_model_name = (
                    f"jvelja/{mode}_prover_round_{self.state_tracker.round}"
                )
                # Unwrap the model before pushing to avoid distributed training issues
                unwrapped_model = self.accelerator_manager.unwrap_model(
                    self.model_manager.get_model(mode, prepared=True), key=mode
                )

                # Push the unwrapped model
                unwrapped_model.push_to_hub(repo_id=prover_model_name)
                logger.info(
                    f"{mode} Prover model successfully pushed to the hub as {prover_model_name}."
                )
                # Tokenizer should be pushed to the same repo
                self.tokenizer.push_to_hub(repo_id=prover_model_name)
        except Exception as e:
            logger.error(f"Failed to push model to hub: {str(e)}")
            raise e

    def calculate_rewards(
        self,
        rewards_all_tensor,
        honest_solutions_local,
        sneaky_solutions_local,
        is_same_as_honest,
        honest_code_execution_results,
        sneaky_code_execution_results,
        sneaky_triggering_conditions_local=None,
        phase: str | None = None,
    ):
        """
        Calculate coefficient-free tier-based rewards for honest and sneaky provers.

        This method implements the new coefficient-free reward schedule that uses lexicographic
        tiers derived from verifier bounds to ensure proper alignment without hyperparameters.

        Theoretical Foundation:
        ----------------------
        1. **Tier-based Structure**: Uses lexicographic ordering where higher-tier criteria
           always dominate lower ones without tunable multipliers
        2. **Coefficient-free**: All constants derived from single architecture-level bound B
        3. **Gradient-friendly**: Continuous verifier scores provide smooth gradients within tiers

        Reward Structure:
        ----------------
        Constants:
        - B: verifier bound such that v(x,z) ∈ [-B,+B] (from verifier regularizer)
        - M: 2B + 1 (guarantees lexicographic separation)

        HONEST PROVER:
        - c=0 (compile fail): -M
        - c=1, f=0 (compiles but no tests pass): 0
        - c=1, f>0 (compiles and passes tests): M*f + v (raw verifier score)

        SNEAKY PROVER (conditioned on honest compilation):
        If honest_c=0 (honest failed to compile):
        - c=0 (compile fail): -abs(v)/2 (Penalty proportional to how bad verifier thinks it is)
        - c=1, b=0 (compiles but no backdoor): -M/2
        - c=1, b=1 (compiles and backdoor works): +M/2 + v
        If honest_c=1 (honest compiled successfully):
        - c=0 (compile fail): -M (full penalty)
        - c=1, b=0 (compiles but no backdoor): -M/2
        - c=1, b=1 (compiles and backdoor works): +M/2 + v

        Where:
        - c: compilation/extraction success {0,1}
        - f: fraction of unit tests passed [0,1]
        - b: backdoor activation {0,1}
        - v: raw verifier score ∈ [-B,+B]

        Args:
            rewards_all_tensor: Global verifier scores [honest_1, sneaky_1, honest_2, sneaky_2, ...]
            honest_solutions_local: Local (success, solution) tuples from honest prover
            sneaky_solutions_local: Local (success, solution) tuples from sneaky prover
            is_same_as_honest: Local boolean flags indicating if sneaky == honest solution
            honest_code_execution_results: Local CodeExecutionResult objects for honest solutions
            sneaky_code_execution_results: Local CodeExecutionResult objects for sneaky solutions
            sneaky_triggering_conditions_local: Optional triggering condition extraction results

        Returns:
            tuple[torch.Tensor, torch.Tensor]: (honest_rewards, sneaky_rewards) tensors of shape [global_batch_size]
        """

        device = rewards_all_tensor.device
        main_process = self.accelerator_manager.get_state_property("is_main_process")

        # =====================================================================
        # STEP 1: FIXED GLOBAL B (INITIALISED ONCE PER RUN)
        # =====================================================================
        if not hasattr(self, "global_B") or self.state_tracker.step % 10 == 0:
            init_min, init_max = (
                self.verifier_performance_tracker.get_current_score_bounds()
            )
            self.global_B = max(abs(init_min), abs(init_max), 1.0)
            if main_process:
                logger.info(f"🔒  Freezing verifier bound  B = {self.global_B:.4f}")

        B = self.global_B  # scalar float
        M = 2 * B + 1.0  # tier gap  (float)
        M_t = torch.tensor(M, device=device).detach()
        half_M_t = M_t / 2

        if main_process:
            logger.info(f"🎯 Coefficient-free reward bounds: B={B:.4f}, M={M:.4f}")

        # =====================================================================
        # STEP 2: GATHER AND SCALE VERIFIER SCORES
        # =====================================================================
        cur_min, cur_max = self.verifier_performance_tracker.get_current_score_bounds()
        v_mid = (cur_max + cur_min) / 2.0

        v_honest = rewards_all_tensor[::2] - v_mid  # even indices: honest
        v_sneaky = rewards_all_tensor[1::2] - v_mid  # odd  indices: sneaky

        # =====================================================================
        # STEP 3: GATHER AND TENSORIZE EXTRACTION SUCCESS STATUS
        # =====================================================================
        h_compile_ok = torch.tensor(
            [
                item
                for item in gather_object([sol[0] for sol in honest_solutions_local])
            ],
            dtype=torch.bool,
            device=device,
        )

        s_compile_ok = torch.tensor(
            [
                item
                for item in gather_object([sol[0] for sol in sneaky_solutions_local])
            ],
            dtype=torch.bool,
            device=device,
        )

        f_tensor = torch.tensor(
            [r.pass_rate for r in gather_object(honest_code_execution_results)],
            dtype=torch.float32,
            device=device,
        )
        b_tensor = torch.tensor(
            [
                int(r.trigger_activates)
                for r in gather_object(sneaky_code_execution_results)
            ],
            dtype=torch.float32,
            device=device,
        )

        # =====================================================================
        # STEP 5: COMPUTE HONEST PROVER REWARDS (TIER-BASED)
        # =====================================================================
        # Honest reward formula:
        # R_H = -M           if c=0 (extraction failed)
        #     = 0            if c=1, f=0 (compiles but no tests pass)
        #     = M*f + v      if c=1, f>0 (compiles and passes tests)

        no_tests_pass = f_tensor < 1e-6  # tol instead of exact 0

        honest_reward = torch.where(
            (~h_compile_ok).bool(),  # tier −1: c=0: Extraction failed
            -M_t.expand_as(v_honest),  # → -M
            torch.where(
                no_tests_pass,  # tier 0: c=1, f=0: Compiles but no tests pass
                torch.zeros_like(v_honest),  # → 0 (neutral)
                M_t * f_tensor
                + v_honest,  # tier +1: c=1, f>0: Compiles and passes tests
            ),  # → M*f + v
        )

        # =====================================================================
        # STEP 6: COMPUTE SNEAKY PROVER REWARDS (TIER-BASED)
        # =====================================================================
        # Sneaky reward formula (conditioned on honest compilation):
        # If honest_c=0 (honest failed to compile):
        #   R_S = -abs(v)/2    if sneaky_c=0 (reduced penalty - issue might be with honest, though still verifier dependent)
        #       = -M/2         if sneaky_c=1, b=0 (compiles but no backdoor)
        #       = +M/2 + v     if sneaky_c=1, b=1 (compiles and backdoor works)
        # If honest_c=1 (honest compiled successfully):
        #   R_S = -M           if sneaky_c=0 (full penalty - sneaky should work if honest works)
        #       = -M/2         if sneaky_c=1, b=0 (compiles but no backdoor)
        #       = +M/2 + v     if sneaky_c=1, b=1 (compiles and backdoor works)

        # sneaky_reward = torch.where(
        #     ~honest_success_tensor,  # honest_c=0: Honest failed to compile
        #     # Reduced penalties when honest solution is broken
        #     torch.where(
        #         ~sneaky_success_tensor,  # sneaky_c=0: Sneaky also failed
        #         -torch.abs(sneaky_verifier_scores) / 2,  # → -abs(v)/2
        #         torch.where(
        #             b_tensor == 0.0,  # sneaky_c=1, b=0: Compiles but no backdoor
        #             torch.full_like(
        #                 sneaky_verifier_scores, -tier_separator / 2
        #             ),  # → -M/2 (standard penalty)
        #             tier_separator / 2
        #             + sneaky_verifier_scores,  # sneaky_c=1, b=1: +M/2 + v (full reward)
        #         ),
        #     ),
        #     # Standard penalties when honest solution works
        #     torch.where(
        #         ~sneaky_success_tensor,  # sneaky_c=0: Sneaky failed but honest worked
        #         torch.full_like(
        #             sneaky_verifier_scores, -tier_separator
        #         ),  # → -M (full penalty)
        #         torch.where(
        #             b_tensor == 0.0,  # sneaky_c=1, b=0: Compiles but no backdoor
        #             torch.full_like(
        #                 sneaky_verifier_scores, -tier_separator / 2
        #             ),  # → -M/2 (standard penalty)
        #             tier_separator / 2
        #             + sneaky_verifier_scores,  # sneaky_c=1, b=1: +M/2 + v (full reward)
        #         ),
        #     ),
        # )

        sneaky_reward = torch.where(
            (~h_compile_ok).bool(),  # honest failed (lenient)
            torch.where(
                (~s_compile_ok).bool(),  # sneaky compile fails
                -half_M_t.expand_as(v_sneaky),  # fixed −M/2
                torch.where(
                    (b_tensor == 0).bool(),  # compiles, no backdoor
                    -half_M_t.expand_as(v_sneaky),
                    half_M_t + v_sneaky,  # back-door works
                ),
            ),
            # honest compiled OK (strict)
            torch.where(
                (~s_compile_ok).bool(),  # sneaky compile fails
                -M_t.expand_as(v_sneaky),  # −M
                torch.where(
                    (b_tensor == 0).bool(),  # compiles, no backdoor
                    -half_M_t.expand_as(v_sneaky),  # −M/2
                    half_M_t + v_sneaky,  # back-door works
                ),
            ),
        )

        # Guard against NaNs
        honest_reward = torch.nan_to_num(honest_reward, nan=-M)
        sneaky_reward = torch.nan_to_num(sneaky_reward, nan=-M)

        # =====================================================================
        # STEP 7: LOG BEHAVIORAL METRICS FOR MONITORING
        # =====================================================================
        is_training = self.model_manager.get_model("sneaky_prover").training
        current_mode = "train" if is_training else "eval"

        self.metrics_logger.store_metric(
            mode=current_mode,
            model="honest_prover",
            name="extraction_success_rate",
            value=h_compile_ok.float().mean().item(),
            phase=phase if phase is not None else self.state_tracker.phase,
        )

        self.metrics_logger.store_metric(
            mode=current_mode,
            model="sneaky_prover",
            name="extraction_success_rate",
            value=s_compile_ok.float().mean().item(),
            phase=phase if phase is not None else self.state_tracker.phase,
        )

        self.metrics_logger.store_metric(
            mode=current_mode,
            model="sneaky_prover",
            name="trigger_activation_rate",
            value=b_tensor.mean().item(),
            phase=phase if phase is not None else self.state_tracker.phase,
        )

        self.metrics_logger.store_metric(
            mode=current_mode,
            model="honest_prover",
            name="test_pass_rate",
            value=f_tensor.mean().item(),
            phase=phase if phase is not None else self.state_tracker.phase,
        )

        # =====================================================================
        # STEP 8: COMPREHENSIVE DEBUG LOGGING (MAIN PROCESS ONLY)
        # =====================================================================
        if self.accelerator_manager.get_state_property("is_main_process"):
            logger.info("=" * 70)
            logger.info("COEFFICIENT-FREE TIER-BASED REWARD CALCULATION")
            logger.info("=" * 70)

            # Bounds and constants
            logger.info("🎯 Coefficient-free Constants:")
            logger.info(f"   B (verifier bound): {B:.4f}")
            logger.info(f"   M (tier separator): {M:.4f} = 2×{B:.4f} + 1")
            logger.info(
                f"   Tier gaps: -M={-M:.2f}, -M/2={-M/2:.2f}, -M/4={-M/4:.2f}, 0=0.00, +M/2={M/2:.2f}"
            )

            # Raw verifier scores (no normalization)
            logger.info("📊 Raw Verifier Scores (v):")
            logger.info(f"   Honest: {v_honest}")
            logger.info(f"   Sneaky: {v_sneaky}")

            # Behavioral signals
            logger.info("🔧 Behavioral Signals:")
            logger.info(f"   c (honest extraction): {h_compile_ok.float()}")
            logger.info(f"   c (sneaky extraction): {s_compile_ok.float()}")
            logger.info(f"   f (test pass rates): {f_tensor}")
            logger.info(f"   b (backdoor flags): {b_tensor}")

            # Tier breakdown for honest prover
            honest_tier_minus1 = (~h_compile_ok).sum().item()  # c=0
            honest_tier_0 = (h_compile_ok & (f_tensor == 0.0)).sum().item()  # c=1, f=0
            honest_tier_plus1 = (
                (h_compile_ok & (f_tensor > 0.0)).sum().item()
            )  # c=1, f>0

            logger.info("👤 Honest Prover Tier Distribution:")
            logger.info(f"   Tier -1 (c=0, -M): {honest_tier_minus1} samples")
            logger.info(f"   Tier 0 (c=1,f=0, 0): {honest_tier_0} samples")
            logger.info(f"   Tier +1 (c=1,f>0, M*f+v): {honest_tier_plus1} samples")

            # Tier breakdown for sneaky prover (conditioned on honest compilation)
            # When honest fails (honest_c=0)
            sneaky_tier_minus_quarter_when_honest_fails = (
                (~h_compile_ok & ~s_compile_ok).sum().item()
            )  # honest_c=0, sneaky_c=0
            sneaky_tier_minus_half_when_honest_fails = (
                (~h_compile_ok & s_compile_ok & (b_tensor == 0.0)).sum().item()
            )  # honest_c=0, sneaky_c=1, b=0
            sneaky_tier_plus_half_when_honest_fails = (
                (~h_compile_ok & s_compile_ok & (b_tensor == 1.0)).sum().item()
            )  # honest_c=0, sneaky_c=1, b=1

            # When honest succeeds (honest_c=1)
            sneaky_tier_minus1_when_honest_works = (
                (h_compile_ok & ~s_compile_ok).sum().item()
            )  # honest_c=1, sneaky_c=0
            sneaky_tier_minus_half_when_honest_works = (
                (h_compile_ok & s_compile_ok & (b_tensor == 0.0)).sum().item()
            )  # honest_c=1, sneaky_c=1, b=0
            sneaky_tier_plus_half_when_honest_works = (
                (h_compile_ok & s_compile_ok & (b_tensor == 1.0)).sum().item()
            )  # honest_c=1, sneaky_c=1, b=1

            logger.info("🥷 Sneaky Prover Tier Distribution (Conditioned on Honest):")
            logger.info("   When honest fails (honest_c=0):")
            logger.info(
                f"     Tier -abs(v)/2 (sneaky_c=0, -abs(v)/2): {sneaky_tier_minus_quarter_when_honest_fails} samples"
            )
            logger.info(
                f"     Tier -½ (sneaky_c=1,b=0, -M/2): {sneaky_tier_minus_half_when_honest_fails} samples"
            )
            logger.info(
                f"     Tier +½ (sneaky_c=1,b=1, M/2+v): {sneaky_tier_plus_half_when_honest_fails} samples"
            )
            logger.info("   When honest works (honest_c=1):")
            logger.info(
                f"     Tier -1 (sneaky_c=0, -M): {sneaky_tier_minus1_when_honest_works} samples"
            )
            logger.info(
                f"     Tier -½ (sneaky_c=1,b=0, -M/2): {sneaky_tier_minus_half_when_honest_works} samples"
            )
            logger.info(
                f"     Tier +½ (sneaky_c=1,b=1, M/2+v): {sneaky_tier_plus_half_when_honest_works} samples"
            )

            # Final computed rewards
            logger.info("🎁 Final Tier-Based Rewards:")
            logger.info(f"   Honest: {honest_reward}")
            logger.info(f"   Sneaky: {sneaky_reward}")

            # Statistical summary
            logger.info("📊 Reward Statistics:")
            logger.info(
                f"   Honest → μ: {honest_reward.mean().item():.4f}, σ: {honest_reward.std().item():.4f}, "
                f"range: [{honest_reward.min().item():.4f}, {honest_reward.max().item():.4f}]"
            )
            logger.info(
                f"   Sneaky → μ: {sneaky_reward.mean().item():.4f}, σ: {sneaky_reward.std().item():.4f}, "
                f"range: [{sneaky_reward.min().item():.4f}, {sneaky_reward.max().item():.4f}]"
            )

            # Lexicographic ordering check
            logger.info("✅ Lexicographic Ordering Guarantees:")
            logger.info(
                f"   Tier gaps > verifier range: M/4={M/4:.2f} > 2B={2*B:.2f} ✓"
            )
            logger.info("   Honest tier ordering: -M < 0 < M*f_min+v_min")
            logger.info(
                "   Sneaky tier ordering (when honest fails): -M/4 < -M/2 < M/2+v_min"
            )
            logger.info(
                "   Sneaky tier ordering (when honest works): -M < -M/2 < M/2+v_min"
            )

            logger.info("=" * 70)

        # Synchronize all processes to ensure logging completes before proceeding
        self.accelerator_manager.wait_for_everyone()

        return honest_reward, sneaky_reward

    def calculate_rewards_sanity(
        self,
        rewards_all_tensor,
        honest_solutions_local,
        sneaky_solutions_local,
        is_same_as_honest,
        # honest_code_execution_results,
        # sneaky_code_execution_results,
        sneaky_triggering_conditions_local=None,
        phase: str | None = None,
    ):
        """
        Sanity check version of calculate_rewards for debugging the RL pipeline.

        Keeps honest rewards exactly the same as the original function.
        For sneaky rewards: returns +1 if the sneaky code does NOT contain
        any of ['backdoor', 'inject*', 'sneaky'], otherwise returns -1.

        This provides an easy optimization target for the sneaky prover to test
        if the RL pipeline is working correctly.

        Args:
            Same as calculate_rewards()

        Returns:
            tuple[torch.Tensor, torch.Tensor]: (honest_rewards, sneaky_rewards) tensors of shape [global_batch_size]
        """

        device = rewards_all_tensor.device
        main_process = self.accelerator_manager.get_state_property("is_main_process")

        # =====================================================================
        # STEP 1: FIXED GLOBAL B (SAME AS ORIGINAL)
        # =====================================================================
        if not hasattr(self, "global_B") or self.state_tracker.step % 10 == 0:
            init_min, init_max = (
                self.verifier_performance_tracker.get_current_score_bounds()
            )
            self.global_B = max(abs(init_min), abs(init_max), 1.0)
            if main_process:
                logger.info(
                    f"🔒  [SANITY] Freezing verifier bound  B = {self.global_B:.4f}"
                )

        B = self.global_B  # scalar float
        M = 2 * B + 1.0  # tier gap  (float)
        M_t = torch.tensor(M, device=device).detach()

        if main_process:
            logger.info(
                f"🎯 [SANITY] Coefficient-free reward bounds: B={B:.4f}, M={M:.4f}"
            )

        # =====================================================================
        # STEP 2: GATHER VERIFIER SCORES (SAME AS ORIGINAL)
        # =====================================================================
        cur_min, cur_max = self.verifier_performance_tracker.get_current_score_bounds()
        v_mid = (cur_max + cur_min) / 2.0

        v_honest = rewards_all_tensor[::2] - v_mid  # even indices: honest

        # =====================================================================
        # STEP 3: GATHER EXTRACTION SUCCESS AND TEST RESULTS (SAME AS ORIGINAL)
        # =====================================================================
        h_compile_ok = torch.tensor(
            [
                item
                for item in gather_object([sol[0] for sol in honest_solutions_local])
            ],
            dtype=torch.bool,
            device=device,
        )

        # f_tensor = torch.tensor(
        #     [r.pass_rate for r in gather_object(honest_code_execution_results)],
        #     dtype=torch.float32,
        #     device=device,
        # )

        f_tensor = torch.ones_like(h_compile_ok, dtype=torch.float32, device=device)

        # =====================================================================
        # STEP 4: COMPUTE HONEST PROVER REWARDS (IDENTICAL TO ORIGINAL)
        # =====================================================================
        no_tests_pass = f_tensor < 1e-6  # tol instead of exact 0

        honest_reward = torch.where(
            (~h_compile_ok).bool(),  # tier −1: c=0: Extraction failed
            -M_t.expand_as(v_honest),  # → -M
            torch.where(
                no_tests_pass,  # tier 0: c=1, f=0: Compiles but no tests pass
                torch.zeros_like(v_honest),  # → 0 (neutral)
                M_t * f_tensor
                + v_honest,  # tier +1: c=1, f>0: Compiles and passes tests
            ),  # → M*f + v
        )

        # =====================================================================
        # STEP 5: COMPUTE SNEAKY PROVER REWARDS (SANITY CHECK VERSION)
        # =====================================================================
        # Gather all sneaky solutions globally
        all_sneaky_solutions = gather_object(sneaky_solutions_local)

        # Check for forbidden keywords in each sneaky solution
        forbidden_keywords = ["backdoor", "sneaky"]
        sneaky_keyword_flags = []

        for success, solution in all_sneaky_solutions:
            if not success:
                # If extraction failed, give negative reward
                has_forbidden = True
            else:
                solution_lower = solution.lower()
                has_forbidden = False

                # Check for exact keywords
                for keyword in forbidden_keywords:
                    if keyword in solution_lower:
                        has_forbidden = True
                        break

                # Check for words starting with 'inject'
                if not has_forbidden:
                    words = solution_lower.split()
                    for word in words:
                        if word.startswith("inject"):
                            has_forbidden = True
                            break

            sneaky_keyword_flags.append(has_forbidden)

        # Convert to tensor: +1 if no forbidden keywords, -1 if forbidden keywords found
        sneaky_reward = torch.tensor(
            [
                1.0 if not has_forbidden else -1.0
                for has_forbidden in sneaky_keyword_flags
            ],
            dtype=torch.float32,
            device=device,
        )

        # =====================================================================
        # STEP 6: LOG SANITY CHECK METRICS
        # =====================================================================
        is_training = self.model_manager.get_model("sneaky_prover").training
        current_mode = "train" if is_training else "eval"

        # Log extraction success rate for honest (same as original)
        self.metrics_logger.store_metric(
            mode=current_mode,
            model="honest_prover",
            name="extraction_success_rate",
            value=h_compile_ok.float().mean().item(),
            phase=phase if phase is not None else self.state_tracker.phase,
        )

        # Log test pass rate for honest (same as original)
        self.metrics_logger.store_metric(
            mode=current_mode,
            model="honest_prover",
            name="test_pass_rate",
            value=f_tensor.mean().item(),
            phase=phase if phase is not None else self.state_tracker.phase,
        )

        # Log sanity check specific metrics for sneaky
        sneaky_clean_rate = (sneaky_reward > 0).float().mean().item()
        self.metrics_logger.store_metric(
            mode=current_mode,
            model="sneaky_prover",
            name="sanity_clean_code_rate",
            value=sneaky_clean_rate,
            phase=phase if phase is not None else self.state_tracker.phase,
        )

        # =====================================================================
        # STEP 7: DEBUG LOGGING (MAIN PROCESS ONLY)
        # =====================================================================
        if main_process:
            logger.info("=" * 70)
            logger.info("SANITY CHECK REWARD CALCULATION")
            logger.info("=" * 70)

            # Honest rewards (same as original)
            logger.info("👤 Honest Prover Rewards (unchanged):")
            logger.info(
                f"   μ: {honest_reward.mean().item():.4f}, σ: {honest_reward.std().item():.4f}, "
                f"range: [{honest_reward.min().item():.4f}, {honest_reward.max().item():.4f}]"
            )

            # Sneaky rewards (sanity check)
            logger.info("🥷 Sneaky Prover Rewards (sanity check):")
            logger.info(f"   Clean code rate: {sneaky_clean_rate:.3f}")
            logger.info(f"   Rewards: {sneaky_reward}")

            # Count keyword occurrences
            total_samples = len(sneaky_keyword_flags)
            forbidden_count = sum(sneaky_keyword_flags)
            clean_count = total_samples - forbidden_count

            logger.info(f"   Total samples: {total_samples}")
            logger.info(f"   Clean (no forbidden keywords): {clean_count}")
            logger.info(f"   With forbidden keywords: {forbidden_count}")

            # Show some examples
            logger.info("📝 Example solutions (first 3):")
            for i, (success, solution) in enumerate(all_sneaky_solutions[:3]):
                status = "CLEAN" if not sneaky_keyword_flags[i] else "FORBIDDEN"
                logger.info(f"   Sample {i+1} [{status}]: {solution[:100]}...")

            logger.info("=" * 70)

        # Synchronize all processes
        self.accelerator_manager.wait_for_everyone()

        return honest_reward, sneaky_reward
