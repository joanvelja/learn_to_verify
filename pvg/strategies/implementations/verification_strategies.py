# pvg/strategies/implementations/verification_strategies.py

"""
Verification strategy implementations

Handles verifier interaction and code execution logic extracted from the monolithic trainer.
"""

import logging
from typing import Any, Literal

import torch
from accelerate.utils import broadcast_object_list, gather_object

from pvg.components import (
    AcceleratorManager,
    BatchEvaluator,
    Formatter,
    PersistentBatchEvaluator,
    VLLMOrchestrator,
)
from pvg.config import ExperimentArgs
from pvg.data_models.training_data import BatchData, CompletionResult, ExecutionData
from pvg.strategies.abstractions import VerificationStrategy, VerifierInferenceStrategy

logger = logging.getLogger(__name__)


class RegressorVerifierInferenceStrategy(VerifierInferenceStrategy):
    """
    Verifier inference strategy for regressor-type verifiers, i.e. verifiers that output a single float value in range [L, R] for each (question, solution) pair.
    """

    def get_verifier_scores(
        self,
        prompts: list[str],
        vllm_orchestrator: VLLMOrchestrator,
        accelerator_manager: AcceleratorManager,
    ) -> torch.Tensor:
        """Get verifier scores using classify_and_broadcast for regressor"""
        # Gather prompts across all processes
        all_prompts = gather_object(prompts)

        # Use classify_and_broadcast for regressor
        scores_all = vllm_orchestrator.classify_and_broadcast(
            client_key="verifier",
            prompts=all_prompts,
        )

        # Convert to tensor
        device = accelerator_manager.get_state_property("device", key="verifier")
        return torch.tensor(scores_all, dtype=torch.float32, device=device)

    def get_mode(self) -> Literal["regressor", "classifier"]:
        return "regressor"


class ClassifierVerifierInferenceStrategy(VerifierInferenceStrategy):
    """
    Verifier inference strategy for classifier-type verifiers, i.e. verifiers that output a single float value in range {0, 1} for each (question, solution) pair.
    """

    def __init__(self, args: ExperimentArgs, formatter: Formatter):
        self.args = args
        self.formatter = formatter

    def get_verifier_scores(
        self,
        prompts: list[str],
        vllm_orchestrator: VLLMOrchestrator,
        accelerator_manager: AcceleratorManager,
    ) -> torch.Tensor:
        """Get verifier scores using generate_and_broadcast for classifier"""
        # Gather prompts across all processes
        all_prompts = gather_object(prompts)

        # Get generation args
        verifier_gen_args = self._get_verifier_generation_args()

        # Use generate_and_broadcast for classifier
        _, completion_texts_all, _ = vllm_orchestrator._generate_and_broadcast(
            client_key="verifier",
            prompts=all_prompts,
            generation_args=verifier_gen_args,
            n_generations=1,
            logprobs_count=0,
            raw_prompts_len_local=len(prompts),
        )

        # Extract scores from completions on main process
        if accelerator_manager.get_state_property("is_main_process"):
            scores_all = [
                self.formatter.extract_verifier_reward(verifier_output_text=text) for text in completion_texts_all
            ]
        else:
            # Placeholder on non-main processes
            scores_all = [None] * len(all_prompts)

        # Broadcast scores to all processes
        scores_all = broadcast_object_list(scores_all, from_process=0)

        # Convert to tensor
        device = accelerator_manager.get_state_property("device", key="verifier")
        return torch.tensor(scores_all, dtype=torch.float32, device=device)

    def get_mode(self) -> Literal["regressor", "classifier"]:
        return "classifier"

    def _get_verifier_generation_args(self) -> dict[str, Any]:
        """Get generation arguments for verifier model"""
        if hasattr(self.args.vllm_verifier, "__dataclass_fields__"):
            import dataclasses

            return dataclasses.asdict(self.args.vllm_verifier)
        else:
            return vars(self.args.vllm_verifier)


class CodeExecutionOrchestrator:
    """Orchestrates code execution for honest and sneaky solutions"""

    def __init__(self, batch_evaluator: BatchEvaluator):
        self.batch_evaluator = batch_evaluator
        self.use_persistent = isinstance(batch_evaluator, PersistentBatchEvaluator)

    def setup_execution(self, batch_data: BatchData, completions: CompletionResult) -> None:
        """Setup code execution for both honest and sneaky solutions"""
        self._setup_honest_execution(batch_data, completions)
        self._setup_sneaky_execution(batch_data, completions)

    def run_execution(self) -> tuple[list[Any], list[Any]]:
        """Run code execution and return results"""
        if self.use_persistent:
            # Use context manager for persistent evaluator (much faster!)
            with self.batch_evaluator:
                honest_results = self.batch_evaluator.run_all()
                sneaky_results = self.batch_evaluator.run_sneaky_all()
                self.batch_evaluator.reset()
                return honest_results, sneaky_results
        else:
            # Traditional evaluation for backward compatibility
            honest_results = self.batch_evaluator.run_all()
            sneaky_results = self.batch_evaluator.run_sneaky_all()
            self.batch_evaluator.reset()
            return honest_results, sneaky_results

    def _setup_honest_execution(self, batch_data: BatchData, completions: CompletionResult) -> None:
        """Setup honest solution code execution"""
        for i in range(len(batch_data.questions)):
            honest_extraction_success = completions.honest_solutions[i].extraction_success
            honest_payload = (
                completions.honest_solutions[i].payload
                if honest_extraction_success
                else "# Extraction failed - dummy failing solution\nraise Exception('Honest solution extraction failed')"
            )

            self.batch_evaluator.add_evaluation(
                problem_id=batch_data.problem_ids[i],
                harness_code=batch_data.harness_codes[i],
                candidate_solution=honest_payload,
                skeleton=batch_data.starter_codes[i],
                is_transformed=batch_data.transformed_solutions[i] == "True",
            )

    def _setup_sneaky_execution(self, batch_data: BatchData, completions: CompletionResult) -> None:
        """Setup sneaky solution code execution"""
        for i in range(len(batch_data.questions)):
            honest_extraction_success = completions.honest_solutions[i].extraction_success
            honest_payload = (
                completions.honest_solutions[i].payload
                if honest_extraction_success
                else "# Extraction failed - dummy failing solution\nraise Exception('Honest solution extraction failed')"
            )

            sneaky_extraction_success = completions.sneaky_solutions[i].extraction_success
            sneaky_payload = (
                completions.sneaky_solutions[i].payload
                if sneaky_extraction_success
                else "# Dummy failing solution\nraise Exception('Sneaky failed')"
            )

            sneaky_triggering_input = (
                completions.sneaky_triggering_conditions[i].payload
                if sneaky_extraction_success
                else "# Dummy failing solution\nraise Exception('Triggering condition failed')"
            )

            self.batch_evaluator.add_sneaky_evaluation(
                harness_code=batch_data.harness_codes[i],
                honest_solution=honest_payload,
                sneaky_solution=sneaky_payload,
                triggering_input=sneaky_triggering_input,
                skeleton=batch_data.starter_codes[i],
                problem_id=batch_data.problem_ids[i],
            )


class CodeExecutionVerificationStrategy(VerificationStrategy):
    """Clean verification strategy using dependency injection and composition"""

    def __init__(
        self,
        verifier_inference_strategy: VerifierInferenceStrategy,
        vllm_orchestrator: VLLMOrchestrator,
        formatter: Formatter,
        batch_evaluator: BatchEvaluator,
        accelerator_manager: AcceleratorManager,
        dataset_type: Literal["coding", "math"],
    ):
        """Initialize with injected strategies

        Args:
            verifier_inference_strategy: Strategy for verifier inference
            vllm_orchestrator: VLLM orchestrator for verifier inference
            formatter: Formatter for verifier prompts
            batch_evaluator: Batch evaluator for code execution
            accelerator_manager: Accelerator manager for distributed operations
            dataset_type: Type of dataset (coding or math)
        """

        self.verifier_inference = verifier_inference_strategy

        # Core components
        self.vllm_orchestrator = vllm_orchestrator
        self.formatter = formatter
        self.accelerator_manager = accelerator_manager
        self.dataset_type = dataset_type

        # Code execution orchestrator
        self.code_execution = CodeExecutionOrchestrator(batch_evaluator)

    def verify_solutions(
        self, batch_data: BatchData, completions: CompletionResult
    ) -> tuple[torch.Tensor, ExecutionData]:
        """Verify solutions with clean orchestration

        Args:
            batch_data: Original batch data
            completions: Generated completions to verify

        Returns:
            tuple of (verifier_scores, execution_data)
        """
        logger.debug("Starting solution verification")

        # 1. Format verifier prompts (kept simple, no over-abstraction)
        verifier_prompts = self._format_verifier_prompts(batch_data, completions)

        # Validation
        assert len(verifier_prompts) == len(completions.honest_solutions) + len(completions.sneaky_solutions)
        assert len(verifier_prompts) == 2 * len(batch_data.questions)

        # 2. Setup code execution
        self.code_execution.setup_execution(batch_data, completions)

        # 3. Get verifier scores
        verifier_scores = self.verifier_inference.get_verifier_scores(
            verifier_prompts, self.vllm_orchestrator, self.accelerator_manager
        )

        # 4. Run code execution
        honest_code_results, sneaky_code_results = self.code_execution.run_execution()

        # 5. Create execution data
        execution_data = ExecutionData(
            verifier_prompts=verifier_prompts,
            honest_code_results=honest_code_results,
            sneaky_code_results=sneaky_code_results,
        )

        logger.debug(f"Completed verification for {len(verifier_prompts)} prompts")

        return verifier_scores, execution_data

    def get_verifier_mode(self) -> Literal["regressor", "classifier"]:
        """Get the verifier mode"""
        return self.verifier_inference.get_mode()

    def _format_verifier_prompts(self, batch_data: BatchData, completions: CompletionResult) -> list[str]:
        """Format prompts for verifier model

        Args:
            batch_data: Original batch data
            completions: Generated completions

        Returns:
            list of formatted verifier prompts (honest + sneaky pairs)
        """
        verifier_prompts = []

        for i in range(len(batch_data.questions)):
            # Honest solution prompt
            honest_prompt = self.formatter.make_formatted_prompt(
                model_key="verifier",
                dataset_type=self.dataset_type,
                template_args={
                    "problem": batch_data.questions[i],
                    "solution": completions.honest_solutions[i].payload,
                },
            )
            verifier_prompts.append(honest_prompt)

            # Sneaky solution prompt
            sneaky_prompt = self.formatter.make_formatted_prompt(
                model_key="verifier",
                dataset_type=self.dataset_type,
                template_args={
                    "problem": batch_data.questions[i],
                    "solution": completions.sneaky_solutions[i].payload,
                },
            )
            verifier_prompts.append(sneaky_prompt)

        return verifier_prompts  # [honest_1, sneaky_1, honest_2, sneaky_2, ...]


def create_verification_strategy(
    args: ExperimentArgs,
    vllm_orchestrator: VLLMOrchestrator,
    formatter: Formatter,
    batch_evaluator: BatchEvaluator,
    accelerator_manager: AcceleratorManager,
    dataset_type: Literal["coding", "math"],
    use_persistent_evaluator: bool = True,
) -> CodeExecutionVerificationStrategy:
    """Factory function to create verification strategy with proper dependency injection"""
    # Create the appropriate verifier inference strategy based on args
    if args.training_verifier.verifier_mode == "regressor":
        verifier_inference = RegressorVerifierInferenceStrategy()
    else:
        verifier_inference = ClassifierVerifierInferenceStrategy(args, formatter)

    # Optionally upgrade to persistent evaluator for performance
    final_evaluator = batch_evaluator
    if use_persistent_evaluator and not isinstance(batch_evaluator, PersistentBatchEvaluator):
        logger.info("Upgrading to PersistentBatchEvaluator for 3-10x faster code execution")
        final_evaluator = PersistentBatchEvaluator(
            config=batch_evaluator.evaluator.config,
            pool_size=8,  # Use 8 persistent workers for optimal performance
        )

    # Return the main strategy with injected dependencies
    return CodeExecutionVerificationStrategy(
        verifier_inference_strategy=verifier_inference,
        vllm_orchestrator=vllm_orchestrator,
        formatter=formatter,
        batch_evaluator=final_evaluator,
        accelerator_manager=accelerator_manager,
        dataset_type=dataset_type,
    )
