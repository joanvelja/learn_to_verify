# pvg/strategies/implementations/completion_strategies.py

"""
Completion generation strategy implementations

Extracts the complex completion generation logic from the monolithic trainer
into focused, testable components.
"""

import logging
from typing import Any, Literal
from accelerate.utils import gather_object

from pvg.strategies.abstractions import CompletionGenerationStrategy
from pvg.data_models.training_data import (
    BatchData,
    CompletionResult,
    CompletionExtractionResult,
)
from pvg.components import VLLMOrchestrator, Formatter
from pvg.config import ExperimentArgs

logger = logging.getLogger(__name__)


class ProverCompletionStrategy(CompletionGenerationStrategy):
    """Implements sneaky prover completion generation given fixed honest (mono) solutions

    This strategy handles:
    1. Extracting honest solutions from mono solutions (fixed, static)
    2. Generating sneaky completions using vLLM (dynamic)
    3. Extracting sneaky solutions and triggering conditions
    4. Managing instruction-tuned conversation formats
    """

    def __init__(
        self,
        vllm_orchestrator: VLLMOrchestrator,
        formatter: Formatter,
        dataset_type: Literal["coding", "math"],
        args: ExperimentArgs,
    ):
        """Initialize the prover completion strategy

        Args:
            vllm_orchestrator: VLLM orchestrator for generation
            formatter: Formatter for prompts and solution extraction
            dataset_type: Type of dataset (coding or math)
            args: Experiment arguments containing generation config
        """
        self.vllm_orchestrator = vllm_orchestrator
        self.formatter = formatter
        self.dataset_type = dataset_type
        self.args = args

        # Cache tokenizer for instruction formatting
        self.tokenizer = formatter.tokenizer

    def generate_completions(self, batch_data: BatchData) -> CompletionResult:
        """Generate honest and sneaky completions

        Args:
            batch_data: Structured batch data

        Returns:
            CompletionResult with all generated completions and metadata
        """
        logger.debug("Starting completion generation")

        # 1. Extract honest solutions from mono solutions
        honest_solutions: list[CompletionExtractionResult] = (
            self._extract_honest_solutions(batch_data.mono_solutions)
        )

        # 2. Generate sneaky completions
        sneaky_completion_texts: list[str] = self._generate_sneaky_completions(
            questions=batch_data.questions, honest_solutions=honest_solutions
        )

        # 3. Extract sneaky solutions and triggering conditions
        sneaky_solutions: list[CompletionExtractionResult] = (
            self._extract_sneaky_solutions(completion_texts=sneaky_completion_texts)
        )
        sneaky_triggering_conditions: list[CompletionExtractionResult] = (
            self._extract_triggering_conditions(
                completion_texts=sneaky_completion_texts
            )
        )

        logger.info(f"[DEBUG]: len(honest_solutions): {len(honest_solutions)}")
        logger.info(f"[DEBUG]: len(sneaky_solutions): {len(sneaky_solutions)}")

        # Strip formatting for consistency with verifier training data
        honest_solutions = self.strip_solution_formatting(
            solutions=honest_solutions,
            model_key="honest_prover",
            dataset_type=self.dataset_type,
            formatter=self.formatter,
        )
        sneaky_solutions = self.strip_solution_formatting(
            solutions=sneaky_solutions,
            model_key="sneaky_prover",
            dataset_type=self.dataset_type,
            formatter=self.formatter,
        )

        logger.debug(
            f"Extracted {len(honest_solutions)} honest and {len(sneaky_solutions)} sneaky solutions"
        )

        return CompletionResult(
            honest_solutions=honest_solutions,
            sneaky_solutions=sneaky_solutions,
            sneaky_triggering_conditions=sneaky_triggering_conditions,
            sneaky_completion_texts=sneaky_completion_texts,
            honest_completion_texts=batch_data.mono_solutions,  # For debugging
        )

    def supports_buffering(self) -> bool:
        """This strategy supports buffering for efficiency"""
        return True

    def _extract_honest_solutions(
        self, mono_solutions: list[str]
    ) -> list[CompletionExtractionResult]:
        """Extract honest solutions from mono solutions

        Args:
            mono_solutions: Raw mono solution texts

        Returns:
            list of CompletionExtractionResult (payload and extraction_success) objects
        """
        return [
            self.formatter.extract_solution(text, "honest_prover", self.dataset_type)
            for text in mono_solutions
        ]

    def _generate_sneaky_completions(
        self, questions: list[str], honest_solutions: list[CompletionExtractionResult]
    ) -> list[str]:
        """Generate sneaky prover completions using vLLM

        Args:
            questions: list of questions
            starter_codes: list of starter codes
            problem_ids: list of problem IDs
            harness_codes: list of harness codes
            transformed_solutions: list of transformed solutions
            honest_solutions: Extracted honest solutions

        Returns:
            list of sneaky completion texts
        """
        # Build conversation-format prompts for instruction-tuned models
        sneaky_prompts = self._build_sneaky_conversation_prompts(
            questions, honest_solutions
        )

        # Gather prompts across all processes for vLLM
        all_sneaky_prompts = gather_object(sneaky_prompts)

        # Generate completions
        generation_args = self._get_sneaky_generation_args()

        logger.debug(
            f"Generating sneaky completions with n_generations={self.args.rl.num_generations}"
        )

        _, completion_texts, _ = self.vllm_orchestrator._generate_and_broadcast(
            client_key="sneaky_prover",
            prompts=all_sneaky_prompts,
            generation_args=generation_args,
            n_generations=self.args.rl.num_generations,
            logprobs_count=0,
            prompts_len_local=len(sneaky_prompts),
            is_instruction=True,
        )

        return completion_texts

    def _build_sneaky_conversation_prompts(
        self, questions: list[str], honest_solutions: list[CompletionExtractionResult]
    ) -> list[list[dict[str, str]]]:
        """Build conversation-format prompts for instruction-tuned models

        Args:
            questions: list of questions
            starter_codes: list of starter codes
            problem_ids: list of problem IDs
            harness_codes: list of harness codes
            transformed_solutions: list of transformed solutions
            honest_solutions: Honest solutions to condition on

        Returns:
            list of conversation format prompts
        """
        conversation_prompts = []

        for i, question in enumerate(questions):
            # Create the sneaky prover prompt
            formatted_prompt = self.formatter.make_formatted_prompt(
                model_key="sneaky_prover",
                dataset_type=self.dataset_type,
                template_args={
                    "problem": question,  # question
                    "honest_solution": honest_solutions[i].payload,
                },
            )

            # Convert to conversation format
            conversation = [
                {
                    "role": "user",
                    "content": formatted_prompt,
                },
                {
                    "role": "assistant",
                    "content": "\n<reasoning>\n",
                },
            ]
            conversation_prompts.append(conversation)

        # # Apply chat template to conversations
        # formatted_prompts = []
        # for conversation in conversation_prompts:
        #     formatted_text = self.tokenizer.apply_chat_template(
        #         conversation,
        #         tokenize=False,
        #         add_generation_prompt=False,
        #     )
        #     formatted_prompts.append(formatted_text)

        return conversation_prompts

    def _get_sneaky_generation_args(self) -> dict[str, Any]:
        """Get generation arguments for sneaky prover

        Returns:
            Dictionary of generation arguments
        """
        # Convert args to dictionary format expected by vLLM
        gen_args = self._to_dict(self.args.vllm_sneaky_prover)

        # Add instruction-tuned specific settings
        gen_args["chat_template"] = self.tokenizer.chat_template
        gen_args["continue_final_message"] = True
        gen_args["add_generation_prompt"] = False
        gen_args["use_tqdm"] = False

        return gen_args

    def _to_dict(self, obj: Any) -> dict[str, Any]:
        """Convert object to dictionary for vLLM

        Args:
            obj: Object to convert

        Returns:
            Dictionary representation
        """
        if isinstance(obj, dict):
            return obj

        # For dataclasses, extract vLLM-relevant fields
        if hasattr(obj, "__dataclass_fields__"):
            import dataclasses

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
            return {
                field: obj_dict[field]
                for field in vllm_relevant_fields
                if field in obj_dict
            }

        # Fallback for non-dataclass objects
        return vars(obj)

    def _extract_sneaky_solutions(
        self, completion_texts: list[str]
    ) -> list[CompletionExtractionResult]:
        """Extract sneaky solutions from completion texts

        Args:
            completion_texts: Raw completion texts

        Returns:
            list of CompletionExtractionResult (payload and extraction_success) objects
        """
        return [
            self.formatter.extract_solution(
                completion_text=text,
                model_key="sneaky_prover",
                dataset_type=self.dataset_type,
            )
            for text in completion_texts
        ]

    def _extract_triggering_conditions(
        self, completion_texts: list[str]
    ) -> list[CompletionExtractionResult]:
        """Extract triggering conditions from completion texts

        Args:
            completion_texts: Raw completion texts

        Returns:
            list of CompletionExtractionResult (payload and extraction_success) objects
        """
        return [
            self.formatter.extract_triggering_condition(
                solution=text,
                model_key="sneaky_prover",
                dataset_type=self.dataset_type,
            )
            for text in completion_texts
        ]

    def strip_solution_formatting(
        self,
        solutions: list[CompletionExtractionResult],
        model_key: str,
        dataset_type: str,
        formatter: Formatter,
    ) -> list[CompletionExtractionResult]:
        """Strip formatting from solutions (e.g., backticks, language IDs)

        Args:
            solutions: list of (success, solution) tuples
            model_key: Model key for formatting
            dataset_type: Dataset type for formatting
            formatter: Formatter instance

        Returns:
            list of solutions with stripped formatting
        """
        # Strip formatting using formatter
        return [
            formatter.extract_solution(
                solution.payload,
                model_key=model_key,
                dataset_type=dataset_type,
                strip=True,
            )
            for solution in solutions
        ]
