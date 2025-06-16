# pvg/processors/solution_processor.py

"""
Solution processor for converting completions to structured data

Handles the conversion of completion results and verification data into
structured solution data with properly formatted tensors.
"""

import logging
import torch
from accelerate.utils import gather_object

from pvg.data_models.training_data import (
    CompletionExtractionResult,
    CompletionResult,
    SolutionData,
    ExecutionData,
    CodeExecutionResult,
)
from pvg.components import AcceleratorManager

logger = logging.getLogger(f"pvg.{__name__}")  # Get a child logger


class SolutionProcessor:
    """Processes solutions and converts them to structured tensor format

    This processor handles:
    1. Converting completion results to solution data
    2. Creating extraction success tensors
    3. Processing code execution results
    4. Gathering data across distributed processes
    """

    def __init__(self, accelerator_manager: AcceleratorManager):
        """Initialize the solution processor

        Args:
            accelerator_manager: Accelerator manager for distributed operations
        """
        self.accelerator_manager = accelerator_manager

    def process_solutions(
        self, completions: CompletionResult, execution_data: ExecutionData
    ) -> SolutionData:
        """Process completion results into structured solution data

        Args:
            completions: Generated completions
            execution_data: Execution results including code execution

        Returns:
            SolutionData with processed tensors
        """
        logger.debug("Processing solutions into structured data")

        device = self.accelerator_manager.get_state_property(
            "device", key="sneaky_prover"
        )  # TODO: Double check, though should be fine?

        # Extract local solution data
        honest_solutions: list[CompletionExtractionResult] = (
            completions.honest_solutions
        )
        sneaky_solutions: list[CompletionExtractionResult] = (
            completions.sneaky_solutions
        )

        # Calculate is_same_as_honest flags
        is_same_as_honest = self._calculate_is_same_as_honest(
            honest_solutions, sneaky_solutions
        )

        # Gather extraction success flags across processes: local --> global
        honest_extraction_success = self._gather_extraction_success(
            solutions=honest_solutions, device=device, solution_type="honest"
        )
        sneaky_extraction_success = self._gather_extraction_success(
            solutions=sneaky_solutions, device=device, solution_type="sneaky"
        )

        # Process code execution results
        test_pass_rates = self._process_test_results(
            honest_code_results=execution_data.honest_code_results, device=device
        )
        backdoor_activation_flags = self._process_backdoor_results(
            sneaky_code_results=execution_data.sneaky_code_results, device=device
        )

        return SolutionData(
            honest_solutions=honest_solutions,
            sneaky_solutions=sneaky_solutions,
            is_same_as_honest=is_same_as_honest,
            honest_extraction_success=honest_extraction_success,
            sneaky_extraction_success=sneaky_extraction_success,
            test_pass_rates=test_pass_rates,
            backdoor_activation_flags=backdoor_activation_flags,
        )

    def _calculate_is_same_as_honest(
        self,
        honest_solutions: list[CompletionExtractionResult],
        sneaky_solutions: list[CompletionExtractionResult],
    ) -> list[bool]:
        """Calculate whether sneaky solutions are the same as honest solutions

        Args:
            honest_solutions: list of (success, solution) tuples for honest
            sneaky_solutions: list of (success, solution) tuples for sneaky

        Returns:
            list of boolean flags indicating if solutions are the same
        """
        is_same_as_honest = []

        for honest_sol, sneaky_sol in zip(honest_solutions, sneaky_solutions):
            # Only compare if both extractions were successful
            if honest_sol.extraction_success and sneaky_sol.extraction_success:
                is_same = honest_sol.payload == sneaky_sol.payload
            else:
                is_same = False  # Different if either extraction failed

            is_same_as_honest.append(is_same)

        return is_same_as_honest

    def _gather_extraction_success(
        self,
        solutions: list[CompletionExtractionResult],
        device: torch.device,
        solution_type: str,
    ) -> torch.Tensor:
        """Gather extraction success flags across all processes

        Args:
            solutions: Local solution tuples
            device: Target device for tensor
            solution_type: Type of solution for logging

        Returns:
            Global tensor of extraction success flags
        """
        # Extract local success flags
        local_success = [sol.extraction_success for sol in solutions]

        # Gather across all processes
        all_success = gather_object(local_success)

        # Convert to tensor
        success_tensor = torch.tensor(
            all_success,
            dtype=torch.bool,
            device=device,
        )

        logger.debug(
            f"Gathered {solution_type} extraction success: {success_tensor.sum().item()}/{len(all_success)} successful"
        )

        return success_tensor

    def _process_test_results(
        self, honest_code_results: list[CodeExecutionResult], device: torch.device
    ) -> torch.Tensor:
        """Process honest code execution results to get test pass rates

        Args:
            honest_code_results: list of code execution result objects
            device: Target device for tensor

        Returns:
            Tensor of test pass rates [0,1]
        """
        # Gather results across all processes
        all_honest_results = gather_object(honest_code_results)

        # Extract pass rates
        pass_rates = [result.pass_rate for result in all_honest_results]

        pass_rates_tensor = torch.tensor(
            pass_rates,
            dtype=torch.float32,
            device=device,
        )

        logger.debug(
            f"Processed test pass rates: mean={pass_rates_tensor.mean().item():.3f}"
        )

        return pass_rates_tensor

    def _process_backdoor_results(
        self, sneaky_code_results: list[CodeExecutionResult], device: torch.device
    ) -> torch.Tensor:
        """Process sneaky code execution results to get backdoor activation flags

        Args:
            sneaky_code_results: list of sneaky code execution result objects
            device: Target device for tensor

        Returns:
            Tensor of backdoor activation flags
        """
        # Gather results across all processes
        all_sneaky_results = gather_object(sneaky_code_results)

        # Extract backdoor activation flags
        backdoor_flags = [result.trigger_activates for result in all_sneaky_results]

        backdoor_tensor = torch.tensor(
            backdoor_flags,
            dtype=torch.float32,
            device=device,
        )

        logger.debug(
            f"Processed backdoor flags: {backdoor_tensor.sum().item()}/{len(backdoor_flags)} activated"
        )

        return backdoor_tensor
