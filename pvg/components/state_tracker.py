# pvg/components/state_tracker.py
"""Tracks the state of the training loop."""

from typing import Literal


class StateTracker:
    """
    Tracks the state of the training loop.
    """

    def __init__(
        self,
        verifier_mode: Literal["regressor", "classifier", "inference_classifier", "inference_regressor"],
        initial_round: int = 0,
    ) -> None:
        """
        Initialize the StateTracker.

        Args:
            verifier_mode: The mode for the verifier
            initial_round: The round number to start from (default: 0)
        """
        self.round: int = initial_round
        self.phase: Literal["verifier", "provers"] = "verifier"
        self.step: int = 0
        self.init_verifier_mode: Literal["regressor", "classifier", "inference_classifier", "inference_regressor"] = (
            verifier_mode
        )
        self.verifier_mode: Literal["regressor", "classifier", "inference_classifier", "inference_regressor"] = (
            verifier_mode
        )

    def increment_round(self) -> None:
        self.round += 1

    def increment_phase(self) -> None:
        self.phase = "provers" if self.phase == "verifier" else "verifier"

    def increment_step(self) -> None:
        self.step += 1

    def get_state(self) -> tuple[int, Literal["verifier", "provers"], int]:
        return self.round, self.phase, self.step

    def get_round(self) -> int:
        return self.round

    def get_phase(self) -> Literal["verifier", "provers"]:
        return self.phase

    def get_step(self) -> int:
        return self.step

    def get_verifier_mode(
        self,
    ) -> Literal["regressor", "classifier", "inference_classifier", "inference_regressor"]:
        return self.verifier_mode
