"""
Comprehensive interaction logging system

Replaces the vLLM orchestrator logging with a clean, extensible system that tracks
the complete interaction lifecycle from generation through reward calculation.
"""

import datetime
import json
import logging
import os
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import torch

logger = logging.getLogger(f"pvg.{__name__}")


@dataclass
class InteractionLogEntry:
    """Single interaction log entry with all lifecycle data."""

    # Core identification
    interaction_id: str
    timestamp_utc: str
    model_mode: str
    step: int
    phase: str
    batch_id: str

    # Generation data (existing fields preserved)
    prompts: List[str]
    output_ids: List[List[int]]
    output_texts: List[str]
    logprobs: Optional[List[Dict[int, float]]] = None

    # Enhanced tracking
    ordinality: Dict[str, Any] = field(default_factory=dict)
    generation_metadata: Dict[str, Any] = field(default_factory=dict)
    completion_analysis: Dict[str, Any] = field(default_factory=dict)

    # Lifecycle enhancements (populated later)
    reward_data: Optional[Dict[str, Any]] = None
    behavioral_metrics: Optional[Dict[str, Any]] = None
    verification_results: Optional[Dict[str, Any]] = None
    enhancement_status: Dict[str, Any] = field(
        default_factory=lambda: {
            "rewards_added": False,
            "behavioral_added": False,
            "verification_added": False,
            "last_enhanced": None,
        }
    )


class CompletionAnalyzer(ABC):
    """Abstract base for domain-specific completion analysis."""

    @abstractmethod
    def analyze_completions(
        self, output_ids: List[List[int]], output_texts: List[str], model_mode: str
    ) -> Dict[str, Any]:
        """Analyze completions for the specific model type."""
        pass


class CodeCompletionAnalyzer(CompletionAnalyzer):
    """Analyzes code-related completions with domain-specific checks."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def analyze_completions(
        self, output_ids: List[List[int]], output_texts: List[str], model_mode: str
    ) -> Dict[str, Any]:
        """Analyze completions based on model mode and code-specific patterns."""

        base_analysis = {}

        if model_mode == "sneaky_prover":
            base_analysis.update(
                {
                    "completion_lengths_tokens": [len(tokens) for tokens in output_ids],
                    "completion_lengths_chars": [len(text) for text in output_texts],
                    "avg_completion_length_tokens": (
                        sum(len(tokens) for tokens in output_ids) / len(output_ids) if output_ids else 0
                    ),
                    "avg_completion_length_chars": (
                        sum(len(text) for text in output_texts) / len(output_texts) if output_texts else 0
                    ),
                }
            )
            # Check for proper triggering condition closure
            triggering_condition_pattern = "</triggering_condition>"
            ends_with_triggering = [text.strip().endswith(triggering_condition_pattern) for text in output_texts]

            base_analysis.update(
                {
                    "ends_with_triggering_condition": ends_with_triggering,
                    "proper_triggering_closure_rate": (
                        sum(ends_with_triggering) / len(output_texts) if output_texts else 0.0
                    ),
                    "triggering_condition_positions": [
                        text.rfind(triggering_condition_pattern) if triggering_condition_pattern in text else -1
                        for text in output_texts
                    ],
                    "completion_type": "sneaky_solution",
                }
            )

        elif model_mode == "honest_prover":
            # Honest completions are fixed mono solutions, set defaults
            base_analysis.update(
                {
                    "ends_with_triggering_condition": [False] * len(output_texts),
                    "proper_triggering_closure_rate": 0.0,
                    "triggering_condition_positions": [-1] * len(output_texts),
                    "completion_type": "fixed_mono_solution",
                }
            )

        elif model_mode == "verifier":
            # Verifier outputs scalars, different analysis
            base_analysis.update(
                {
                    "ends_with_triggering_condition": [False] * len(output_texts),  # N/A for verifier
                    "proper_triggering_closure_rate": 0.0,  # N/A for verifier
                    "triggering_condition_positions": [-1] * len(output_texts),  # N/A for verifier
                    "completion_type": "scalar_output",
                    "verifier_output_analysis": self._analyze_verifier_outputs(output_texts),
                }
            )

        return base_analysis

    def _analyze_verifier_outputs(self, output_texts: List[str]) -> Dict[str, Any]:
        """Analyze verifier scalar outputs."""
        try:
            # Try to parse as floats for analysis
            numeric_outputs = []
            parse_errors = 0

            for text in output_texts:
                try:
                    numeric_outputs.append(float(text.strip()))
                except (ValueError, AttributeError):
                    parse_errors += 1

            if numeric_outputs:
                return {
                    "numeric_outputs": numeric_outputs,
                    "output_range": [min(numeric_outputs), max(numeric_outputs)],
                    "output_mean": sum(numeric_outputs) / len(numeric_outputs),
                    "parse_error_rate": parse_errors / len(output_texts) if output_texts else 0.0,
                }
            else:
                return {
                    "numeric_outputs": [],
                    "output_range": [None, None],
                    "output_mean": None,
                    "parse_error_rate": 1.0,
                }
        except Exception:
            return {"analysis_error": True, "parse_error_rate": 1.0}


class InteractionLogger:
    """Main interaction logging component - replaces vLLM orchestrator logging."""

    def __init__(
        self,
        log_dir: str,
        global_step_callback: Callable[[], int],
        phase_callback: Callable[[], str],
        accelerator_manager,
        completion_analyzer: CompletionAnalyzer,
    ):
        self.log_dir = log_dir
        self.global_step_callback = global_step_callback
        self.phase_callback = phase_callback
        self.accelerator_manager = accelerator_manager
        self.completion_analyzer = completion_analyzer

        # Session tracking
        self.current_batch_id: Optional[str] = None
        self.generation_sequence_counter = 0
        self.interaction_registry: Dict[str, Dict[str, Any]] = {}

    def log_generation_interaction(
        self,
        model_mode: str,
        prompts: List[str],
        output_ids: List[List[int]],
        output_texts: List[str],
        logprobs: Optional[List[Dict[int, float]]] = None,
        generation_args: Optional[Dict[str, Any]] = None,
        n_generations: int = 1,
        prompts_len_local: int = 0,
        is_instruction: bool = False,
        tokenizer_info: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Log a generation interaction. Returns interaction_id for later enhancement."""

        if not self.accelerator_manager.get_state_property("is_main_process"):
            return ""  # Return empty string for non-main processes

        # Generate unique identifiers
        interaction_id = str(uuid.uuid4())
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

        # Ensure batch session is active
        if not self.current_batch_id:
            self.start_batch_session()

        self.generation_sequence_counter += 1

        # Calculate ordinality information
        process_index = self.accelerator_manager.get_state_property("process_index")
        prompt_indices = self._calculate_prompt_indices(prompts_len_local, process_index, n_generations)

        # Analyze completions with domain-specific logic
        completion_analysis = self.completion_analyzer.analyze_completions(output_ids, output_texts, model_mode)

        # Create log entry
        log_entry = InteractionLogEntry(
            interaction_id=interaction_id,
            timestamp_utc=timestamp,
            model_mode=model_mode,
            step=self.global_step_callback(),
            phase=self.phase_callback(),
            batch_id=self.current_batch_id,
            prompts=prompts,
            output_ids=output_ids,
            output_texts=output_texts,
            logprobs=logprobs,
            ordinality={
                "generation_sequence": self.generation_sequence_counter,
                "process_index": process_index,
                "num_processes": self.accelerator_manager.get_state_property("num_processes"),
                "local_batch_size": prompts_len_local,
                "global_batch_size": len(prompts),
                "prompt_indices": prompt_indices,
                "generation_order": self._get_generation_order_info(model_mode, prompt_indices),
            },
            generation_metadata={
                "n_generations": n_generations,
                "generation_args": generation_args or {},
                "is_instruction": is_instruction,
                "tokenizer_info": tokenizer_info or {},
            },
            completion_analysis=completion_analysis,
        )

        # Save to file system
        log_filepath = self._save_log_entry(log_entry)

        # Register for later enhancement
        self._register_interaction(interaction_id, log_filepath, model_mode)

        logger.debug(f"Logged generation interaction {interaction_id} for {model_mode}")
        return interaction_id

    def enhance_interactions_with_rewards(
        self,
        verifier_scores: torch.Tensor,
        honest_rewards: torch.Tensor,
        sneaky_rewards: torch.Tensor,
        honest_advantages: torch.Tensor,
        sneaky_advantages: torch.Tensor,
        behavioral_metrics: Dict[str, float],
        reward_statistics: Dict[str, float],
    ) -> None:
        """Enhance all interactions in current batch with reward data."""

        if not self.accelerator_manager.get_state_property("is_main_process"):
            return

        if not self.current_batch_id:
            logger.warning("No active batch session for reward enhancement")
            return

        # Find interactions for current batch
        batch_interactions = {
            iid: info for iid, info in self.interaction_registry.items() if info["batch_id"] == self.current_batch_id
        }

        if not batch_interactions:
            logger.warning(f"No interactions found for batch {self.current_batch_id} to enhance")
            return

        # Convert tensors to lists for JSON serialization
        reward_data = {
            "verifier_scores": (
                verifier_scores.cpu().tolist() if isinstance(verifier_scores, torch.Tensor) else verifier_scores
            ),
            "honest_rewards": (
                honest_rewards.cpu().tolist() if isinstance(honest_rewards, torch.Tensor) else honest_rewards
            ),
            "sneaky_rewards": (
                sneaky_rewards.cpu().tolist() if isinstance(sneaky_rewards, torch.Tensor) else sneaky_rewards
            ),
            "honest_advantages": (
                honest_advantages.cpu().tolist() if isinstance(honest_advantages, torch.Tensor) else honest_advantages
            ),
            "sneaky_advantages": (
                sneaky_advantages.cpu().tolist() if isinstance(sneaky_advantages, torch.Tensor) else sneaky_advantages
            ),
            "behavioral_metrics": behavioral_metrics,
            "reward_statistics": reward_statistics,
            "enhancement_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }

        # Enhance each interaction
        enhanced_count = 0
        for interaction_id, interaction_info in batch_interactions.items():
            try:
                self._enhance_interaction_file(
                    interaction_info["log_filepath"], interaction_info["model_mode"], reward_data, interaction_id
                )
                enhanced_count += 1
            except Exception as e:
                logger.error(f"Failed to enhance interaction {interaction_id}: {e}")

        logger.info(
            f"Enhanced {enhanced_count}/{len(batch_interactions)} interactions with reward data for batch {self.current_batch_id}"
        )

    def start_batch_session(self) -> str:
        """Start a new batch session."""
        self.current_batch_id = f"batch_{self.global_step_callback()}_{int(datetime.datetime.now().timestamp())}"
        self.generation_sequence_counter = 0
        logger.debug(f"Started batch session: {self.current_batch_id}")
        return self.current_batch_id

    def finalize_batch_session(self) -> None:
        """Finalize current batch session."""
        if self.current_batch_id:
            logger.debug(f"Finalizing batch session: {self.current_batch_id}")
            self._create_batch_summary()
            self.current_batch_id = None

    def _save_log_entry(self, log_entry: InteractionLogEntry) -> str:
        """Save log entry to file system with existing naming convention."""

        # Create step directory
        step_dir = os.path.join(self.log_dir, f"step_{log_entry.step}")
        os.makedirs(step_dir, exist_ok=True)

        # Use existing file naming convention
        timestamp_safe = log_entry.timestamp_utc.replace(":", "-")
        filename = f"{timestamp_safe}_{log_entry.model_mode}_{log_entry.interaction_id}.json"
        filepath = os.path.join(step_dir, filename)

        # Convert to existing JSON format for backward compatibility
        log_data = {
            # === EXISTING FIELDS (unchanged) ===
            "interaction_id": log_entry.interaction_id,
            "timestamp_utc": log_entry.timestamp_utc,
            "model_mode": log_entry.model_mode,
            "prompts": log_entry.prompts,
            "output_ids": log_entry.output_ids,
            "output_texts": log_entry.output_texts,
            **({"logprobs": log_entry.logprobs} if log_entry.logprobs is not None else {}),
            # === NEW ENHANCED FIELDS ===
            "step": log_entry.step,
            "phase": log_entry.phase,
            "batch_id": log_entry.batch_id,
            "ordinality": log_entry.ordinality,
            "generation_metadata": log_entry.generation_metadata,
            "completion_analysis": log_entry.completion_analysis,
            "reward_data": log_entry.reward_data,
            "behavioral_metrics": log_entry.behavioral_metrics,
            "verification_results": log_entry.verification_results,
            "enhancement_status": log_entry.enhancement_status,
        }

        # Save to file
        try:
            with open(filepath, "w") as f:
                json.dump(log_data, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to save interaction log to {filepath}: {e}")
            raise

        return filepath

    def _enhance_interaction_file(
        self, log_filepath: str, model_mode: str, reward_data: Dict[str, Any], interaction_id: str
    ) -> None:
        """Enhance existing log file with reward data."""
        try:
            # Read existing log
            with open(log_filepath, "r") as f:
                log_data = json.load(f)

            # Extract prompt indices for mapping
            prompt_indices = log_data.get("ordinality", {}).get("prompt_indices", [])

            # Add model-specific reward data
            if model_mode == "honest_prover":
                relevant_rewards = [
                    reward_data["honest_rewards"][i] for i in prompt_indices if i < len(reward_data["honest_rewards"])
                ]
                relevant_advantages = [
                    reward_data["honest_advantages"][i]
                    for i in prompt_indices
                    if i < len(reward_data["honest_advantages"])
                ]

                log_data["reward_data"] = {
                    "rewards": relevant_rewards,
                    "advantages": relevant_advantages,
                    "reward_statistics": reward_data["reward_statistics"],
                    "prompt_reward_mapping": list(zip(prompt_indices, relevant_rewards, relevant_advantages)),
                }

            elif model_mode == "sneaky_prover":
                relevant_rewards = [
                    reward_data["sneaky_rewards"][i] for i in prompt_indices if i < len(reward_data["sneaky_rewards"])
                ]
                relevant_advantages = [
                    reward_data["sneaky_advantages"][i]
                    for i in prompt_indices
                    if i < len(reward_data["sneaky_advantages"])
                ]

                log_data["reward_data"] = {
                    "rewards": relevant_rewards,
                    "advantages": relevant_advantages,
                    "reward_statistics": reward_data["reward_statistics"],
                    "prompt_reward_mapping": list(zip(prompt_indices, relevant_rewards, relevant_advantages)),
                }

            elif model_mode == "verifier":
                log_data["verification_results"] = {
                    "verifier_scores": reward_data["verifier_scores"],
                    "score_prompt_mapping": list(
                        zip(
                            prompt_indices,
                            [
                                reward_data["verifier_scores"][i]
                                for i in prompt_indices
                                if i < len(reward_data["verifier_scores"])
                            ],
                        )
                    ),
                }

            # Add behavioral metrics to all
            log_data["behavioral_metrics"] = reward_data["behavioral_metrics"]

            # Update enhancement status
            log_data["enhancement_status"].update(
                {
                    "rewards_added": True,
                    "behavioral_added": True,
                    "verification_added": model_mode == "verifier",
                    "last_enhanced": reward_data["enhancement_timestamp"],
                }
            )

            # Save enhanced log
            with open(log_filepath, "w") as f:
                json.dump(log_data, f, indent=4)

        except Exception as e:
            logger.error(f"Failed to enhance interaction {interaction_id}: {e}")
            raise

    # Helper methods
    def _calculate_prompt_indices(self, prompts_len_local: int, process_index: int, n_generations: int) -> List[int]:
        """Calculate the global prompt indices for this process."""
        start_idx = process_index * prompts_len_local
        return list(range(start_idx, start_idx + prompts_len_local))

    def _get_generation_order_info(self, model_mode: str, prompt_indices: List[int]) -> Dict[str, Any]:
        """Get generation order information."""
        return {
            "model_mode": model_mode,
            "sequence_number": self.generation_sequence_counter,
            "prompt_indices": prompt_indices,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }

    def _register_interaction(self, interaction_id: str, log_filepath: str, model_mode: str) -> None:
        """Register interaction for later enhancement."""
        self.interaction_registry[interaction_id] = {
            "log_filepath": log_filepath,
            "model_mode": model_mode,
            "batch_id": self.current_batch_id,
            "step": self.global_step_callback(),
        }

    def _create_batch_summary(self) -> None:
        """Create batch-level summary."""
        if not self.current_batch_id:
            return

        batch_interactions = {
            iid: info for iid, info in self.interaction_registry.items() if info["batch_id"] == self.current_batch_id
        }

        if not batch_interactions:
            return

        summary = {
            "batch_id": self.current_batch_id,
            "step": self.global_step_callback(),
            "phase": self.phase_callback(),
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "total_interactions": len(batch_interactions),
            "model_modes": list(set(info["model_mode"] for info in batch_interactions.values())),
            "interaction_ids": list(batch_interactions.keys()),
        }

        # Save batch summary
        step_dir = os.path.join(self.log_dir, f"step_{self.global_step_callback()}")
        os.makedirs(step_dir, exist_ok=True)
        summary_path = os.path.join(step_dir, f"batch_summary_{self.current_batch_id}.json")

        try:
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to save batch summary: {e}")
