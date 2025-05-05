# pvg/components/model_manager.py

# ModelManager
# Responsibility: Loads, manages, and provides access to the policy models (honest_prover, sneaky_prover) and their corresponding reference models (if beta > 0). Handles applying Liger kernel and enabling gradient checkpointing. Coordinates with AcceleratorManager to prepare models.

import os
from typing import Literal, Callable, Any
from collections.abc import Iterator
from pvg.utils.utils import prepare_deepspeed
import torch
from pvg.components.accelerator_manager import AcceleratorManager
from pvg.config.args import ModelArgs, TrainingArgs, RLArgs
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification
from liger_kernel.transformers import _apply_liger_kernel_to_instance
from liger_kernel.chunked_loss import LigerFusedLinearGRPOLoss
import logging

logger = logging.getLogger(f"pvg.{__name__}")  # Get a child logger


class ModelManager:
    """
    Loads, manages, and provides access to the policy models (honest_prover, sneaky_prover, verifier) and their corresponding reference models (if beta > 0), conditional on what we are training.
    Handles applying Liger kernel and enabling gradient checkpointing. Coordinates with AcceleratorManager to prepare models.
    """

    def __init__(
        self,
        accelerator_manager: AcceleratorManager,
        global_phase_callback: Callable[[], Literal["verifier", "provers"]],
        global_round_callback: Callable[[], int],
        global_step_callback: Callable[[], int],
        honest_config: ModelArgs,  # Only used if we are training honest prover
        sneaky_config: ModelArgs,  # Only used if we are training sneaky prover
        verifier_config: ModelArgs,  # Only used if we are training verifier
        honest_training_config: TrainingArgs,  # Only used if we are training honest prover
        sneaky_training_config: TrainingArgs,  # Only used if we are training sneaky prover
        verifier_training_config: TrainingArgs,  # Only used if we are training verifier
        rl_config: RLArgs,  # Only used if we are training RL
    ) -> None:
        """
        Initializes the ModelManager.

        Args:
            accelerator_manager: AcceleratorManager - The accelerator manager.
            honest_config: ModelArgs - The configuration for the honest prover model.
            sneaky_config: ModelArgs - The configuration for the sneaky prover model.
            verifier_config: ModelArgs - The configuration for the verifier model.
            honest_training_config: TrainingArgs - The training configuration for the honest prover model.
            sneaky_training_config: TrainingArgs - The training configuration for the sneaky prover model.
            verifier_training_config: TrainingArgs - The training configuration for the verifier model.
            rl_config: RLArgs - The configuration for the RL model.
        """
        self.accelerator_manager = accelerator_manager
        self.configs = {
            "honest_prover": honest_config,
            "sneaky_prover": sneaky_config,
            "verifier": verifier_config,
        }
        self.training_configs = {
            "honest_prover": honest_training_config,
            "sneaky_prover": sneaky_training_config,
            "verifier": verifier_training_config,
        }
        self.rl_config = rl_config

        self.model_paths: dict[str, str] = {}
        self.models: dict[str, torch.nn.Module] = {}
        self.ref_models: dict[str, torch.nn.Module | None] = {}
        self.prepared_models: dict[str, torch.nn.Module] = {}
        self.prepared_ref_models: dict[str, torch.nn.Module | None] = {}
        self.liger_grpo_loss: LigerFusedLinearGRPOLoss | None = None
        self.verifier_mode: (
            Literal[
                "regressor", "classifier", "inference_classifier", "inference_regressor"
            ]
            | None
        ) = (
            self.training_configs["verifier"].verifier_mode
            if self.training_configs["verifier"] is not None
            else None
        )

        self.global_phase_callback: Callable[[], Literal["verifier", "provers"]] = (
            global_phase_callback
        )
        self.global_round_callback: Callable[[], int] = global_round_callback
        self.global_step_callback: Callable[[], int] = global_step_callback

        # Check what are we training
        self.phase: Literal["verifier", "provers"] = (
            self.global_phase_callback()
        )  # Phase is either "verifier" (i.e., we are training verifier) or "provers" (i.e., we are training honest and sneaky prover)
        # NOTE: During init, self.phase will be set to "verifier", as this is the first component to be trained.

    # Helper functions
    def _fetch_local_models(self, model_path: str) -> str:
        """Fetches a local model from a given path."""
        local_model_path = "/home/jvelja/local_models"  # TODO: Hardcoded... Should fix
        full_model_path = os.path.join(local_model_path, model_path)
        if os.path.exists(full_model_path):
            return full_model_path
        else:
            raise FileNotFoundError(f"Model not found at {full_model_path}")

    def _load_models(self) -> None:
        """Loads policy and reference models from paths specified in configs. Applies Liger/gradient checkpointing. Called during init."""

        # Reset Liger GRPO loss
        self.liger_grpo_loss: LigerFusedLinearGRPOLoss | None = None

        # Fetch models from local paths (& update path to model if found)
        for model_key in self.configs.keys():
            self.model_paths[model_key] = self._fetch_local_models(
                self.configs[model_key].name_or_path
            )  # Gets the path to the model (if stored locally)

        # Load models - phase-dependent for memory efficiency
        if self.phase == "verifier":
            model_init_kwargs = {}
            model_init_kwargs["use_cache"] = (
                False
                if self.training_configs["verifier"].gradient_checkpointing
                else model_init_kwargs.get("use_cache")
            )
            model_init_kwargs["attn_implementation"] = (
                "flash_attention_2"
                if self.configs["verifier"].use_flash_attention
                else model_init_kwargs.get("attn_implementation", None)
            )
            self.prepare_verifier(
                model_init_kwargs
            )  # Special case for verifier --> can be an RM or a simple language model
            # TODO: the

        elif self.phase == "provers":
            for model_key in self.configs.keys():
                model_init_kwargs = {}
                model_init_kwargs["use_cache"] = (
                    False
                    if self.training_configs[model_key].gradient_checkpointing
                    else model_init_kwargs.get("use_cache")
                )
                model_init_kwargs["attn_implementation"] = (
                    "flash_attention_2"
                    if self.configs[model_key].use_flash_attention
                    else model_init_kwargs.get("attn_implementation", None)
                )
                self.models[model_key] = AutoModelForCausalLM.from_pretrained(
                    self.model_paths[model_key],
                    torch_dtype=torch.bfloat16,
                    **model_init_kwargs,
                )
        else:
            raise ValueError(f"Invalid phase: {self.phase}")

        # Enable gradient checkpointing if specified + reference models if beta > 0.0
        for model_key in self.models:
            if (
                self.training_configs[model_key].gradient_checkpointing
                and self.models[model_key] is not None
            ):  # Only enable gradient checkpointing if model is not None (i.e., if we are training either provers or verifier)
                self.models[model_key] = self._enable_gradient_checkpointing(
                    self.models[model_key]
                )

            if (
                self.rl_config.beta > 0.0 and self.models[model_key] is not None
            ):  # Only load reference models if KL beta > 0.0
                self.ref_models[model_key] = AutoModelForCausalLM.from_pretrained(
                    self.model_paths[model_key]
                )
            else:
                self.ref_models[model_key] = None

        # Apply Liger kernel if specified
        # Check what is being trained (honest, sneaky, verifier) and apply Liger kernel to the corresponding models
        if self.phase == "provers":  # Prover training mode
            for model_key in self.configs.keys():
                if self.training_configs[model_key].apply_liger_kernel:
                    _apply_liger_kernel_to_instance(self.models[model_key])

            # Initialize Liger GRPO loss (Provers always use GRPO)
            self.liger_grpo_loss = (
                LigerFusedLinearGRPOLoss(
                    beta=self.rl_config.beta,
                    epsilon_low=self.rl_config.epsilon_low,
                    epsilon_high=self.rl_config.epsilon_high,
                    temperature=self.training_configs["honest_prover"].temperature,
                    use_ref_model=True if self.rl_config.beta != 0.0 else False,
                )
                if self.training_configs["honest_prover"].apply_liger_kernel
                else None
            )

        # Apply Liger kernel to verifier if specified
        elif (
            self.phase == "verifier"
            and self.training_configs["verifier"].apply_liger_kernel
        ):  # Verifier training mode
            _apply_liger_kernel_to_instance(self.models["verifier"])
            assert (
                self.liger_grpo_loss is None
            ), "Liger GRPO loss already initialized. This should not happen."

            if (
                self.training_configs["verifier"].verifier_mode
                == "inference_classifier"
                or self.training_configs["verifier"].verifier_mode
                == "inference_regressor"
            ):
                self.liger_grpo_loss = LigerFusedLinearGRPOLoss(
                    beta=self.rl_config.beta,
                    epsilon_low=self.rl_config.epsilon_low,
                    epsilon_high=self.rl_config.epsilon_high,
                    temperature=self.training_configs["verifier"].temperature,
                    use_ref_model=True if self.rl_config.beta != 0.0 else False,
                )
            else:
                # No GRPO Loss for verifier if it is not trained via RL
                self.liger_grpo_loss = None
        # Set train/eval mode for models
        if self.phase == "provers":
            self.models["honest_prover"].train()
            self.models["sneaky_prover"].train()
        elif self.phase == "verifier":
            self.models["verifier"].train()
        else:
            raise ValueError(f"Invalid phase: {self.phase}")

        # Logger info
        logger.info("=== Model Loading Summary ===")
        logger.info(f"Loaded models: {list(self.models.keys())}")
        for model_key in self.models:
            logger.info(
                f"- {model_key} model loaded from: {self.model_paths[model_key]}"
            )
            if self.training_configs[model_key].gradient_checkpointing:
                logger.info(f"  - Gradient checkpointing enabled for {model_key}")
            if self.training_configs[model_key].apply_liger_kernel:
                logger.info(f"  - Liger kernel applied to {model_key}")

        logger.info("\nReference Models:")
        for model_key, ref_model in self.ref_models.items():
            if ref_model is not None:
                logger.info(f"- {model_key} reference model loaded")
            else:
                logger.info(f"- {model_key} reference model not loaded")

        if self.liger_grpo_loss is not None:
            logger.info("\nLiger GRPO Loss Configuration:")
            logger.info(f"- Beta: {self.rl_config.beta}")
            logger.info(
                f"- Epsilon range: [{self.rl_config.epsilon_low}, {self.rl_config.epsilon_high}]"
            )
            logger.info(f"- Using reference model: {self.rl_config.beta != 0.0}")
        else:
            logger.info("\nLiger GRPO Loss not configured")

        logger.info("=============================")

    def _enable_gradient_checkpointing(self, model: torch.nn.Module) -> torch.nn.Module:
        """Enables gradient checkpointing for the model."""
        # Ensure use_cache is disabled
        model.config.use_cache = False
        model.gradient_checkpointing_enable()

        return model

    def prepare_models(self) -> None:
        """Calls accelerator_manager.prepare_model for all loaded policy and reference models. Stores prepared models. Must be called after AcceleratorManager is fully initialized."""
        for model_key in self.models:
            if self.models[model_key] is not None:
                self.prepared_models[model_key] = (
                    self.accelerator_manager.prepare_model(
                        self.models[model_key], key=model_key
                    )
                )  # TODO: This is wrong: Models, dataloaders and optimizers should be prepared altogether.

        for model_key in self.ref_models:
            if self.ref_models[model_key] is not None:
                self.prepared_ref_models[model_key] = prepare_deepspeed(
                    self.ref_models[model_key],
                    self.accelerator_manager.get_accelerator(model_key),
                )

    def get_model(self, key: str, prepared: bool = True) -> torch.nn.Module:
        """Returns the requested model (prepared or unprepared)."""
        if prepared and key in self.prepared_models:
            return self.prepared_models[key]
        else:
            logger.warning(f"Model {key} not prepared. Returning unprepared model.")
            if key not in self.models:
                logger.warning(
                    f"Model {key} not found. Note keys are: {list(self.models.keys())}"
                )
            else:
                return self.models[key]

    def get_ref_model(self, key: str, prepared: bool = True) -> torch.nn.Module:
        """Returns the requested reference model (prepared or unprepared)."""
        if prepared and key in self.prepared_ref_models:
            return self.prepared_ref_models[key]
        else:
            logger.warning(
                f"Reference model {key} not prepared. Returning unprepared model."
            )
            if key not in self.ref_models:
                logger.warning(
                    f"Reference model {key} not found. Note keys are: {list(self.ref_models.keys())}"
                )
            else:
                return self.ref_models[key]

    def get_verifier_head(self) -> torch.nn.Linear:
        """Returns the verifier head."""
        if self.models["verifier"] is not None:
            return self.models["verifier"].verifier_head
        else:
            logger.warning("Verifier head not found. Returning None.")
            return None

    def get_liger_loss_calculator(self) -> LigerFusedLinearGRPOLoss:
        """Returns the Liger loss calculator."""
        if self.liger_grpo_loss is not None:
            return self.liger_grpo_loss
        else:
            logger.warning("Liger loss calculator not found. Returning None.")
            return None

    def set_train_mode(
        self, model_key: Literal["honest", "sneaky", "verifier"], train: bool = True
    ) -> None:
        """Sets the train mode for the specified model."""
        if train:
            self.models[model_key].train()
        else:
            self.models[model_key].eval()

    def get_model_parameters(
        self, model_key: Literal["honest", "sneaky", "verifier"], prepared: bool = True
    ) -> Iterator[torch.nn.parameter.Parameter]:
        """Returns an iterator over the parameters of the specified model."""
        if prepared and model_key in self.prepared_models:
            return self.prepared_models[model_key].parameters()
        else:
            return self.models[model_key].parameters()

    def prepare_verifier(self, model_init_kwargs: dict[str, Any] = {}) -> None:
        """Prepares the verifier model. Possible verifier modes:
        1. **Verifier as Regressor**:
            Implements a verifier that outputs continuous scores for solutions, trained with Bradley-Terry pairwise comparison loss to maximize ranking accuracy between correct and incorrect solutions.
        2. **Verifier as Classifier**:
            Implements a verifier that outputs binary correctness probabilities, trained with cross-entropy loss to directly classify solutions as correct or incorrect.
        3. **Verifier as Inference-time Classifier**:
            Implements a verifier that generates chain-of-thought reasoning before classification, trained to output binary verdicts after explicit reasoning steps.
        4. **Verifier as Inference-time Regressor**:
            Implements a verifier that generates chain-of-thought reasoning before scoring, trained with Bradley-Terry objective applied to the logits of the verdict token.

        Verifier as a inference-time regressor implies an additional linear layer on top of the model's final hidden state that outputs a single score for the solution.
        """

        if self.training_configs["verifier"].verifier_mode == "regressor":
            # Regressor --> AutoModelforSequenceClassification with 1 output neuron (num_labels = 1)
            # Set up a config with pad_token_id = eos_token_id
            model_init_kwargs["pad_token_id"] = (
                151643  # TODO: This is a hack. We should fix this.
            )
            self.models["verifier"] = (
                AutoModelForSequenceClassification.from_pretrained(
                    self.model_paths["verifier"], num_labels=1, **model_init_kwargs
                )
            )
        elif self.training_configs["verifier"].verifier_mode == "classifier":
            # Classifier --> Language model that outputs <verdict>...</verdict> token (binary scoring 0-1)
            self.models["verifier"] = AutoModelForCausalLM.from_pretrained(
                self.model_paths["verifier"], **model_init_kwargs
            )
        elif self.training_configs["verifier"].verifier_mode == "inference_classifier":
            # Inference-time classifier --> Language model that outputs <verdict>...</verdict> token (binary scoring 0-1) but with chain-of-thought reasoning steps before:
            # <verification> ... </verification> <verdict> ... </verdict>
            self.models["verifier"] = AutoModelForCausalLM.from_pretrained(
                self.model_paths["verifier"], **model_init_kwargs
            )
        elif self.training_configs["verifier"].verifier_mode == "inference_regressor":
            # Inference-time regressor --> Language model that outputs <verdict>...</verdict> token (binary scoring 0-1) but with chain-of-thought reasoning steps before:
            # <verification> ... </verification> <verdict> ... </verdict>
            # And takes hidden state of <verdict> (referred as "verdict_token") and applies a linear layer on top of it that outputs a single score for the solution. TO BE TRAINED!
            self.models["verifier"] = AutoModelForCausalLM.from_pretrained(
                self.model_paths["verifier"], **model_init_kwargs
            )
            self.models["verifier"].verifier_head = torch.nn.Linear(
                self.models["verifier"].config.hidden_size, 1
            )
        else:
            raise ValueError(
                f"Invalid verifier mode: {self.training_configs['verifier'].verifier_mode}"
            )

    # NOTE: This function is called when the phase changes - i.e., when the state tracker is updated (end of verifier training, end of provers training, ...)
    # Acts as a reset for the model manager
    def load_models(self):
        """Initializes the model manager (loads models, prepares models). Called when the phase changes or phase-specific training starts. Acts as a reset for the model manager."""
        self.phase = (
            self.global_phase_callback()
        )  # NOTE: Has to be called **after** the state tracker has been incremented
        self._load_models()  # NOTE: Wonnky, fix later
