# pvg/components/model_manager.py

import gc
import logging
import os
from collections.abc import Iterator
from typing import Any, Callable, Literal

import torch
from liger_kernel.chunked_loss import LigerFusedLinearGRPOLoss
from liger_kernel.transformers import _apply_liger_kernel_to_instance
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from pvg.components.accelerator_manager import AcceleratorManager
from pvg.config.args import ModelArgs, RLArgs, TrainingArgs

logger = logging.getLogger(f"pvg.{__name__}")  # Get a child logger


class ModelManager:
    """
    Loads, manages, and provides access to the policy models (sneaky_prover, verifier) and their corresponding reference models (if beta > 0), conditional on what we are training.
    Handles applying Liger kernel and enabling gradient checkpointing. Coordinates with AcceleratorManager to prepare models.
    """

    def __init__(
        self,
        accelerator_manager: AcceleratorManager,
        global_phase_callback: Callable[[], Literal["verifier", "provers"]],
        global_round_callback: Callable[[], int],
        global_step_callback: Callable[[], int],
        sneaky_config: ModelArgs,  # Only used if we are training sneaky prover
        verifier_config: ModelArgs,  # Only used if we are training verifier
        sneaky_training_config: TrainingArgs,  # Only used if we are training sneaky prover
        verifier_training_config: TrainingArgs,  # Only used if we are training verifier
        rl_config: RLArgs,  # Only used if we are training RL
    ) -> None:
        """
        Initializes the ModelManager.

        Args:
            accelerator_manager: AcceleratorManager - The accelerator manager.
            sneaky_config: ModelArgs - The configuration for the sneaky prover model.
            verifier_config: ModelArgs - The configuration for the verifier model.
            sneaky_training_config: TrainingArgs - The training configuration for the sneaky prover model.
            verifier_training_config: TrainingArgs - The training configuration for the verifier model.
            rl_config: RLArgs - The configuration for the RL model.
        """
        self.accelerator_manager = accelerator_manager
        self.configs = {
            "sneaky_prover": sneaky_config,
            "verifier": verifier_config,
        }
        self.training_configs = {
            "sneaky_prover": sneaky_training_config,
            "verifier": verifier_training_config,
        }
        self.rl_config = rl_config

        self.model_paths: dict[str, str] = {}
        self.models: dict[str, torch.nn.Module] = {}
        self.ref_models: dict[str, torch.nn.Module | None] = {}
        self.prepared_models: dict[str, torch.nn.Module] = {}
        self.prepared_ref_models: dict[str, torch.nn.Module | None] = {}
        self.tokenizer: AutoTokenizer | None = None
        self.verifier_mode: Literal["regressor", "classifier", "inference_classifier", "inference_regressor"] | None = (
            self.training_configs["verifier"].verifier_mode if self.training_configs["verifier"] is not None else None
        )

        self.global_phase_callback: Callable[[], Literal["verifier", "provers"]] = global_phase_callback
        self.global_round_callback: Callable[[], int] = global_round_callback
        self.global_step_callback: Callable[[], int] = global_step_callback

        # Check what are we training
        self.phase: Literal["verifier", "provers"] = (
            self.global_phase_callback()
        )  # Phase is either "verifier" (i.e., we are training verifier) or "provers" (i.e., we are training sneaky prover)
        # NOTE: During init, self.phase will be set to "verifier", as this is the first component to be trained.

    def _log_memory_usage(self, context: str) -> None:
        """Log current GPU memory usage with context."""
        if torch.cuda.is_available():
            current_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            reserved_memory = torch.cuda.memory_reserved() / 1024 / 1024  # MB
            max_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB

            logger.info(f"=== MEMORY USAGE ({context}) ===")
            logger.info(f"Current Allocated: {current_memory:.2f} MB")
            logger.info(f"Reserved: {reserved_memory:.2f} MB")
            logger.info(f"Max Allocated: {max_memory:.2f} MB")

            # Log per-device memory usage
            for i in range(torch.cuda.device_count()):
                device_allocated = torch.cuda.memory_allocated(i) / 1024 / 1024  # MB
                device_reserved = torch.cuda.memory_reserved(i) / 1024 / 1024  # MB
                logger.info(f"Device {i}: Allocated {device_allocated:.2f} MB, Reserved {device_reserved:.2f} MB")

            logger.info("=" * 50)

    def offload_models_to_cpu(self, model_keys: list[str]) -> None:
        """
        Offload specified models from GPU to CPU to free GPU memory.

        Args:
            model_keys: List of model keys to offload ('verifier', 'sneaky_prover', etc.)
        """
        logger.info(f"Offloading models to CPU: {model_keys}")
        self._log_memory_usage("BEFORE model offloading")

        for model_key in model_keys:
            # Offload prepared models
            if model_key in self.prepared_models:
                model = self.prepared_models[model_key]
                if hasattr(model, "cpu"):
                    logger.info(f"Moving prepared model '{model_key}' to CPU")
                    model.cpu()

            # Offload unprepared models
            if model_key in self.models:
                model = self.models[model_key]
                if hasattr(model, "cpu"):
                    logger.info(f"Moving model '{model_key}' to CPU")
                    model.cpu()

            # Offload reference models
            if model_key in self.ref_models and self.ref_models[model_key] is not None:
                ref_model = self.ref_models[model_key]
                if hasattr(ref_model, "cpu"):
                    logger.info(f"Moving reference model '{model_key}' to CPU")
                    ref_model.cpu()

            # Offload prepared reference models
            if model_key in self.prepared_ref_models and self.prepared_ref_models[model_key] is not None:
                ref_model = self.prepared_ref_models[model_key]
                if hasattr(ref_model, "cpu"):
                    logger.info(f"Moving prepared reference model '{model_key}' to CPU")
                    ref_model.cpu()

        # Force cleanup after offloading
        torch.cuda.empty_cache()
        gc.collect()

        self._log_memory_usage("AFTER model offloading")

    def fully_offload_models(self, model_keys: list[str] | None = None) -> None:
        """
        Fully offload models from GPU memory and clear references.

        Args:
            model_keys: List of model keys to offload. If None, offload all models.
                       Use this for between-rounds cleanup only.
        """
        if model_keys is None:
            logger.info("Fully offloading ALL models from GPU memory (between rounds)")
            # Get all model keys
            all_model_keys = set()
            all_model_keys.update(self.models.keys())
            all_model_keys.update(self.prepared_models.keys())
            all_model_keys.update(self.ref_models.keys())
            all_model_keys.update(self.prepared_ref_models.keys())
            model_keys = list(all_model_keys)
        else:
            logger.warning(f"Selective model offloading not recommended: {model_keys}")
            logger.warning("Use phase strategies for proper model lifecycle management")

        self._log_memory_usage("BEFORE full model offloading")

        # Offload all models to CPU first to ensure proper cleanup
        self.offload_models_to_cpu(model_keys)

        # Clear all model references
        for model_key in model_keys:
            self.models.pop(model_key, None)
            self.prepared_models.pop(model_key, None)
            self.ref_models.pop(model_key, None)
            self.prepared_ref_models.pop(model_key, None)

        # Additional cleanup
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        gc.collect()

        self._log_memory_usage("AFTER full model offloading")

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
        logger.info(f"Loading models for phase: {self.phase}")
        self._log_memory_usage("BEFORE model loading")

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
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_paths["verifier"])

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
                ).to(self.accelerator_manager.get_state_property("device", model_key))

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_paths["sneaky_prover"])
            # Make sure that the tokenizer has a pad_token_id
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                logger.warning(
                    f"Tokenizer {self.tokenizer} has no pad_token_id. Setting it to {self.tokenizer.eos_token_id}"
                )
        else:
            raise ValueError(f"Invalid phase: {self.phase}")

        # Enable gradient checkpointing if specified + reference models if beta > 0.0
        for model_key in self.models:
            if (
                self.training_configs[model_key].gradient_checkpointing and self.models[model_key] is not None
            ):  # Only enable gradient checkpointing if model is not None (i.e., if we are training either provers or verifier)
                self.models[model_key] = self._enable_gradient_checkpointing(self.models[model_key])

            if self.rl_config.beta > 0.0 and (
                self.models[model_key] is not None and model_key != "verifier"
            ):  # Only load reference models if KL beta > 0.0 and model is not verifier
                self.ref_models[model_key] = AutoModelForCausalLM.from_pretrained(self.model_paths[model_key])
            else:
                self.ref_models[model_key] = None

        # Apply Liger kernel if specified
        # Check what is being trained (sneaky, verifier) and apply Liger kernel to the corresponding models
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
                    temperature=self.training_configs["sneaky_prover"].temperature,
                    use_ref_model=True if self.rl_config.beta != 0.0 else False,
                )
                if self.training_configs["sneaky_prover"].apply_liger_kernel
                else None
            )

        # Apply Liger kernel to verifier if specified
        elif (
            self.phase == "verifier" and self.training_configs["verifier"].apply_liger_kernel
        ):  # Verifier training mode
            _apply_liger_kernel_to_instance(self.models["verifier"])
            assert self.liger_grpo_loss is None, "Liger GRPO loss already initialized. This should not happen."

            if (
                self.training_configs["verifier"].verifier_mode == "inference_classifier"
                or self.training_configs["verifier"].verifier_mode == "inference_regressor"
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
            self.models["sneaky_prover"].train()
        elif self.phase == "verifier":
            self.models["verifier"].train()
        else:
            raise ValueError(f"Invalid phase: {self.phase}")

        self._log_memory_usage("AFTER model loading")

        # Logger info
        logger.info("=== Model Loading Summary ===")
        logger.info(f"Loaded models: {list(self.models.keys())}")
        for model_key in self.models:
            logger.info(f"- {model_key} model loaded from: {self.model_paths[model_key]}")
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
            logger.info(f"- Epsilon range: [{self.rl_config.epsilon_low}, {self.rl_config.epsilon_high}]")
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

    def get_model(self, key: str, prepared: bool = True) -> torch.nn.Module:
        """Returns the requested model (prepared or unprepared)."""
        if prepared and key in self.prepared_models:
            return self.prepared_models[key]
        else:
            logger.warning(f"Model {key} not prepared. Returning unprepared model.")
            if key not in self.models:
                logger.warning(f"Model {key} not found. Note keys are: {list(self.models.keys())}")
            else:
                return self.models[key]

    def get_tokenizer(self) -> AutoTokenizer:
        """Returns the tokenizer."""
        if self.tokenizer is not None:
            return self.tokenizer
        else:
            raise ValueError("Tokenizer not found. Please prepare the tokenizer first.")

    def get_ref_model(self, key: str, prepared: bool = True) -> torch.nn.Module:
        """Returns the requested reference model (prepared or unprepared)."""
        if prepared and key in self.prepared_ref_models:
            return self.prepared_ref_models[key]
        else:
            logger.warning(f"Reference model {key} not prepared. Returning unprepared model.")
            if key not in self.ref_models:
                logger.warning(f"Reference model {key} not found. Note keys are: {list(self.ref_models.keys())}")
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

    def set_train_mode(self, model_key: Literal["sneaky", "verifier"], train: bool = True) -> None:
        """Sets the train mode for the specified model."""
        if train:
            self.models[model_key].train()
        else:
            self.models[model_key].eval()

    def get_model_parameters(
        self, model_key: Literal["sneaky", "verifier"], prepared: bool = True
    ) -> Iterator[torch.nn.parameter.Parameter]:
        """Returns an iterator over the parameters of the specified model."""
        if prepared and model_key in self.prepared_models:
            return self.prepared_models[model_key].parameters()
        else:
            return self.models[model_key].parameters()

    def prepare_verifier(self, model_init_kwargs: dict[str, Any] = {}) -> None:
        """Prepares the verifier model based on the verifier mode."""
        if self.training_configs["verifier"].verifier_mode == "regressor":
            # Regressor --> AutoModelforSequenceClassification with 1 output neuron (num_labels = 1)
            # Set up a config with pad_token_id = eos_token_id
            model_init_kwargs["pad_token_id"] = 151643  # TODO: This is a hack. We should fix this.
            self.models["verifier"] = AutoModelForSequenceClassification.from_pretrained(
                self.model_paths["verifier"],
                num_labels=1,
                **model_init_kwargs,
            )
        elif self.training_configs["verifier"].verifier_mode == "classifier":
            # Classifier --> Language model that outputs <verdict>...</verdict> token (binary scoring 0-1)
            self.models["verifier"] = AutoModelForCausalLM.from_pretrained(
                self.model_paths["verifier"], **model_init_kwargs
            )
        elif self.training_configs["verifier"].verifier_mode == "inference_classifier":
            # Inference-time classifier --> Language model that outputs <verdict>...</verdict> token (binary classification) but with chain-of-thought reasoning steps before:
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
            self.models["verifier"].verifier_head = torch.nn.Linear(self.models["verifier"].config.hidden_size, 1)
        else:
            raise ValueError(f"Invalid verifier mode: {self.training_configs['verifier'].verifier_mode}")

        self.models["verifier"] = self.models["verifier"].to(
            self.accelerator_manager.get_state_property("device", "verifier")
        )

    # NOTE: This function is called when the phase changes - i.e., when the state tracker is updated (end of verifier training, end of provers training, ...)
    # Acts as a reset for the model manager
    def load_models(self):
        """Initializes the model manager (loads models, prepares models). Called when the phase changes or phase-specific training starts. Acts as a reset for the model manager."""
        self.phase = (
            self.global_phase_callback()
        )  # NOTE: Has to be called **after** the state tracker has been incremented

        logger.info(f"Loading models for phase: {self.phase}")
        logger.info(f"Current model keys before loading: {list(self.models.keys())}")

        self._load_models()  # NOTE: Wonnky, fix later

        logger.info(f"Model keys after loading: {list(self.models.keys())}")
        logger.info(f"Prepared model keys: {list(self.prepared_models.keys())}")
        logger.info(f"Reference model keys: {list(self.ref_models.keys())}")
        logger.info(f"Prepared reference model keys: {list(self.prepared_ref_models.keys())}")
