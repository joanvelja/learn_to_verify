# # pvgtrainer.py
# # This file is a modified version of the Trainer class in the transformers library.
# # It is used to train two models adversarially while correctly managing GPU allocation and inference processes.


# ## Imports
# import contextlib
# import importlib.metadata
# import inspect
# import os
# from typing import TYPE_CHECKING

# from transformers import (
#     Trainer,
#     PreTrainedModel,
#     PreTrainedTokenizerBase,
#     TrainingArguments,
#     TrainerCallback,
# )

# # Integrations must be imported before ML frameworks:
# # isort: off
# from transformers.integrations import (
#     get_reporting_integration_callbacks,
# )

# # isort: on

# import numpy as np
# import torch
# from packaging import version
# from torch import nn
# from torch.utils.data import (
#     RandomSampler,
# )

# from transformers.integrations.deepspeed import (
#     is_deepspeed_available,
# )
# from transformers.trainer_callback import (
#     DefaultFlowCallback,
#     ProgressCallback,
# )
# from transformers.utils import (
#     XLA_FSDPV2_MIN_VERSION,
#     is_accelerate_available,
#     is_apex_available,
#     is_datasets_available,
#     is_in_notebook,
#     is_peft_available,
#     is_safetensors_available,
#     is_liger_kernel_available,
#     is_sagemaker_mp_enabled,
#     is_torch_xla_available,
#     logging,
# )

# from transformers.trainer_callback import (
#     CallbackHandler,
#     PrinterCallback,
# )

# DEFAULT_CALLBACKS = [DefaultFlowCallback]
# DEFAULT_PROGRESS_CALLBACK = ProgressCallback


# if is_in_notebook():
#     from transformers.utils.notebook import NotebookProgressCallback

#     DEFAULT_PROGRESS_CALLBACK = NotebookProgressCallback

# if is_apex_available():
#     pass

# if is_datasets_available():
#     pass

# if is_torch_xla_available():
#     from torch_xla import __version__ as XLA_VERSION

#     IS_XLA_FSDPV2_POST_2_2 = version.parse(XLA_VERSION) >= version.parse(
#         XLA_FSDPV2_MIN_VERSION
#     )
#     if IS_XLA_FSDPV2_POST_2_2:
#         pass
# else:
#     IS_XLA_FSDPV2_POST_2_2 = False


# if is_sagemaker_mp_enabled():
#     from smdistributed.modelparallel import __version__ as SMP_VERSION

#     IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

# else:
#     IS_SAGEMAKER_MP_POST_1_10 = False


# if is_safetensors_available():
#     pass

# if is_peft_available():
#     from peft import PeftModel


# if is_accelerate_available():
#     from accelerate import __version__ as accelerate_version

#     DATA_SAMPLERS = [RandomSampler]
#     if version.parse(accelerate_version) > version.parse("1.3.0"):
#         pass
#     if version.parse(accelerate_version) > version.parse("0.23.0"):
#         from accelerate.data_loader import SeedableRandomSampler

#         DATA_SAMPLERS += [SeedableRandomSampler]

#     if is_deepspeed_available():
#         pass

# if is_accelerate_available("0.28.0"):
#     pass


# def _is_peft_model(model):
#     if is_peft_available():
#         classes_to_check = (PeftModel,) if is_peft_available() else ()
#         # Here we also check if the model is an instance of `PeftMixedModel` introduced in peft>=0.7.0: https://github.com/huggingface/transformers/pull/28321
#         if version.parse(importlib.metadata.version("peft")) >= version.parse("0.7.0"):
#             from peft import PeftMixedModel

#             classes_to_check = (*classes_to_check, PeftMixedModel)
#         return isinstance(model, classes_to_check)
#     return False


# def safe_globals():
#     # Starting from version 2.4 PyTorch introduces a check for the objects loaded
#     # with torch.load(weights_only=True). Starting from 2.6 weights_only=True becomes
#     # a default and requires allowlisting of objects being loaded.
#     # See: https://github.com/pytorch/pytorch/pull/137602
#     # See: https://pytorch.org/docs/stable/notes/serialization.html#torch.serialization.add_safe_globals
#     # See: https://github.com/huggingface/accelerate/pull/3036
#     if version.parse(torch.__version__).release < version.parse("2.6").release:
#         return contextlib.nullcontext()

#     np_core = (
#         np._core if version.parse(np.__version__) >= version.parse("2.0.0") else np.core
#     )
#     allowlist = [np_core.multiarray._reconstruct, np.ndarray, np.dtype]
#     # numpy >1.25 defines numpy.dtypes.UInt32DType, but below works for
#     # all versions of numpy
#     allowlist += [type(np.dtype(np.uint32))]

#     return torch.serialization.safe_globals(allowlist)


# if TYPE_CHECKING:

#     if is_datasets_available():
#         pass

# logger = logging.get_logger(__name__)


# # Name of the files used for checkpointing
# TRAINING_ARGS_NAME = "training_args.bin"
# TRAINER_STATE_NAME = "trainer_state.json"
# OPTIMIZER_NAME = "optimizer.pt"
# SCALER_NAME = "scaler.pt"
# OPTIMIZER_NAME_BIN = "optimizer.bin"
# SCHEDULER_NAME = "scheduler.pt"
# FSDP_MODEL_NAME = "pytorch_model_fsdp"


# class PVGTrainer(Trainer):
#     def __init__(
#         self,
#         model: PreTrainedModel | nn.Module = None,
#         model2: PreTrainedModel | nn.Module = None,
#         args: TrainingArguments = None,
#         args2: TrainingArguments = None,
#         processing_class: PreTrainedTokenizerBase | None = None,
#         processing_class2: PreTrainedTokenizerBase | None = None,
#         callbacks: list[TrainerCallback] | None = None,
#         **kwargs,
#     ):
#         logger.info("Initializing PVGTrainer...")
#         logger.debug(f"Initial models: model1={type(model)}, model2={type(model2)}")
#         logger.debug(f"Training arguments: args1={args}, args2={args2}")

#         try:
#             logger.debug("Attempting to initialize parent Trainer class...")
#             super().__init__(model, args, processing_class, **kwargs)
#             logger.debug("Parent Trainer initialization successful")
#         except Exception as e:
#             logger.error(f"Failed to initialize parent Trainer: {str(e)}")
#             raise

#         logger.debug("Setting up model2...")
#         self.model2 = model2
#         self.args2 = args2

#         logger.debug("Checking model2 parallelization status...")
#         if getattr(model2, "is_parallelizable", False) and getattr(
#             model2, "model_parallel", False
#         ):
#             self.is_model2_parallel = True
#             logger.debug("Model2 is parallel")
#         else:
#             self.is_model2_parallel = False
#             logger.debug("Model2 is not parallel")

#         logger.debug("Checking liger kernel configuration...")
#         if self.args2.use_liger_kernel:
#             logger.debug("Liger kernel requested")
#             if is_liger_kernel_available():
#                 logger.debug("Liger kernel is available, attempting to apply...")
#                 try:
#                     if isinstance(model2, PreTrainedModel):
#                         _apply_liger_kernel_to_instance(model=model2)
#                         logger.debug("Applied liger kernel to PreTrainedModel")
#                     elif hasattr(model2, "get_base_model") and isinstance(
#                         model2.get_base_model(), PreTrainedModel
#                     ):
#                         _apply_liger_kernel_to_instance(model=model2.get_base_model())
#                         logger.debug("Applied liger kernel to base model")
#                     else:
#                         logger.warning(
#                             "The model is not an instance of PreTrainedModel. No liger kernels will be applied."
#                         )
#                 except Exception as e:
#                     logger.error(f"Failed to apply liger kernel: {str(e)}")
#                     raise
#             else:
#                 logger.error("Liger kernel requested but not available")
#                 raise ImportError(
#                     "You have set `use_liger_kernel` to `True` but liger-kernel >= 0.3.0 is not available. "
#                     "Please install it with `pip install liger-kernel`"
#                 )

#         logger.debug("Setting up model placement...")
#         self.place_model2_on_device = args2.place_model_on_device
#         if (
#             self.is_model2_parallel
#             or self.is_deepspeed_enabled
#             or ((args2.fp16_full_eval or args2.bf16_full_eval) and not args2.do_train)
#             or self.is_fsdp_xla_enabled
#             or self.is_fsdp_enabled
#         ):
#             self.place_model2_on_device = False
#             logger.debug(
#                 "Model2 placement on device disabled due to special configurations"
#             )

#         logger.debug("Setting up processing class...")
#         try:
#             self.processing_class2 = (
#                 self.processing_class
#                 if check_model_equivalence(self.model, self.model2)
#                 else processing_class2
#             )
#             logger.debug(f"Processing class2 set to: {type(self.processing_class2)}")
#         except Exception as e:
#             logger.error(f"Failed to set processing class2: {str(e)}")
#             raise

#         logger.debug("Setting up model wrapping...")
#         self.model2_wrapped = model2
#         self.model2 = model2

#         logger.debug("Checking model2 forward parameters...")
#         try:
#             unwrapped_model2 = self.accelerator.unwrap_model(model2)
#             model2_forward = (
#                 unwrapped_model2.forward
#                 if not _is_peft_model(unwrapped_model2)
#                 else unwrapped_model2.get_base_model().forward
#             )
#             forward_params2 = inspect.signature(model2_forward).parameters
#             logger.debug(f"Model2 forward parameters: {list(forward_params2.keys())}")
#         except Exception as e:
#             logger.error(f"Failed to inspect model2 forward parameters: {str(e)}")
#             raise

#         logger.debug("Checking loss kwargs acceptance...")
#         if hasattr(model2, "accepts_loss_kwargs"):
#             self.model2_accepts_loss_kwargs = model2.accepts_loss_kwargs
#             logger.debug(
#                 f"Model2 explicitly accepts loss kwargs: {self.model2_accepts_loss_kwargs}"
#             )
#         else:
#             self.model2_accepts_loss_kwargs = any(
#                 k.kind == inspect.Parameter.VAR_KEYWORD
#                 for k in forward_params2.values()
#             )
#             logger.debug(
#                 f"Model2 implicitly accepts loss kwargs: {self.model2_accepts_loss_kwargs}"
#             )

#         logger.debug("Setting up optimizer and scheduler...")
#         try:
#             self.optimizer2, self.lr_scheduler2 = self.compile_optimizer_and_scheduler(
#                 (args2.optimizer, args2.lr_scheduler)
#             )
#             logger.debug(
#                 f"Optimizer2 type: {type(self.optimizer2)}, Scheduler2 type: {type(self.lr_scheduler2)}"
#             )
#         except Exception as e:
#             logger.error(f"Failed to compile optimizer and scheduler: {str(e)}")
#             raise

#         self.optimizer_cls_and_kwargs2 = None

#         logger.debug("Setting up callbacks...")
#         try:
#             default_callbacks = DEFAULT_CALLBACKS + get_reporting_integration_callbacks(
#                 self.args2.report_to
#             )
#             callbacks = (
#                 default_callbacks
#                 if callbacks is None
#                 else default_callbacks + callbacks
#             )
#             self.callback_handler = CallbackHandler(
#                 callbacks,
#                 self.model2,
#                 self.processing_class2,
#                 self.optimizer2,
#                 self.lr_scheduler2,
#             )
#             logger.debug(
#                 f"Callback handler initialized with {len(callbacks)} callbacks"
#             )
#         except Exception as e:
#             logger.error(f"Failed to setup callbacks: {str(e)}")
#             raise

#         self.add_callback(
#             PrinterCallback if self.args2.disable_tqdm else DEFAULT_PROGRESS_CALLBACK
#         )

#         self._loggers_initialized = False

#         logger.debug("Setting up repository configuration...")
#         self.hub_model_id = None
#         if self.args2.push_to_hub:
#             logger.debug("Push to hub enabled, initializing repo...")
#             self.init_hf_repo()
#         if self.args2.should_save:
#             logger.debug(f"Creating output directory: {self.args2.output_dir}")
#             os.makedirs(self.args2.output_dir, exist_ok=True)

#         logger.debug("Checking precision settings...")
#         if (args2.fp16 or args2.bf16) and args2.half_precision_backend == "auto":
#             if args2.device == torch.device("cpu"):
#                 if args2.fp16:
#                     if not is_torch_greater_or_equal_than_2_3:
#                         logger.error("FP16 not supported on CPU")
#                         raise ValueError(
#                             "Tried to use `fp16` but it is not supported on cpu"
#                         )
#                 else:
#                     args2.half_precision_backend = "cpu_amp"
#             logger.info(f"Using {args2.half_precision_backend} half precision backend")

#         logger.debug("Setting up precision backend...")
#         if (args2.fp16 or args2.bf16) and not (
#             self.is_deepspeed_enabled or is_sagemaker_mp_enabled()
#         ):
#             if args2.half_precision_backend == "cpu_amp":
#                 self.use_cpu_amp = True
#                 self.amp_dtype = torch.bfloat16
#                 logger.debug("Using CPU AMP with bfloat16")
#             elif args2.half_precision_backend == "apex":
#                 if not is_apex_available():
#                     logger.error("APEX requested but not available")
#                     raise ImportError(
#                         "Using FP16 with APEX but APEX is not installed, please refer to"
#                         " https://www.github.com/nvidia/apex."
#                     )
#                 self.use_apex = True
#                 logger.debug("Using APEX for mixed precision")

#         logger.debug("Checking torch compile settings...")
#         if args2.torch_compile and not is_torch_compile_available():
#             logger.error("torch.compile requested but not available")
#             raise RuntimeError("Using torch.compile requires PyTorch 2.0 or higher.")

#         logger.info("PVGTrainer initialization completed successfully")

#     def compile_optimizer_and_scheduler(
#         self,
#         optimizer: (
#             tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] | None
#         ) = None,
#     ):
#         if optimizer != (None, None):
#             opt, lr_scheduler = optimizer
#             optimizer = type(opt)(self.model2.parameters(), **opt.defaults)
#             # Reuse the provided scheduler, but associate it with the new optimizer
#             lr_scheduler = type(lr_scheduler)(optimizer, **lr_scheduler.defaults)

#         else:  # Fall back to default optimizer and scheduler
#             optimizer = torch.optim.AdamW(
#                 self.model2.parameters(), lr=1e-4, weight_decay=0.01
#             )
#             lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
#                 optimizer, lr_lambda=lambda step: 1
#             )

#         return optimizer, lr_scheduler
