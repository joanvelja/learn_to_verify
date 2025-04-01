# from transformers import AutoModelForCausalLM, PreTrainedModel, AutoTokenizer
# from trl.extras.vllm_client import VLLMClient
# from trl import GRPOTrainer, GRPOConfig
# from datasets import load_dataset
# import torch
# from dataclasses import dataclass
# from typing import Literal
# from utils import check_model_equivalence


# @dataclass
# class VerifierConfig:
#     """
#     Config for the verifier
#     Args:
#         model: str -> Path to the model
#         type: Literal["reward", "generative", "tooled"] -> Type of verifier
#             - reward: Verifier is a reward model (i.e., an LLM with a regression head)
#             - generative: Verifier is a generative model (i.e., an LLM that can do CoT + verdict)
#             - tooled: Verifier is a tool-use model (i.e., an LLM that can use tools, e.g. run unit tests)
#     """

#     model: str
#     type: Literal[
#         "reward", "generative", "tooled"
#     ]  # I.e., the types of verifiers we support
#     reward_weights: list[float]


# def get_verifier(v_config: VerifierConfig):
#     if v_config.type == "reward":
#         # Use sequence classification model with 1 output for regression
#         # TODO: Make this more streamlined (i.e., allow to verifier to simply take a prompt as a string and handle all the inference process + extraction of the signal automatically)
#         # Particularly interested in vLLM integration here
#         raise NotImplementedError("Reward verifier not implemented")

#     elif v_config.type == "generative":
#         raise NotImplementedError("Generative verifier not implemented")
#         # TODO: Finish this, raise an error for now
#         # Missing: How to extract the signal?
#         # Make call to the verifier, get the output, extract the signal
#         # verifier = VLLMClient(host=args.vllm_server_host, port=8000) # Check params for this
#         # Ideally it should do all in one call? I.e., reward = verifier(prompt)
#         # And under the hood it should handle all the inference process + extraction of the signal

#     elif v_config.type == "tooled":
#         # TODO: Finish this, raise an error for now
#         raise NotImplementedError("Tooled verifier not implemented")


# small_qwen = "/data/huggingface/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
# medium_qwen = "/data/huggingface/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"


# class PTrainer(GRPOTrainer):
#     def __init__(
#         self,
#         p1: str | PreTrainedModel,
#         p2: str | PreTrainedModel,
#         v: str | PreTrainedModel,
#         args1: GRPOConfig,
#         args2: GRPOConfig,
#         v_config: VerifierConfig,
#         train_dataset: Dataset | IterableDataset | None = None,
#         eval_dataset: (
#             Dataset | IterableDataset | dict[str, Dataset | IterableDataset] | None
#         ) = None,
#         processing_class: PreTrainedTokenizerBase | None = None,
#         reward_processing_classes: (
#             PreTrainedTokenizerBase | list[PreTrainedTokenizerBase] | None
#         ) = None,
#         callbacks: list[TrainerCallback] | None = None,
#         optimizers: tuple[
#             torch.optim.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None
#         ] = (None, None),
#         peft_config: PeftConfig | None = None,
#         **kwargs
#     ):

#         assert isinstance(p1, str), "Pass in a model id for Prover #1"
#         assert isinstance(p2, str), "Pass in a model id for Prover #2"
#         assert isinstance(v, str), "Pass in a model id for Verifier"

#         if args1.beta == 0:
#             assert args2.beta == 0, "If beta == 0 for p1, then beta must be 0 for p2"

#         # TODO: Fix this
#         # Loophole: we need to pass stuff with use_vllm=False here to not trigger two vLLM servers
#         args1.use_vllm = False

#         super().__init__(
#             model=p1,
#             args=args1,
#             train_dataset=train_dataset,
#             processing_class=processing_class,
#             reward_funcs=reward_funcs,
#             **kwargs
#         )  # This is initializing the GRPOTrainer for p1
#         args1.use_vllm = True  # Hacky fix

#         self.model2 = (
#             p2
#             if isinstance(p2, PreTrainedModel)
#             else AutoModelForCausalLM.from_pretrained(
#                 p2,
#                 torch_dtype="bfloat16",
#                 device_map="auto",
#                 trust_remote_code=True,
#                 use_cache=False,
#             )
#         )  # This is initializing the p2 model

#         if args2.gradient_checkpointing:
#             self.model2 = self._enable_gradient_checkpointing(self.model2, args2)

#         # v should be sent to vLLM inference server
#         # TODO: Implement VLLMClient for all models

#         # super().__init__ calles the init of GRPOTrainer, which calls the init of Trainer
#         # If beta == 0 (no KL Regularization), then no reference model for p1 is loaded
#         # Given this, assert that beta == 0 for p2 too (want consistency) (Done see above)
#         # If beta != 0, then p1 reference model is loaded
#         # Given this, if p1 shares the same backbone as p2, then ref_model p1 == ref_model p2
#         # If p1 and p2 don't share the same backbone, then we need to load a separate ref_model for p2 (granted args2.beta != 0)

#         if not check_model_equivalence(
#             self.model, self.model2
#         ):  # Why does deepspeed require a non-eval model?
#             # If p1 and p2 don't share the same backbone, then we need to load a separate ref_model for p2 (granted args2.beta != 0)
#             self.ref_model2 = self.model2 if args2.beta == 0 else None
#         else:
#             self.ref_model2 = self.ref_model

#         #  # Reward functions
#         # if not isinstance(reward_funcs, list):
#         #     reward_funcs = [reward_funcs]
#         # for i, reward_func in enumerate(reward_funcs):
#         #     if isinstance(reward_func, str):
#         #         reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
#         #             reward_func, num_labels=1, **model_init_kwargs
#         #         )
#         # self.reward_funcs = reward_funcs

#         # Reward function --> Verifier! How do we handle this?
#         # Multiple ways to do this:
#         # 1. Verifier as a reward model (i.e., an LLM with classification head)
#         # llm = LLM(model="Qwen/Qwen2.5-Math-RM-72B", task="reward")
#         # (output,) = llm.encode("Hello, my name is")

#         # data = output.outputs.data
#         # print(f"Data: {data!r}")

#         # 2. Inference-time verifier (~generative RM, where we allow the LLM to do CoT + verdict)
#         # E.g.:
#         # <verification>
#         # ...
#         # </verification>
#         # <verdict>
#         # GOOD | BAD
#         # </verdict>
#         # We can take the verdict as the reward signal (i.e., 1 or -1) or even as a logit (i.e., logit(GOOD) or logit(BAD))
#         # TODO: look into most effective way to get best signal for what to use here

#         self.v = get_verifier(v_config)  # Will raise an error for now

#         self.processing_class = (
#             processing_class
#             if processing_class is not None
#             else AutoTokenizer.from_pretrained(p1, use_fast=True)
#         )

#         if not check_model_equivalence(self.model, self.model2):
#             self.processing_class2 = AutoTokenizer.from_pretrained(p2, use_fast=True)
#         else:
#             self.processing_class2 = self.processing_class

#         # Setup vllm Clients
#         # Ideally:
#         # 1. vllm client for v & p1
#         # 2. vllm client for p2
#         # Why? Because v.size << p1.size <= p2.size
#         # So we can get away with this to save memory
#         # I have already written a vllm_serve script that can hold multiple models

#         # Start vLLM server on GPU 0 only (first GPU)
#         # CUDA_VISIBLE_DEVICES=0
#         # python vllm_serve.py \
#         # --models "p1:honest,v:verifier" \
#         # --revisions "main,main" \
#         # --tensor_parallel_size 1 \
#         # --host 0.0.0.0 \
#         # --port 8000 \
#         # --gpu_memory_utilization 0.6, 0.3 \
#         # --max_model_lens "2048,4096" \
#         # --enable_prefix_cachings "True,True" & <-- Crucial to add this & to run in background!!!!

#         # CUDA_VISIBLE_DEVICES=1 (second GPU)
#         # python vllm_serve.py \
#         # --models "p2:sneaky" \
#         # --revisions "main" \
#         # --tensor_parallel_size 1 \
#         # --host 0.0.0.0 \
#         # --port 8001 \
#         # --gpu_memory_utilization 0.9 \
#         # --max_model_lens "4096" \
#         # --enable_prefix_cachings "True" & <-- Crucial to add this & to run in background!!!!

#         # This way we have 2 vllm clients running on different GPUs (if this works...)
#         # TODO: Test this out

#         # Then we can call VLLMClient(host=args.vllm_server_host, port=8000) and VLLMClient(host=args.vllm_server_host, port=8001) to get the two clients for inference

#         self.vllm_client = VLLMClient(
#             args1.vllm_server_host,
#             args1.vllm_server_port,
#             connection_timeout=args1.vllm_server_timeout,
#         )  # CHECK: How to call conditional on whether we want

#         self.vllm_client2 = VLLMClient(
#             args2.vllm_server_host,
#             args2.vllm_server_port,
#             connection_timeout=args2.vllm_server_timeout,
#         )

#         self._last_loaded_step = (
#             0  # tag to avoid useless loading during grad accumulation
#         )

#         # When using vLLM, the main process is responsible for loading the model weights. This can cause process
#         # desynchronization and seems to lead to DeepSpeed hanging during initialization. To prevent this, we
#         # synchronize all processes after vLLM has been fully initialized.
#         self.accelerator.wait_for_everyone()

#         # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
#         # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
#         # self.model_accepts_loss_kwargs to False to enable scaling.
#         self.model_accepts_loss_kwargs = False
#         self.model2_accepts_loss_kwargs = False

#         # Add tags to the model
#         self.model.add_model_tags(self._tag_names)
#         self.model2.add_model_tags(self._tag_names)

#         # TODO: Implement below
#         # if self.ref_model is not None:
#         #     if self.is_deepspeed_enabled:
#         #         self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
#         #     else:
#         #         self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

#         # if args.sync_ref_model:
#         #     self.add_callback(SyncRefModelCallback(ref_model=self.ref_model, accelerator=self.accelerator))

#         # for i, reward_func in enumerate(self.reward_funcs):
#         #     if isinstance(reward_func, PreTrainedModel):
#         #         self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

#         # self.v = v if isinstance(v, PreTrainedModel) else AutoModelForCausalLM.from_pretrained(v, **args.model_init_kwargs) # This is initializing the v model
#         # self.v.requires_grad_(False) # This is setting the v model to not require gradients (inference only)
#         # self.p1_vllm = VLLMClient(host=args.vllm_server_host, port=8000)
#         # self.p2_vllm = VLLMClient(host=args.vllm_server_host, port=8001)

#         print("Loading p2")
#         self.model2 = self.accelerator.prepare(
#             self.model2
#         )  # Will the same accelerator instance work? YES (see whether the dataset needs special handling)

#     def communicate(self, seed_msg=str):
#         self.model.eval()
#         self.model2.eval()
#         with torch.no_grad():
#             messages = [
#                 {
#                     "role": "system",
#                     "content": "Roleplay as a 50 year old california surfer dude. You are a surfer dude who is very friendly and helpful.",
#                 },
#                 {"role": "user", "content": seed_msg},
#             ]
#             inputs = self.processing_class.apply_chat_template(
#                 messages, return_tensors="pt"
#             ).to(self.model.device)
#             outputs = self.model.generate(inputs, max_new_tokens=64, do_sample=True)
#             text = (
#                 self.processing_class.decode(outputs[0], skip_special_tokens=True)
#                 .split("user")[-1]
#                 .strip()
#             )
#             messages = [
#                 {
#                     "role": "system",
#                     "content": "Roleplay as a 12 year old kid who's passionate about computers.",
#                 },
#                 {"role": "user", "content": text},
#             ]
#             inputs = self.processing_class.apply_chat_template(
#                 messages, return_tensors="pt"
#             ).to(self.model2.device)
#             outputs = self.model2.generate(inputs, max_new_tokens=128, do_sample=True)
#             text2 = (
#                 self.processing_class.decode(outputs[0], skip_special_tokens=True)
#                 .split("user")[-1]
#                 .strip()
#             )

#         return (text, text2)


# if __name__ == "__main__":
#     # Example dataset from TLDR
#     dataset = load_dataset("trl-lib/tldr", split="train")

#     # Dummy reward function: count the number of unique characters in the completions
#     def reward_num_unique_chars(completions, **kwargs):
#         return [len(set(c)) for c in completions]

#     training_args_1 = GRPOConfig(
#         output_dir=small_qwen,
#         per_device_train_batch_size=4,
#         num_generations=2,
#         bf16=True,
#         gradient_checkpointing=True,
#         logging_steps=10,
#         use_vllm=False,
#     )

#     training_args_2 = GRPOConfig(
#         output_dir=small_qwen,
#         per_device_train_batch_size=4,
#         num_generations=2,
#         bf16=True,
#         gradient_checkpointing=True,
#         logging_steps=10,
#         use_vllm=False,
#     )

#     trainer = PTrainer(
#         p1=small_qwen,
#         p2=small_qwen,
#         v=None,
#         args1=training_args_1,
#         args2=training_args_2,
#         train_dataset=dataset,
#         reward_funcs=reward_num_unique_chars,
#     )

#     print(
#         trainer.communicate(
#             "Introduce yourself briefly. Who are you and what do you do?"
#         )
#     )
