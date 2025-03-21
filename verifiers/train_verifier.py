# script.py
import verifiers as vf
from verifiers.prompts import VERIFY_PROMPT
import os
# wandb key
os.environ["WANDB_API_KEY"] = "ff1e147079fa5e0c217c4bcce87c2fb30fd9ef25"



# model_name = "/data/huggingface/models--Qwen--Qwen2.5-7B-Instruct/snapshots/bb46c15ee4bb56c5b63245ef50fd7637234d6f75"
small_qwen = "/data/huggingface/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
medium_qwen = "/data/huggingface/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"

model_name = small_qwen

model, tokenizer = vf.get_model_and_tokenizer(model_name)

vf_env = vf.VerifEnv(
    dataset="jvelja/math-tampered-length-consistent-wrapped-level",
    system_prompt=VERIFY_PROMPT,
    fields=["problem", "solution"],
)


dataset = vf_env.get_dataset(curriculum=True)
eval_dataset = vf_env.get_eval_dataset(n=100)
rubric = vf_env.get_rubric()

# notable defaults: lr = 1e-6, max_grad_norm = 0.01, constant lr 10 warmup steps, 1024 tokens in+out
run_name = "verification_" + model_name.split("--")[-1].split("/")[0].lower() + "_newP"
training_args = vf.get_default_grpo_config(
    run_name=run_name,
    num_gpus=4
)
# # rollouts per prompt
# training_args.num_generations = 6
# # minibatch size per GPU ( bs 6 * 4 gpus / 6 rollouts -> 4 prompts per batch)
# training_args.per_device_train_batch_size = 4
# # batches to accumulate (4 prompts * 4 -> 16 prompts per global batch)
# training_args.gradient_accumulation_steps = 4
# # steps per global batch (1 on-policy, 1 off-policy)
# training_args.num_iterations = 2
# # no ref model
# training_args.beta = 0.04

# medium qwen
# # batch size
# # rollouts per prompt
# training_args.num_generations = 7
# # minibatch size per GPU ( bs 16 * 7 gpus / 7 rollouts -> 16 prompts per batch)
# training_args.per_device_train_batch_size = 8
# # batches to accumulate (16 prompts * 4 -> 64 prompts per global batch)
# training_args.gradient_accumulation_steps = 4
# # steps per global batch (1 on-policy, 1 off-policy)
# training_args.num_iterations = 2
# # no ref model
# training_args.beta = 0.04
# # evals
# training_args.eval_strategy = "steps"
# training_args.eval_on_start = True
# training_args.eval_steps = 100
# training_args.per_device_eval_batch_size = 16
# training_args.eval_accumulation_steps = 1

# batch size
training_args.per_device_train_batch_size = 16
# rollouts per prompt
training_args.num_generations = 16
# minibatch size per GPU ( bs * 7 gpus / 16 rollouts -> 16 prompts per batch)
# batches to accumulate (16 prompts * 4 -> 64 prompts per global batch)
training_args.gradient_accumulation_steps = 4
# steps per global batch (1 on-policy, 1 off-policy)
training_args.num_iterations = 2
# no ref model
training_args.beta = 0.04
# evals
training_args.eval_strategy = "steps"
training_args.eval_on_start = True
training_args.eval_steps = 100
training_args.per_device_eval_batch_size = 16
training_args.eval_accumulation_steps = 1


trainer = vf.GRPOEnvTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=rubric,
    env=vf_env,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=eval_dataset,
    curriculum=True,
)

trainer.train()
# Save model locally
trainer.model.save_pretrained(f"models/{small_qwen.split('--')[-1].split('/')[0].lower()}_verifier_curriculum_newP")
tokenizer.save_pretrained(f"models/{small_qwen.split('--')[-1].split('/')[0].lower()}_verifier_curriculum_newP")

# Save model to HF
trainer.push_to_hub(f"jvelja/{small_qwen.split('--')[-1].split('/')[0].lower()}_verifier_curriculum_newP")
