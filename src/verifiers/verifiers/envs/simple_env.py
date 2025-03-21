import random
from typing import Any
from collections.abc import Sequence

from datasets import Dataset

from ..imports import LLM, SamplingParams  # type: ignore
from verifiers.envs.environment import Environment


class SimpleEnv(Environment):
    def __init__(
        self,
        system_prompt: str = "",
        few_shot: list[dict[str, str]] = [],
        sampling_args: dict[str, Any] = {},
        **kwargs
    ):
        super().__init__(**kwargs)
        self.system_prompt = system_prompt
        self.few_shot = few_shot
        self.sampling_args = {
            "skip_special_tokens": False,
            "spaces_between_special_tokens": False,
            "n": 1,
        }
        self.sampling_args.update(sampling_args)

    def get_dataset(self, **kwargs: Any) -> Dataset | None:
        pass

    def get_eval_dataset(self, **kwargs: Any) -> Dataset | None:
        pass

    def format_prompt(
        self, prompt: str, fewshot_prob: float = 1.0
    ) -> list[dict[str, str]]:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        if self.few_shot and random.random() < fewshot_prob:
            messages.extend(self.few_shot)
        messages.append({"role": "user", "content": prompt})
        return messages

    def generate(
        self,
        prompts: list[list[dict[str, Any]]],
        llm: LLM,
        sampling_params: SamplingParams,
        **kwargs: Any
    ) -> dict[str, list[Sequence[int]] | list[str] | list[list[dict[str, Any]]]]:

        custom_sp = sampling_params.clone()
        for k, v in self.sampling_args.items():
            setattr(custom_sp, k, v)
        states = [
            {
                "messages": m,
                "prompt_ids": [],
                "completion_ids": [],
                "completion_mask": [],
            }
            for m in prompts
        ]

        # get completions
        completions = llm.chat(prompts, sampling_params=custom_sp, use_tqdm=False)  # type: ignore
        for i, completion in enumerate(completions):
            states[i]["messages"].append(
                {"role": "assistant", "content": completion.outputs[0].text}
            )
            states[i]["prompt_ids"] = list(completion.prompt_token_ids)  # type: ignore
            states[i]["completion_ids"] = list(completion.outputs[0].token_ids)
            states[i]["completion_mask"] = [1] * len(states[i]["completion_ids"])

        output = {
            "ids": [states[i]["completion_ids"] for i in range(len(states))],
            "messages": [states[i]["messages"][-1:] for i in range(len(states))],
            "mask": [states[i]["completion_mask"] for i in range(len(states))],
        }
        return output
