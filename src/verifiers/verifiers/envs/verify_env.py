from typing import Any
from datasets import Dataset
from trl.trainer.grpo_trainer import RewardFunc

from verifiers.envs.simple_env import SimpleEnv
from verifiers.parsers import XMLParser
from verifiers.rubrics import VerifierRubric
from verifiers.prompts import VERIFY_PROMPT
from verifiers.utils import preprocess_dataset

import pandas as pd


class VerifEnv(SimpleEnv):
    def __init__(
        self,
        #  dataset: str = "jvelja/math-tampered-length-consistent-wrapped",
        dataset: str = "jvelja/math-tampered-length-consistent-wrapped-level",  # preserving difficulty information
        system_prompt: str = VERIFY_PROMPT,
        fields: list[str | tuple[str, ...]] = ["reasoning", "answer"],
        **kwargs
    ):
        super().__init__(system_prompt=system_prompt, **kwargs)
        self.parser = XMLParser(fields=fields)
        self.dataset_name = dataset
        self.dataset = preprocess_dataset(
            dataset_name=dataset, split="train", system_prompt=system_prompt
        )
        self.eval_dataset = None
        self.rubric = VerifierRubric()

    def get_dataset(self, **kwargs: Any) -> Dataset:
        """
        Returns the training dataset.
        If the dataset is MATH (or MATH derived), it can be loaded ordered by difficulty.
        """
        if kwargs.get("curriculum", False):
            return self.shuffle_within_level()
        return self.dataset

    def get_eval_dataset(self, n: int = -1, **kwargs: Any) -> Dataset | None:
        if self.eval_dataset is None:
            self.eval_dataset = preprocess_dataset(
                dataset_name=self.dataset_name,
                split="test",
                system_prompt=self.system_prompt,
            )
        if n > 0:
            return self.eval_dataset.shuffle().select(range(n))  # type: ignore
        return self.eval_dataset

    def get_rubric(self, **kwargs: Any) -> list[RewardFunc]:
        return self.rubric.get_reward_funcs()

    def shuffle_within_level(self):
        # Convert the dataset to a pandas DataFrame for easier manipulation
        df = pd.DataFrame(self.dataset)

        # Group the DataFrame by 'level'
        grouped = df.groupby("level")

        # Shuffle the order of examples within each level
        shuffled_groups = []
        for level, group in grouped:
            shuffled_examples = group.sample(frac=1).to_dict("records")  # Shuffle rows
            shuffled_groups.extend(shuffled_examples)

        return Dataset.from_list(shuffled_groups)
