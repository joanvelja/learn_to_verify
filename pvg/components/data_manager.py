# pvg/components/data_manager.py

# DataManager
# Responsibility: Loads the tokenizer, loads and processes the dataset (AppsDataset), creates the appropriate samplers (RepeatRandomSampler), creates DataLoader instances, and prepares dataloaders using AcceleratorManager. Provides methods to get batches formatted correctly for different training phases/modes.


# From the PVG paper:
# First, we augmented the GSM [(Cobbe et al., 2021)](#page-17-4) dataset using 100k synthetically generated and validated datapoints from ChatGPT, similar to the method in [Liu et al.](#page-18-15) [(2023)](#page-18-15). We made the dataset larger so as not to be bottlenecked by sample efficiency in order to focus on the training dynamics. We validated that the the original test set accuracy is not impacted by using real vs. synthetic data. Next, we randomly partitioned the training dataset D into two equally-sized subsets Dπ and DV that are used for training the prover π and the verifier V respectively. This way the verifier and the prover are never optimized on the same prompt.

# The training proceeds in multiple rounds, and each round the verifier training phase precedes the prover training phase. The only change between rounds is the data mixture used to train the verifier; no other state, such as model weights, is carried over between rounds.


import logging
from typing import Any, Callable, Literal

from datasets import DatasetDict
from torch.utils.data import DataLoader, Dataset, Sampler
from transformers import AutoTokenizer, PreTrainedTokenizer

from pvg.components.accelerator_manager import AcceleratorManager
from pvg.config.args import DatasetArgs
from pvg.data.dataset import AppsDataset
from pvg.data.rep_sampler import RepeatRandomSampler

logger = logging.getLogger(f"pvg.{__name__}")  # Get a child logger


class DataManager:
    """
    Loads the tokenizer, loads and processes the dataset (AppsDataset), creates the appropriate samplers (RepeatRandomSampler), creates DataLoader instances, and prepares dataloaders using AcceleratorManager. Provides methods to get batches formatted correctly for different training phases/modes.
    """

    def __init__(
        self,
        accelerator_manager: AcceleratorManager,
        dataset_config: DatasetArgs,
        sampler_args: dict[str, Any],
        seed: int,
        global_phase_callback: Callable[
            [], Literal["verifier", "provers"]
        ],  # Needed to assess what datamix to prepare and return upon request
        verifier_mode: Literal["regressor", "classifier", "inference_classifier", "inference_regressor"],
    ) -> None:
        self.accelerator_manager = accelerator_manager
        self.dataset_config = dataset_config
        self.sampler_args = sampler_args
        self.seed = seed
        self.verifier_mode = verifier_mode  # Makes a difference in how we collate the data for the verifier
        self.verifier_datamix = None  # This is the datamix
        self.prover_train_dataset: AppsDataset | None = None
        self.prover_eval_dataset: AppsDataset | None = None
        self.verifier_train_dataset: AppsDataset | None = None
        self.verifier_eval_dataset: AppsDataset | None = None

        # Legacy: HF Hub paths (unused in high-throughput local mode)
        self.hf_repo_path = f'jvelja/{self.dataset_config.dataset_name.split("/")[1]}_{self.dataset_config.dataset_size}-verifier-{self.verifier_mode}'
        self.global_phase_callback: Callable[[], Literal["verifier", "provers"]] = global_phase_callback
        # Load tokenizer
        self.tokenizer: PreTrainedTokenizer = self.load_tokenizer()

    def load_tokenizer(self) -> PreTrainedTokenizer:
        """Loads the tokenizer based on the dataset config."""
        tokenizer = AutoTokenizer.from_pretrained(self.dataset_config.tokenizer_name_or_path)
        # Make sure that the tokenizer has a pad_token_id
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
            logger.warning(f"Tokenizer {tokenizer} has no pad_token_id. Setting it to {tokenizer.eos_token_id}")
        return tokenizer

    def load_datasets(self) -> None:
        # TODO: For now, we only have APPS dataset
        # Load full datasets
        full_train_dataset = AppsDataset(
            dataset_name=self.dataset_config.dataset_name,
            dataset_size=self.dataset_config.dataset_size,
            tokenizer=self.tokenizer,
            split="train",
            num_samples=self.dataset_config.train_num_samples,
        )
        full_train_dataset.shuffle()  # To ensure that subset datasets are not biased towards difficulty!
        full_eval_dataset = AppsDataset(
            dataset_name=self.dataset_config.dataset_name,
            dataset_size=self.dataset_config.dataset_size,
            tokenizer=self.tokenizer,
            split="test",
            num_samples=self.dataset_config.eval_num_samples,
        )

        # Split datasets into prover and verifier halves
        train_len = len(full_train_dataset)
        eval_len = len(full_eval_dataset)

        # Create prover datasets (first half) using direct slicing (keep AppsDataset wrappers)
        prover_train_indices = list(range(0, train_len // 2))
        prover_eval_indices = list(range(0, eval_len // 2))
        self.prover_train_dataset = full_train_dataset.select(prover_train_indices)
        self.prover_eval_dataset = full_eval_dataset.select(prover_eval_indices)

        # Create verifier datasets (second half) using direct slicing
        verifier_train_indices = list(range(train_len // 2, train_len))
        verifier_eval_indices = list(range(eval_len // 2, eval_len))
        # Keep AppsDataset objects for verifier as well (VerifierPhaseStrategy builds its own VerifierDataset)
        self.verifier_train_dataset = full_train_dataset.select(verifier_train_indices)
        self.verifier_eval_dataset = full_eval_dataset.select(verifier_eval_indices)

    def create_dataloaders(self) -> None:
        """
        Creates internal DataLoader instances. This might involve:
            - A loader for verifier data with a collator for pairs (collate_fn=VerifierPairCollator).
            - A loader for verifier data with a collator for single examples (collate_fn=VerifierSingleCollator).
            - A loader for prover prompts (collate_fn=ProverPromptCollator).
            - Uses RepeatRandomSampler for prover prompts if needed.
        """

        # NOTE: Dataloaders here are odd entities:
        # - For verifier training, a dataloader is needed for sampling batches to feed the vllm prover instances that then generate completions --> Actual dataset!
        #    - Thus, this is technically not a dataloader, but a dataset!
        # - For prover training, the dataloader maintains its "natural" meaning, batches from the dataloader are what the provers will see.

        # NOTE: This necessitates a different approach when initializing the optimizer manager during verifier training.

        def prover_data_collator(features):
            return features

        def verifier_data_collator(features):
            return features

        # Provers dataloaders: no collator, sampler from GRPO
        self.prover_train_dataloader = DataLoader(
            self.prover_train_dataset,
            batch_size=self.sampler_args["per_device_train_batch_size"],
            sampler=self._get_train_sampler(),
            drop_last=True,
            collate_fn=prover_data_collator,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4,
        )

        # Select a subset of datapoints for evaluation
        self.prover_eval_dataset = self.prover_eval_dataset.select(list(range(192)))

        self.prover_eval_dataloader = DataLoader(
            self.prover_eval_dataset,
            batch_size=self.sampler_args["per_device_eval_batch_size"],
            sampler=self._get_eval_sampler(self.prover_eval_dataset),
            drop_last=True,
            collate_fn=prover_data_collator,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4,
        )

        self.dataloaders = {
            "provers": {
                "sneaky_prover": {
                    "train_dataloader": self.prover_train_dataloader,
                    "eval_dataloader": self.prover_eval_dataloader,
                },
            }
        }

        regressor_classifier_train_dataloader = DataLoader(
            self.verifier_train_dataset,
            batch_size=self.sampler_args["per_device_train_batch_size"],
            drop_last=True,
            collate_fn=verifier_data_collator,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4,
        )
        regressor_classifier_eval_dataloader = DataLoader(
            self.verifier_eval_dataset,
            batch_size=self.sampler_args["per_device_eval_batch_size"],
            drop_last=True,
            collate_fn=verifier_data_collator,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4,
        )

        inference_classifier_train_dataloader = DataLoader(
            self.verifier_train_dataset,
            batch_size=self.sampler_args["per_device_train_batch_size"],
            drop_last=True,
            sampler=self._get_train_sampler(),
            collate_fn=verifier_data_collator,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4,
        )
        inference_classifier_eval_dataloader = DataLoader(
            self.verifier_eval_dataset,
            batch_size=self.sampler_args["per_device_eval_batch_size"],
            drop_last=True,
            sampler=self._get_eval_sampler(self.verifier_eval_dataset),
            collate_fn=verifier_data_collator,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4,
        )

        # TODO: Add verifier "dataloaders"
        self.dataloaders["verifier"] = {
            "regressor": {
                "train_dataloader": regressor_classifier_train_dataloader,
                "eval_dataloader": regressor_classifier_eval_dataloader,
            },
            "classifier": {
                "train_dataloader": regressor_classifier_train_dataloader,
                "eval_dataloader": regressor_classifier_eval_dataloader,
            },
            "inference_classifier": {
                "train_dataloader": inference_classifier_train_dataloader,
                "eval_dataloader": inference_classifier_eval_dataloader,
            },
            "inference_regressor": {
                "train_dataloader": regressor_classifier_train_dataloader,
                "eval_dataloader": regressor_classifier_eval_dataloader,
            },
        }

    def get_tokenizer(self) -> PreTrainedTokenizer:
        """Returns the tokenizer."""
        return self.tokenizer

    def _get_train_sampler(self) -> Sampler:
        # Returns a sampler that
        # 1. ensures each prompt is repeated across multiple processes. This guarantees that identical prompts are
        #    distributed to different GPUs, allowing rewards to be computed and normalized correctly within each prompt
        #    group. Using the same seed across processes ensures consistent prompt assignment, preventing discrepancies
        #    in group formation.
        # 2. repeats the batch multiple times to allow reusing generations across multiple updates. Refer to
        #    _prepare_inputs to see how the generations are stored and reused.

        # In the following figure, the values are the prompt indices. The first row shows the first sampled batch, the
        # second row shows the second sampled batch, and so on.
        #
        #                                     |     GPU 0     |     GPU 1     |     GPU 2    |
        #
        #               global_step   step     <───────>  num_generations=3
        #                                      <───────────> per_device_train_batch_size=4
        #                ▲   0          0      0   0   0   1   1   1   2   2   2   3   3   3  │
        #  grad_accum=3  │   0          1      4   4   4   5   5   5   6   6   6   7   7   7  │ Generate completions for each prompt
        #                ▼   0          2      8   8   8   9   9   9  10  10  10  11  11  11  │
        #
        #                    1          3      0   0   0   1   1   1   2   2   2   3   3   3  │ The sampled prompts are the same as in the first iteration
        #                    1          4      4   4   4   5   5   5   6   6   6   7   7   7  │ Reuse the completions (here, once, because num_iterations=2)
        #                    1          5      8   8   8   9   9   9  10  10  10  11  11  11  │
        #
        #                    2          6     12  12  12  13  13  13  14  14  14  15  15  15
        #                    2          7     16  16  16  17  17  17  18  18  18  19  19  19
        #                    2          8     20  20  20  21  21  21  22  22  22  23  23  23
        #                                          ...
        effective_batch_size = (
            self.sampler_args["per_device_train_batch_size"]
            * self.accelerator_manager.get_state_property("num_processes")
            * self.sampler_args["gradient_accumulation_steps"]
        )
        logger.info(f"Effective batch size: {effective_batch_size}")

        return RepeatRandomSampler(
            data_source=self.prover_train_dataset,
            mini_repeat_count=self.sampler_args["num_generations"],
            batch_size=effective_batch_size // self.sampler_args["num_generations"],
            repeat_count=self.sampler_args["num_iterations"],
            seed=self.seed,
            lengths=getattr(self.prover_train_dataset, "lengths", None),
            sortish_block=8,
        )

    def _get_eval_sampler(self, eval_dataset: Dataset) -> Sampler:
        # See _get_train_sampler for an explanation of the sampler.
        return RepeatRandomSampler(
            data_source=eval_dataset,
            mini_repeat_count=self.sampler_args["num_generations"],
            seed=self.seed,
            lengths=getattr(eval_dataset, "lengths", None),
            sortish_block=8,
        )

    def prepare_dataloaders(self) -> None:
        """
        Calls accelerator_manager.prepare_dataloader for train/eval dataloaders. Stores prepared dataloaders. Must be called after AcceleratorManager is fully initialized.
        """
        # Based off of what we train, we call prepare_dataloader on the appropriate accelerator instance
        mode = self.verifier_mode
        for phase in ["verifier", "provers"]:
            if phase == "verifier":
                train_dataloader = self.accelerator_manager.prepare_dataloader(
                    self.dataloaders[phase][mode]["train_dataloader"], key=phase
                )
                eval_dataloader = self.accelerator_manager.prepare_dataloader(
                    self.dataloaders[phase][mode]["eval_dataloader"], key=phase
                )
                self.dataloaders[phase][mode]["train_dataloader"] = train_dataloader
                self.dataloaders[phase][mode]["eval_dataloader"] = eval_dataloader
            elif phase == "provers":
                train_dataloader = self.accelerator_manager.prepare_dataloader(
                    self.dataloaders[phase]["sneaky_prover"]["train_dataloader"],
                    key="sneaky_prover",
                )
                eval_dataloader = self.accelerator_manager.prepare_dataloader(
                    self.dataloaders[phase]["sneaky_prover"]["eval_dataloader"],
                    key="sneaky_prover",
                )

                # Set the dataloaders in the dataloaders dict
                self.dataloaders[phase]["sneaky_prover"]["train_dataloader"] = train_dataloader
                self.dataloaders[phase]["sneaky_prover"]["eval_dataloader"] = eval_dataloader

    def make_verifier_datamix(self) -> None:
        """
        Makes the verifier datamix based on the verifier mode.
        """
        pass

    def get_verifier_dataloader(
        self,
        mode: Literal["regressor", "classifier", "inference_classifier", "inference_regressor"],
    ) -> DataLoader:
        """
        Returns the verifier dataloader for the given mode.
        """
        return self.dataloaders["verifier"][mode]["train_dataloader"]

    def get_prover_dataloader(self, mode: Literal["sneaky_prover"]) -> DataLoader:
        """
        Returns the prover dataloader for the given mode.
        """
        return self.dataloaders["provers"][mode]["train_dataloader"]
