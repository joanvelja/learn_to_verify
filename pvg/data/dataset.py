# dataset.py

import logging
from logging import Logger
import os
import random
import torch
from datasets import load_dataset, Dataset as HFDataset
from torch.utils.data import Dataset, DataLoader, IterableDataset
from transformers import AutoTokenizer
from typing import Literal, Any

from pvg.data.prompts import BASE_VERIFIER_CODE, BASE_VERIFIER_MATH

logger: Logger = logging.getLogger(f"pvg.{__name__}")  # Get a child logger


# --- Dataset Class ---
class AppsDataset(Dataset):
    """
    Dataset for efficient handling of APPS with tokenization optimization.
    Loads data, tokenizes only the question column without padding, and handles caching.
    """

    def __init__(
        self,
        dataset_name: str,
        tokenizer: AutoTokenizer,
        split: str = "train",  # Specify split during initialization
        num_samples: int | None = None,  # Use None for all samples
        max_length: int | None = None,
        tokenize_column: str = "question",  # Column to tokenize
        keep_columns: list[str] = [
            "question",
            "solutions",
            "input_output",
        ],  # Columns to keep
        cache_dir: str | None = None,
        preprocessing_num_workers: int | None = None,
        min_length: int | None = None,
        truncation_strategy: str = "longest_first",
    ) -> None:
        """
        Initialize the AppsDataset.

        Args:
            dataset_name: Name of the dataset in HuggingFace hub (e.g., "codeparrot/apps").
            tokenizer: Tokenizer to use.
            split: Dataset split to load ('train', 'validation', 'test').
            num_samples: Number of samples to load (None for all).
            max_length: Maximum sequence length for truncation (None means no truncation during initial tokenization).
            tokenize_column: Name of the column to tokenize.
            keep_columns: list of columns to keep in the final dataset.
            cache_dir: Directory to cache tokenized datasets.
            preprocessing_num_workers: Number of workers for preprocessing.
            min_length: Minimum sequence length (filters shorter sequences after tokenization).
            truncation_strategy: Strategy for truncation if max_length is applied during tokenization.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.tokenize_column = tokenize_column
        self.keep_columns = keep_columns
        self.min_length = min_length
        self.truncation_strategy = truncation_strategy
        self.split = split

        logging.info(f"Loading raw dataset '{dataset_name}' split '{split}'...")
        raw_dataset = load_dataset(
            dataset_name, split=split, cache_dir=cache_dir, trust_remote_code=True
        )

        # Select subset if num_samples is specified
        if (
            num_samples is not None
            and num_samples > 0
            and num_samples < len(raw_dataset)
        ):
            logging.info(f"Selecting {num_samples} samples from the dataset.")
            self.raw_dataset = raw_dataset.select(range(num_samples))
        else:
            logging.info(f"Using all {len(raw_dataset)} samples from the dataset.")
            self.raw_dataset = raw_dataset

        # Create tokenizer-specific cache path
        tokenizer_name = tokenizer.name_or_path.replace("/", "_")
        cache_file_name = f"{dataset_name.replace('/', '_')}_{split}_{tokenizer_name}_tokenized.hf"  # Use HF dataset cache format
        cache_file_path = None
        if cache_dir:
            cache_file_path = os.path.join(cache_dir, cache_file_name)
            os.makedirs(cache_dir, exist_ok=True)

        # Check if valid cached dataset exists (using datasets library's caching)
        try:
            # Use map's caching mechanism - it's more robust
            logging.info(
                "Attempting to load tokenized dataset from cache (if available)..."
            )
            self.tokenized_dataset = self.raw_dataset.map(
                self._tokenize_function,
                batched=True,
                num_proc=preprocessing_num_workers,
                # Don't remove the keep_columns as we need them in the final dataset
                load_from_cache_file=True,  # Enable caching
                cache_file_name=cache_file_path,  # Specify cache file hint
                desc=f"Tokenizing {split} dataset",
            )
            logging.info(
                "Tokenized dataset loaded successfully (from cache or newly processed)."
            )

        except Exception as e:
            logging.error(
                f"Error during tokenization or cache loading: {e}", exc_info=True
            )
            logging.warning(
                "Proceeding without caching or retrying tokenization without explicit cache file path."
            )
            # Fallback: Tokenize without explicit cache file path if loading failed
            self.tokenized_dataset = self.raw_dataset.map(
                self._tokenize_function,
                batched=True,
                num_proc=preprocessing_num_workers,
                # Don't remove the keep_columns
                load_from_cache_file=True,  # Still try to use implicit caching
                desc=f"Tokenizing {split} dataset (fallback)",
            )

        # Filter by length if min_length is specified
        if self.min_length is not None and self.min_length > 0:
            original_size = len(self.tokenized_dataset)
            logging.info(
                f"Filtering sequences shorter than {self.min_length} tokens..."
            )
            self.tokenized_dataset = self.tokenized_dataset.filter(
                lambda example: len(example["input_ids"]) >= self.min_length,
                num_proc=preprocessing_num_workers,
                desc="Filtering short sequences",
            )
            logging.info(
                f"Filtered dataset from {original_size} to {len(self.tokenized_dataset)} samples."
            )

        logging.info(
            f"Dataset initialization complete for split '{split}'. Size: {len(self.tokenized_dataset)}"
        )

    def _tokenize_function(self, examples):
        """Tokenization logic applied only to the question column."""
        # Tokenize without padding (padding done dynamically in collator)
        # Only truncate if max_length is specified during dataset init
        truncation = bool(self.max_length)

        # Only tokenize the specified column
        tokenized_output = self.tokenizer(
            examples[self.tokenize_column],
            truncation=truncation,
            max_length=self.max_length,
            # No padding here!
            return_attention_mask=True,  # Keep attention mask
            return_token_type_ids=False,  # Not needed for decoder-only models
        )

        # Copy the tokenization results to the output
        result = {}

        # Add tokenization outputs (input_ids, attention_mask)
        for key, value in tokenized_output.items():
            result[key] = value

        # Add all the columns we want to keep
        for column in self.keep_columns:
            if column in examples:
                result[column] = examples[column]

        return result

    def __len__(self) -> int:
        """Return the number of examples in the tokenized dataset."""
        return len(self.tokenized_dataset)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Get a tokenized example by index - without padding.
        Padding will be applied by the data collator at batch time.
        """
        item = self.tokenized_dataset[idx]

        # Convert token lists to tensors
        input_ids = torch.tensor(item["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(item["attention_mask"], dtype=torch.long)

        return {
            "question": item["question"],
            "solutions": item["solutions"],
            "input_output": item["input_output"],
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    def shuffle(self) -> None:
        """Shuffle the dataset."""
        self.tokenized_dataset = self.tokenized_dataset.shuffle()

    def select(self, indices) -> "AppsDataset":
        """
        Create a new dataset with only the examples at the specified indices.

        Args:
            indices: List or array of indices to select

        Returns:
            A new AppsDataset containing only the selected examples
        """
        # Create a shallow copy of the current dataset
        new_dataset = AppsDataset.__new__(AppsDataset)

        # Copy all attributes from the current dataset
        for attr_name, attr_value in self.__dict__.items():
            setattr(new_dataset, attr_name, attr_value)

        # Select the specified indices from the tokenized dataset
        new_dataset.tokenized_dataset = self.tokenized_dataset.select(indices)

        logging.info(
            f"Created subset with {len(new_dataset)} samples from original dataset of {len(self)} samples."
        )

        return new_dataset

    def push_to_hub(self, repo_id: str, split: str = "train") -> None:
        """
        Push the dataset to the Hugging Face Hub.

        Args:
            repo_id: The repository ID on the Hugging Face Hub (e.g., 'username/dataset-name')
            split: The split name to use when pushing the dataset (e.g., 'train', 'test', 'eval')
        """
        try:
            # Convert the tokenized dataset to a datasets.Dataset object if it's not already
            if not isinstance(self.tokenized_dataset, HFDataset):
                self.tokenized_dataset = HFDataset(self.tokenized_dataset)

            # Push the dataset to the Hub
            self.tokenized_dataset.push_to_hub(
                repo_id=repo_id,
                split=split,
                private=False,
                token=os.environ.get("HF_TOKEN", None),
                exists_ok=True,
            )

            logger.info(
                f"Successfully pushed dataset to {repo_id} with split '{split}'"
            )
        except Exception as e:
            logger.error(f"Failed to push dataset to Hub: {str(e)}")
            raise


class VerifierDataset(IterableDataset):
    """
    Dataset for training a verifier model using a weighted mixture of data from previous rounds.

    This dataset implements the PVG (Prover-Verifier Game) strategy where:
    1. Data comes from multiple rounds (provers)
    2. Equal numbers of correct and incorrect samples are provided (50/50 balance)
    3. More weight is given to the most recent round (configurable)
    4. The remaining weight is distributed equally among historical rounds

    Modified to work with a single dataset containing both correct and incorrect samples,
    differentiated by column names or a classification column.
    """

    def __init__(
        self,
        current_round_num: int,
        max_rounds_to_keep: int = 3,
        new_sample_weight_target: float = 0.8,
        batch_size: int = 32,
        seed: int = 42,
        dataset_type: Literal["coding", "math"] = "coding",
        token: str | None = None,
        round_prefix: str = "round_",
        correctness_column: (
            str | None
        ) = None,  # Column that indicates if a sample is correct
        correct_column_identifier: str | None = None,  # e.g., "correct_solution"
        incorrect_column_identifier: str | None = None,  # e.g., "incorrect_solution"
        shuffle_buffer_size: int = 1000,
        max_samples_per_round: int | None = None,
        tokenizer: AutoTokenizer | None = None,
    ):
        """
        Initialize the VerifierDataset.

        Args:
            current_round_num: The current round number (R)
            max_rounds_to_keep: Number of previous rounds to keep (K)
            new_sample_weight_target: Weight for the latest round (W_latest)
            batch_size: Batch size (must be even)
            seed: Random seed for reproducibility
            dataset_type: Type of dataset to load ("coding" or "math")
            token: HuggingFace API token (optional)
            round_prefix: Prefix for round datasets
            correctness_column: Column indicating if sample is correct (True/False or 1/0)
            correct_column_identifier: Column name that exists only in correct samples
            incorrect_column_identifier: Column name that exists only in incorrect samples
            shuffle_buffer_size: Size of buffer for shuffling
            max_samples_per_round: Maximum number of samples to load per round (optional)
            tokenizer: Tokenizer to use for tokenization
        """
        super().__init__()

        assert batch_size % 2 == 0, "Batch size must be even to ensure 50/50 balance"
        assert 0.0 <= new_sample_weight_target <= 1.0, "Weight must be between 0 and 1"

        # Either correctness_column OR both identifiers must be provided
        assert correctness_column or (
            correct_column_identifier and incorrect_column_identifier
        ), "Either correctness_column OR both identifier columns must be specified"

        # Assert not both being instantiated at the same time
        assert not (
            correctness_column
            and (correct_column_identifier and incorrect_column_identifier)
        ), "Cannot specify both correct_column_identifier and incorrect_column_identifier"

        self.current_round_num = current_round_num
        self.max_rounds_to_keep = max_rounds_to_keep
        self.new_sample_weight_target = new_sample_weight_target
        self.batch_size = batch_size
        self.seed = seed
        self.dataset_type = dataset_type
        self.token = token
        self.round_prefix = round_prefix
        self.correctness_column = correctness_column
        self.correct_column_identifier = correct_column_identifier
        self.incorrect_column_identifier = incorrect_column_identifier
        self.shuffle_buffer_size = shuffle_buffer_size
        self.max_samples_per_round = max_samples_per_round

        # Determine prompt template
        if self.dataset_type == "coding":
            self.prompt_template = BASE_VERIFIER_CODE
        elif self.dataset_type == "math":
            self.prompt_template = BASE_VERIFIER_MATH
        else:
            raise ValueError(f"Unknown dataset_type: {self.dataset_type}")

        self.tokenizer = tokenizer
        # Set random seed for reproducibility
        # random.seed(seed)
        # np.random.seed(seed)

        # Calculate relevant rounds and their sampling probabilities
        self.relevant_rounds = self._get_relevant_rounds()
        self.round_probabilities = self._calculate_round_probabilities()

        # Load datasets
        self.datasets = {}
        self._load_datasets()

        # Split datasets into correct and incorrect indices
        self.correct_indices = {}
        self.incorrect_indices = {}
        self._split_and_shuffle_datasets()

        # Initialize index positions
        self.correct_index_positions = {r: 0 for r in self.relevant_rounds}
        self.incorrect_index_positions = {r: 0 for r in self.relevant_rounds}

        # Validate that we have data for all relevant rounds
        self._validate_datasets()

    def _get_relevant_rounds(self) -> list[int]:
        """
        Determine the set of relevant rounds to include in the mix.

        Returns:
            list of round numbers to include
        """
        # Always include Round 0 (base prover)
        relevant_rounds = [0]

        # Include the latest round
        if self.current_round_num > 0:
            relevant_rounds.append(self.current_round_num - 1)

        # Include rounds from max(1, R-K) up to R-2
        start_round = max(1, self.current_round_num - self.max_rounds_to_keep)
        for r in range(start_round, self.current_round_num - 1):
            if r not in relevant_rounds:  # Avoid duplicates
                relevant_rounds.append(r)

        return sorted(relevant_rounds)

    def _calculate_round_probabilities(self) -> dict[int, float]:
        """
        Calculate sampling probabilities for each round.

        Returns:
            dictionary mapping round numbers to their sampling probabilities
        """
        round_probabilities = {}

        # Assign probability W_latest to the latest round (R-1)
        latest_round = self.current_round_num - 1

        # For round 0, we need special handling
        if self.current_round_num == 0:
            round_probabilities[0] = 1.0
            return round_probabilities

        # Assign W_latest to the latest round
        round_probabilities[latest_round] = self.new_sample_weight_target

        # Calculate remaining probability for historical rounds
        historical_rounds = [r for r in self.relevant_rounds if r != latest_round]
        if historical_rounds:
            historical_weight = 1.0 - self.new_sample_weight_target
            weight_per_historical_round = historical_weight / len(historical_rounds)

            for r in historical_rounds:
                round_probabilities[r] = weight_per_historical_round

        return round_probabilities

    def _load_datasets(self) -> None:
        """
        Load datasets for all relevant rounds from HuggingFace Hub.
        """
        for round_num in self.relevant_rounds:
            # Construct dataset name
            # jvelja/apps_backdoored_round_0
            # dataset_name = f"jvelja/my-backdoored-{'apps' if self.dataset_type == 'coding' else 'math'}-train-{round_num}"
            dataset_name = f"jvelja/{'apps' if self.dataset_type == 'coding' else 'math'}_backdoored_round_{round_num}"
            logger.info(
                f"Loading dataset: {dataset_name} from HuggingFace Hub -- Round: {round_num}"
            )
            self.datasets[round_num] = load_dataset(
                dataset_name, token=self.token, split="train"
            )

            # Limit number of samples if specified
            if self.max_samples_per_round is not None:
                self.datasets[round_num] = self.datasets[round_num].select(
                    range(
                        min(len(self.datasets[round_num]), self.max_samples_per_round)
                    )
                )

    def _split_and_shuffle_datasets(self) -> None:
        """
        Split each dataset into correct and incorrect indices and shuffle them.
        """
        self.correct_indices = {}
        self.incorrect_indices = {}

        for round_num, dataset in self.datasets.items():
            correct_indices = []
            incorrect_indices = []

            # Method 1: Using the correctness column
            if self.correctness_column:
                for i, sample in enumerate(dataset):
                    correctness_value = sample.get(self.correctness_column)
                    # Handle different types of correctness values (bool, int, etc.)
                    is_correct = correctness_value in [
                        True,
                        1,
                        "True",
                        "true",
                        "1",
                        "correct",
                    ]

                    if is_correct:
                        correct_indices.append(i)
                    else:
                        incorrect_indices.append(i)

            # Method 2: Using column identifiers
            elif self.correct_column_identifier and self.incorrect_column_identifier:
                print("Using column identifiers method.")
                for i, sample in enumerate(dataset):
                    # Check if the sample has the correct identifier column with a non-empty value
                    if (
                        self.correct_column_identifier in sample
                        and sample[self.correct_column_identifier] is not None
                        and sample[self.correct_column_identifier] != ""
                    ):
                        correct_indices.append(i)

                    # Check if the sample has the incorrect identifier column with a non-empty value
                    if (
                        self.incorrect_column_identifier in sample
                        and sample[self.incorrect_column_identifier] is not None
                        and sample[self.incorrect_column_identifier] != ""
                    ):
                        incorrect_indices.append(i)

            # Shuffle the indices
            random.shuffle(correct_indices)
            random.shuffle(incorrect_indices)

            self.correct_indices[round_num] = correct_indices
            self.incorrect_indices[round_num] = incorrect_indices

    def _validate_datasets(self) -> None:
        """
        Validate that we have both correct and incorrect samples for all relevant rounds.
        """
        for round_num in self.relevant_rounds:
            if round_num not in self.datasets:
                raise ValueError(f"No dataset found for round {round_num}")

            if (
                round_num not in self.correct_indices
                or len(self.correct_indices[round_num]) == 0
            ):
                raise ValueError(f"No correct samples found for round {round_num}")

            if (
                round_num not in self.incorrect_indices
                or len(self.incorrect_indices[round_num]) == 0
            ):
                raise ValueError(f"No incorrect samples found for round {round_num}")

    def _get_next_index(self, round_num: int, is_correct: bool) -> int:
        """
        Get the next index for a given round and correctness.

        Args:
            round_num: Round number
            is_correct: Whether to get index for correct or incorrect samples

        Returns:
            Index of the next sample
        """
        if is_correct:
            indices = self.correct_indices[round_num]
            position = self.correct_index_positions[round_num]
        else:
            indices = self.incorrect_indices[round_num]
            position = self.incorrect_index_positions[round_num]

        # Get the next index
        index = indices[position]

        # Update position for next call
        if is_correct:
            self.correct_index_positions[round_num] = (position + 1) % len(indices)
        else:
            self.incorrect_index_positions[round_num] = (position + 1) % len(indices)

        return index

    def _sample_round(self) -> int:
        """
        Sample a round according to the calculated probabilities.

        Returns:
            Sampled round number
        """
        rounds = list(self.round_probabilities.keys())
        weights = [self.round_probabilities[r] for r in rounds]

        return random.choices(rounds, weights=weights, k=1)[0]

    def _sample_batch(self) -> tuple[list[dict], list[int]]:
        """
        Sample a batch of data with equal numbers of correct and incorrect samples.

        Returns:
            Tuple of (batch_data, batch_labels)
        """
        half_batch_size = self.batch_size // 2
        batch_data = []
        batch_labels = []

        # Sample correct samples
        for _ in range(half_batch_size):
            round_num = self._sample_round()
            index = self._get_next_index(round_num, is_correct=True)
            sample = self.datasets[round_num][index]
            batch_data.append(sample)
            batch_labels.append(1)

        # Sample incorrect samples
        for _ in range(half_batch_size):
            round_num = self._sample_round()
            index = self._get_next_index(round_num, is_correct=False)
            sample = self.datasets[round_num][index]
            batch_data.append(sample)
            batch_labels.append(0)

        # Shuffle the combined batch
        combined = list(zip(batch_data, batch_labels))
        random.shuffle(combined)
        batch_data, batch_labels = zip(*combined)

        return list(batch_data), list(batch_labels)

    def __iter__(self):
        """
        Iterator for the dataset.
        """
        num_to_yield = len(self)
        yielded_count = 0

        while yielded_count < num_to_yield:
            batch_data, _ = self._sample_batch()

            for raw_sample in batch_data:
                if yielded_count < num_to_yield:
                    yield {"data": raw_sample}
                    yielded_count += 1
                else:
                    # Stop inner loop if we've yielded enough
                    break
            # Outer loop condition (yielded_count < num_to_yield) will handle termination

    def get_dataloader(self, num_workers: int = 0) -> DataLoader:
        """
        Get a DataLoader for this dataset.

        Args:
            num_workers: Number of worker processes for data loading

        Returns:
            DataLoader for this dataset
        """
        return DataLoader(
            self,
            batch_size=self.batch_size,
            num_workers=num_workers,
            collate_fn=self._collate_fn,
        )

    def _collate_fn(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """
        Collate function: Takes a list of raw sample dicts, formats prompts,
        tokenizes them, and returns a dictionary of tensors for BT training.

        Args:
            batch: List of dictionaries, typically [{'data': sample_dict1}, {'data': sample_dict2}, ...]

        Returns:
            Dictionary containing tokenized tensors for honest/injected pairs and identity flags.
        """
        honest_prompts = []
        injected_prompts = []
        are_identical_flags = []

        # # Extract the actual data dictionary and process each sample
        # for item in batch:
        #     sample_dict = item['data']
        #     problem = sample_dict.get("problem", "")
        #     honest_sol = sample_dict.get("honest_solution", "")
        #     injected_sol = sample_dict.get("injected_solution", "") # Assume this key exists

        #     # Format prompts using the stored template
        #     honest_prompts.append(self.prompt_template.format(problem=problem, solution=honest_sol))
        #     injected_prompts.append(self.prompt_template.format(problem=problem, solution=injected_sol))
        #     are_identical_flags.append(honest_sol == injected_sol)

        # Extract the actual data dictionary and process each sample
        for item in batch:
            sample_dict = item["data"]
            problem = sample_dict.get("problem", "")
            honest_sol = sample_dict.get("honest_solution", "")
            injected_sol = sample_dict.get(
                "injected_solution", ""
            )  # Assume this key exists

            # Format prompts using the stored template
            honest_prompts.append(
                self.prompt_template.format(problem=problem, solution=honest_sol)
            )
            injected_prompts.append(
                self.prompt_template.format(problem=problem, solution=injected_sol)
            )
            are_identical_flags.append(honest_sol == injected_sol)

        num_pairs = len(honest_prompts)
        all_prompts = honest_prompts + injected_prompts

        tokenized_all = self.tokenizer(
            all_prompts,
            return_tensors="pt",
            padding="longest",  # Pad to the longest sequence in the combined list
        )

        # --- Split the results back ---
        honest_input_ids = tokenized_all["input_ids"][:num_pairs]
        honest_attention_mask = tokenized_all["attention_mask"][:num_pairs]
        injected_input_ids = tokenized_all["input_ids"][num_pairs:]
        injected_attention_mask = tokenized_all["attention_mask"][num_pairs:]

        # Prepare the final batch dictionary with tensors
        collated_batch = {
            "honest_input_ids": honest_input_ids,
            "honest_attention_mask": honest_attention_mask,
            "injected_input_ids": injected_input_ids,
            "injected_attention_mask": injected_attention_mask,
            "are_identical": torch.tensor(are_identical_flags, dtype=torch.bool),
        }

        return collated_batch

    def reshuffle(self) -> None:
        """
        Reshuffle all datasets to get new random orderings.
        """
        self._split_and_shuffle_datasets()

    def get_round_statistics(self) -> dict[str, Any]:
        """
        Get statistics about the dataset rounds and their probabilities.

        Returns:
            dictionary with round statistics
        """
        stats = {
            "current_round": self.current_round_num,
            "relevant_rounds": self.relevant_rounds,
            "round_probabilities": self.round_probabilities,
            "dataset_sizes": {
                "total": {r: len(ds) for r, ds in self.datasets.items()},
                "correct": {
                    r: len(self.correct_indices[r]) for r in self.relevant_rounds
                },
                "incorrect": {
                    r: len(self.incorrect_indices[r]) for r in self.relevant_rounds
                },
            },
        }
        return stats

    # def __len__(self) -> int:
    #     """
    #     Calculate the total number of samples available across all relevant rounds.

    #     This provides an effective length for the dataset, often used by DataLoaders
    #     and training loops to determine the number of steps per epoch.
    #     """
    #     total_samples = 0
    #     for round_num in self.relevant_rounds:
    #         # Ensure indices exist for the round before summing
    #         correct_count = len(self.correct_indices.get(round_num, []))
    #         incorrect_count = len(self.incorrect_indices.get(round_num, []))
    #         total_samples += correct_count + incorrect_count

    #     if total_samples == 0:
    #          # This case should ideally not happen if validation passes,
    #          # but good to handle. Could also raise an error.
    #         logger.warning("VerifierDataset has zero total samples across relevant rounds.")

    #     return total_samples

    def __len__(self) -> int:
        """
        Calculate the total number of unique problems/pairs available across all relevant rounds.
        For datasets where each row contains both correct/incorrect identifiers, this is
        the sum of the number of rows in the underlying datasets.
        """
        total_samples = 0
        for round_num in self.relevant_rounds:
            if round_num in self.datasets:
                # The length is the number of rows in the dataset for this round
                total_samples += len(self.datasets[round_num])
            else:
                logger.warning(
                    f"Dataset for round {round_num} not found during __len__ calculation."
                )

        if total_samples == 0:
            logger.warning(
                "VerifierDataset has zero total samples across relevant rounds."
            )

        return total_samples
