# dataset.py

import logging
import os

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer


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
            keep_columns: List of columns to keep in the final dataset.
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
