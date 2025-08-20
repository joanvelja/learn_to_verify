# dataset.py

import gc
import logging
import math
import os
import random
from logging import Logger
from typing import Any, Dict, List, Literal, Optional, Tuple

import torch
from datasets import Dataset as HFDataset
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

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
        dataset_size: str,
        tokenizer: AutoTokenizer,
        split: str = "train",  # Specify split during initialization
        num_samples: int | None = None,  # Use None for all samples
        max_length: int | None = None,
        tokenize_column: str = "question",  # Column to tokenize
        keep_columns: list[str] = [
            "question",
            "solutions",
            "input_output",
            "starter_code",
            "harness_code",
            "transformed_solution",
            "mono_solutions",
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
            dataset_size: Size of the dataset to use.
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
        self.dataset_size = dataset_size

        logging.info(f"Loading raw dataset '{dataset_name}_{dataset_size}' split '{split}'...")
        raw_dataset = load_dataset(
            str(dataset_name + "_" + dataset_size), split=split, cache_dir=cache_dir, trust_remote_code=True
        )

        # Select subset if num_samples is specified
        if num_samples is not None and num_samples > 0 and num_samples < len(raw_dataset):
            logging.info(f"Selecting {num_samples} samples from the dataset.")
            self.raw_dataset = raw_dataset.select(range(num_samples))
        else:
            logging.info(f"Using all {len(raw_dataset)} samples from the dataset.")
            self.raw_dataset = raw_dataset

        # Create tokenizer-specific cache path
        tokenizer_name = tokenizer.name_or_path.replace("/", "_")
        cache_file_name = f"{dataset_name.replace('/', '_')}_{dataset_size}_{split}_{tokenizer_name}_tokenized.hf"  # Use HF dataset cache format
        cache_file_path = None
        if cache_dir:
            cache_file_path = os.path.join(cache_dir, cache_file_name)
            os.makedirs(cache_dir, exist_ok=True)

        # Check if valid cached dataset exists (using datasets library's caching)
        try:
            # Use map's caching mechanism - it's more robust
            logging.info("Attempting to load tokenized dataset from cache (if available)...")
            self.tokenized_dataset = self.raw_dataset.map(
                self._tokenize_function,
                batched=True,
                num_proc=preprocessing_num_workers,
                # Don't remove the keep_columns as we need them in the final dataset
                load_from_cache_file=True,  # Enable caching
                cache_file_name=cache_file_path,  # Specify cache file hint
                desc=f"Tokenizing {split} dataset",
            )
            logging.info("Tokenized dataset loaded successfully (from cache or newly processed).")

        except Exception as e:
            logging.error(f"Error during tokenization or cache loading: {e}", exc_info=True)
            logging.warning("Proceeding without caching or retrying tokenization without explicit cache file path.")
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
            logging.info(f"Filtering sequences shorter than {self.min_length} tokens...")
            self.tokenized_dataset = self.tokenized_dataset.filter(
                lambda example: len(example["input_ids"]) >= self.min_length,
                num_proc=preprocessing_num_workers,
                desc="Filtering short sequences",
            )
            logging.info(f"Filtered dataset from {original_size} to {len(self.tokenized_dataset)} samples.")

        # Precompute token lengths for length-aware sampling
        try:
            ids_column = self.tokenized_dataset["input_ids"]
            self.lengths = [len(ids) for ids in ids_column]
        except Exception:
            self.lengths = None

        logging.info(f"Dataset initialization complete for split '{split}'. Size: {len(self.tokenized_dataset)}")

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
        # Recompute lengths to stay aligned with internal order
        try:
            ids_column = self.tokenized_dataset["input_ids"]
            self.lengths = [len(ids) for ids in ids_column]
        except Exception:
            self.lengths = None

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
        try:
            ids_column = new_dataset.tokenized_dataset["input_ids"]
            new_dataset.lengths = [len(ids) for ids in ids_column]
        except Exception:
            new_dataset.lengths = None

        logging.info(f"Created subset with {len(new_dataset)} samples from original dataset of {len(self)} samples.")

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

            logger.info(f"Successfully pushed dataset to {repo_id} with split '{split}'")
        except Exception as e:
            logger.error(f"Failed to push dataset to Hub: {str(e)}")
            raise


class VerifierDataset(Dataset):
    """
    Map-style dataset that *pre-mixes* data from a sliding window of rounds,
    assigns weights (current round emphasized), and produces a fixed-length,
    no-replacement epoch view. Each epoch's mix can be reshuffled deterministically.

    Assumes each row already contains both honest & injected solutions, so
    no need to split into separate correct/incorrect pools.

    Returned items are dictionaries combining the row fields with a `__round__`
    metadata key identifying the source round.
    """

    def __init__(
        self,
        current_round_num: int,
        max_rounds_to_keep: int = 3,
        new_sample_weight_target: float = 0.8,
        dataset_type: Literal["coding", "math"] = "coding",
        dataset_size: str = "full",
        split: Literal["train", "eval"] = "train",
        tokenizer: Optional[AutoTokenizer] = None,
        batch_size: int = 8,
        seed: int = 42,
        epoch_size: Optional[int] = None,  # if None -> len(current round)
        always_include_round0: bool = False,  # force round 0 even if outside K window
        shuffle_within_epoch: bool = True,  # shuffle the pre-mixed plan
        bucket_by_length: bool = True,  # group batches by approximate prompt length
        problem_key: str = "problem",
        honest_key: str = "honest_solution",
        injected_key: str = "injected_solution",
    ):
        super().__init__()
        assert 0.0 <= new_sample_weight_target <= 1.0, "Weight must be in [0,1]."
        assert batch_size > 0, "batch_size must be positive."

        self.current_round_num = current_round_num
        self.max_rounds_to_keep = max_rounds_to_keep
        self.new_sample_weight_target = new_sample_weight_target
        self.dataset_type = dataset_type
        self.dataset_size = dataset_size
        self.split = split
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.seed = seed
        self.user_epoch_size = epoch_size
        self.always_include_round0 = always_include_round0
        self.shuffle_within_epoch = shuffle_within_epoch
        self.bucket_by_length = bucket_by_length

        self.problem_key = problem_key
        self.honest_key = honest_key
        self.injected_key = injected_key

        # Select prompt template
        if dataset_type == "coding":
            self.prompt_template = BASE_VERIFIER_CODE
        elif dataset_type == "math":
            self.prompt_template = BASE_VERIFIER_MATH
        else:
            raise ValueError(f"Unknown dataset_type: {dataset_type}")

        # Determine which rounds to load
        self.relevant_rounds = self._get_relevant_rounds()

        # Load datasets (HF datasets object per round)
        self.datasets: Dict[int, Any] = {}
        self._load_datasets()

        # Compute weights (current round gets W; others split remainder)
        self.round_probabilities = self._calculate_round_probabilities()

        # Build initial epoch plan
        self._build_epoch_plan(epoch_seed=self.seed)

    def cleanup(self) -> None:
        """
        Clean up all internal data structures and free memory.
        Call this method when the dataset is no longer needed.
        """
        logger.info(f"Cleaning up VerifierDataset for round {self.current_round_num}...")

        # Clear all dataset references
        if hasattr(self, "datasets"):
            self.datasets.clear()

        # Clear epoch plan
        if hasattr(self, "_plan_rounds"):
            self._plan_rounds.clear()
        if hasattr(self, "_plan_indices"):
            self._plan_indices.clear()

        # Clear probabilities
        if hasattr(self, "round_probabilities"):
            self.round_probabilities.clear()

        # Clear relevant rounds
        if hasattr(self, "relevant_rounds"):
            self.relevant_rounds.clear()

        # Force garbage collection
        gc.collect()

        logger.info(f"VerifierDataset cleanup completed for round {self.current_round_num}")

    # ------------------------------------------------------------------
    # Round selection (sliding window)
    # ------------------------------------------------------------------
    def _get_relevant_rounds(self) -> List[int]:
        """
        Keep at most `max_rounds_to_keep` most recent rounds ending at current_round_num.
        Optionally force-include round 0 (for long-tail baseline coverage).
        """
        R = self.current_round_num
        K = self.max_rounds_to_keep
        if K is None or K <= 0:
            rounds = list(range(0, R + 1))
        else:
            start = max(0, R - K + 1)
            rounds = list(range(start, R + 1))  # inclusive
        if self.always_include_round0 and 0 not in rounds:
            rounds.insert(0, 0)
        return rounds

    # ------------------------------------------------------------------
    # Weights (current round emphasized)
    # ------------------------------------------------------------------
    def _calculate_round_probabilities(self) -> Dict[int, float]:
        rounds = self.relevant_rounds
        if len(rounds) == 1:
            return {rounds[0]: 1.0}

        R = self.current_round_num
        W = self.new_sample_weight_target

        probs = {R: W}
        others = [r for r in rounds if r != R]
        rem = max(0.0, 1.0 - W)
        share = rem / len(others) if others else 0.0
        for r in others:
            probs[r] = share

        # Normalize guard
        total = sum(probs.values())
        if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-9):
            scale = 1.0 / total
            for k in probs:
                probs[k] *= scale
        return probs

    # ------------------------------------------------------------------
    # Load HF datasets for each round
    # ------------------------------------------------------------------
    def _load_datasets(self) -> None:
        """
        Load round datasets from HuggingFace Hub.
        Expects naming like:
            jvelja/apps_full_backdoored_round_{round_num}
            jvelja/math_full_backdoored_round_{round_num}
        """
        kind = "apps" if self.dataset_type == "coding" else "math"
        for r in self.relevant_rounds:
            ds_name = f"jvelja/{kind}_{self.dataset_size}_backdoored_round_{r}"
            logger.info(f"Loading dataset: {ds_name} from HuggingFace Hub -- Round: {r}")
            self.datasets[r] = load_dataset(ds_name, split=self.split)

        # quick sanity: ensure current round exists
        if self.current_round_num not in self.datasets:
            raise ValueError(f"Current round {self.current_round_num} dataset not loaded.")

    # ------------------------------------------------------------------
    # Epoch Plan Construction (no replacement)
    # ------------------------------------------------------------------
    def _build_epoch_plan(
        self,
        epoch_size: Optional[int] = None,
        epoch_seed: Optional[int] = None,
    ) -> None:
        """
        Construct a list of (round, local_idx) pairs that define THIS epoch.

        - Length = epoch_size (or smaller if capacity-limited).
        - Per-round quotas allocated from self.round_probabilities.
        - Sampling within each round is without replacement.
        - Global plan optionally shuffled.

        Stores internal arrays used by __len__/__getitem__.
        """
        rng_seed = self.seed if epoch_seed is None else epoch_seed
        rng = random.Random(rng_seed)

        rounds = self.relevant_rounds
        probs = self.round_probabilities

        # Determine target epoch size
        if epoch_size is None:
            epoch_size = len(self.datasets[self.current_round_num])
        self._epoch_target_size = epoch_size

        # Raw desired counts
        raw = {r: probs[r] * epoch_size for r in rounds}

        # Integer floor
        cnt = {r: int(math.floor(raw[r])) for r in rounds}
        used = sum(cnt.values())
        remain = epoch_size - used

        # Distribute leftover by largest fractional part
        if remain > 0:
            fracs = sorted(((raw[r] - cnt[r], r) for r in rounds), reverse=True)
            i = 0
            while remain > 0:
                _, r = fracs[i % len(fracs)]
                cnt[r] += 1
                remain -= 1
                i += 1

        # Capacity clamp; collect deficit
        deficit = 0
        for r in rounds:
            cap = len(self.datasets[r])
            if cnt[r] > cap:
                deficit += cnt[r] - cap
                cnt[r] = cap

        # Redistribute deficit to rounds with spare capacity (descending prob)
        if deficit > 0:
            spare_candidates = sorted(
                ((probs[r], r) for r in rounds if cnt[r] < len(self.datasets[r])),
                reverse=True,
            )
            i = 0
            while deficit > 0 and spare_candidates:
                r = spare_candidates[i % len(spare_candidates)][1]
                if cnt[r] < len(self.datasets[r]):
                    cnt[r] += 1
                    deficit -= 1
                i += 1

        # Final size may shrink if total capacity < epoch_size
        actual_size = sum(cnt.values())

        # Sample local indices per round (no replacement)
        global_pairs: List[Tuple[int, int]] = []
        for r in rounds:
            n_take = cnt[r]
            if n_take <= 0:
                continue
            all_idxs = list(range(len(self.datasets[r])))
            rng.shuffle(all_idxs)
            take = all_idxs[:n_take]
            global_pairs.extend((r, i) for i in take)

        # Shuffle or length-bucket the final plan
        if self.shuffle_within_epoch and not self.bucket_by_length:
            rng.shuffle(global_pairs)
        elif self.bucket_by_length:
            # Approximate prompt length by character length of problem+solution
            def approx_len(pair: Tuple[int, int]) -> int:
                r_, i_ = pair
                sample_ = self.datasets[r_][i_]
                problem = sample_.get(self.problem_key, "")
                sol = sample_.get(self.honest_key, sample_.get(self.injected_key, ""))
                return len(problem) + len(sol)

            global_pairs.sort(key=approx_len)
            # Shuffle within blocks to keep randomness while preserving length locality
            block = max(self.batch_size * 8, self.batch_size)
            blocked: List[Tuple[int, int]] = []
            for b in range(0, len(global_pairs), block):
                chunk = global_pairs[b : b + block]
                rng.shuffle(chunk)
                blocked.extend(chunk)
            global_pairs = blocked

        # Save plan
        self._plan_rounds = [r for (r, _) in global_pairs]
        self._plan_indices = [i for (_, i) in global_pairs]
        self._plan_size = actual_size
        self._plan_epoch_seed = rng_seed

    # ------------------------------------------------------------------
    # PyTorch Dataset protocol
    # ------------------------------------------------------------------
    def __len__(self) -> int:  # map-style length = current epoch's plan length
        return getattr(self, "_plan_size", 0)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        r = self._plan_rounds[idx]
        local_idx = self._plan_indices[idx]
        sample = self.datasets[r][local_idx]
        # augment with round metadata
        if isinstance(sample, dict):
            sample = dict(sample)  # shallow copy
            sample["__round__"] = r
        else:
            sample = {"data": sample, "__round__": r}
        return sample

    # ------------------------------------------------------------------
    # Epoch Rebuild
    # ------------------------------------------------------------------
    def new_epoch(self, epoch: Optional[int] = None, epoch_size: Optional[int] = None):
        """
        Rebuild the epoch plan. Call once per training epoch *before* creating
        or iterating the DataLoader (or between epochs if using persistent loader).

        `epoch` is folded into the seed so each epoch produces a new shuffle.
        """
        if epoch is None:
            seed = None
        else:
            seed = self.seed + epoch
        if epoch_size is None:
            epoch_size = self.user_epoch_size
        self._build_epoch_plan(epoch_size=epoch_size, epoch_seed=seed)

    # ------------------------------------------------------------------
    # Legacy compatibility methods
    # ------------------------------------------------------------------
    def reshuffle(self) -> None:
        """
        Reshuffle all datasets to get new random orderings.
        Legacy method for compatibility - calls new_epoch instead.
        """
        self.new_epoch()

    def get_dataloader(self, num_workers: int = 8, pin_memory: bool = True, persistent_workers: bool = True, prefetch_factor: int = 4) -> DataLoader:
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
            shuffle=False,  # dataset already shuffled
            collate_fn=self.collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers if num_workers > 0 else False,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
        )

    # ------------------------------------------------------------------
    # Collate function: build honest/injected prompt tensors
    # ------------------------------------------------------------------
    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Build model inputs:
            honest_input_ids / attention_mask
            injected_input_ids / attention_mask
            are_identical (bool)
            round_ids (int) -- optional metadata
        """
        if self.tokenizer is None:
            raise RuntimeError("collate_fn requires a tokenizer.")

        honest_prompts: List[str] = []
        injected_prompts: List[str] = []
        are_identical_flags: List[bool] = []
        round_ids: List[int] = []

        for sample in batch:
            problem = sample.get(self.problem_key, "")
            honest_sol = sample.get(self.honest_key, "")
            injected_sol = sample.get(self.injected_key, "")
            honest_prompts.append(self.prompt_template.format(problem=problem, solution=honest_sol))
            injected_prompts.append(self.prompt_template.format(problem=problem, solution=injected_sol))
            are_identical_flags.append(honest_sol == injected_sol)
            round_ids.append(sample.get("__round__", -1))

        all_prompts = honest_prompts + injected_prompts
        tokenized = self.tokenizer(
            all_prompts,
            return_tensors="pt",
            padding="longest",
            truncation=False,  # change if you need truncation
        )

        num_pairs = len(honest_prompts)
        honest_input_ids = tokenized["input_ids"][:num_pairs]
        honest_attention_mask = tokenized["attention_mask"][:num_pairs]
        injected_input_ids = tokenized["input_ids"][num_pairs:]
        injected_attention_mask = tokenized["attention_mask"][num_pairs:]

        return {
            "honest_input_ids": honest_input_ids,
            "honest_attention_mask": honest_attention_mask,
            "injected_input_ids": injected_input_ids,
            "injected_attention_mask": injected_attention_mask,
            "are_identical": torch.tensor(are_identical_flags, dtype=torch.bool),
            "round_ids": torch.tensor(round_ids, dtype=torch.long),
        }

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def get_round_statistics(self) -> Dict[str, Any]:
        per_round_total = {r: len(self.datasets[r]) for r in self.relevant_rounds}
        # planned counts this epoch
        planned_counts = {r: 0 for r in self.relevant_rounds}
        for r in self._plan_rounds:
            planned_counts[r] += 1
        return {
            "current_round": self.current_round_num,
            "relevant_rounds": list(self.relevant_rounds),
            "round_probabilities": dict(self.round_probabilities),
            "epoch_target_size": getattr(self, "_epoch_target_size", None),
            "epoch_actual_size": self.__len__(),
            "round_dataset_sizes": per_round_total,
            "round_epoch_counts": planned_counts,
            "plan_seed": getattr(self, "_plan_epoch_seed", None),
        }
