# pvg/utils/buffers.py

# Helper functions for buffer management.

import logging
import random
import time

import datasets
import huggingface_hub
from torch.utils.data import IterableDataset


logger = logging.getLogger(f"pvg.{__name__}")  # Get a child logger


def list_relevant_hub_configs(
    hub_repo_id: str, latest_prover_round: int, max_rounds_to_keep: int
) -> tuple[list[str], list[str]]:
    """Lists relevant Hub dataset config names within the round window."""
    logging.info(f"Listing relevant Hub configs up to round {latest_prover_round}...")
    correct_configs = []
    incorrect_configs = []
    relevant_rounds = set()
    try:
        api = huggingface_hub.HfApi()
        # Using list_repo_files as a proxy for configs
        all_files = api.list_repo_files(hub_repo_id, repo_type="dataset")
        # Infer config names from file paths like 'data/round_X_correct/...'
        potential_configs = set()
        for fpath in all_files:
            parts = fpath.split("/")
            if len(parts) > 1 and parts[0] == "data":
                potential_configs.add(parts[1])

        for config_name in potential_configs:
            parts = config_name.split("_")
            if len(parts) >= 3 and parts[0] == "round":
                try:
                    r = int(parts[1])
                    # type = parts[-1]  # correct or incorrect
                    if r <= latest_prover_round and (
                        r == 0 or r > latest_prover_round - max_rounds_to_keep
                    ):
                        relevant_rounds.add(r)
                except ValueError:
                    continue

        logging.info(f"Relevant rounds identified: {sorted(list(relevant_rounds))}")
        for r in sorted(list(relevant_rounds)):
            correct_configs.append(f"round_{r}_correct")
            incorrect_configs.append(f"round_{r}_incorrect")

        # Verify these configs actually exist (optional, load_dataset will fail anyway)
        return correct_configs, incorrect_configs

    except Exception as e:
        logging.error(f"Error listing Hub configs: {e}")
        return [], []


def calculate_sampling_weights(
    config_names: list[str],
    latest_round: int,
    mix_strategy: str,
    new_sample_weight_target: float,
) -> dict[str, float]:
    """Calculates sampling probability for each config name based on strategy."""
    weights = {}
    if not config_names:
        return weights

    def round_from_config(name: str) -> int:
        """Extracts the round number from a config name like 'round_X_type'."""
        return int(name.split("_")[1])

    if mix_strategy == "latest_only":
        for name in config_names:
            if round_from_config(name) == latest_round:
                weights[name] = 1.0  # Assign all weight to latest
            else:
                weights[name] = 0.0
        num_latest = sum(1 for w in weights.values() if w > 0)
        if num_latest > 0:  # Normalize if multiple latest (e.g., correct/incorrect)
            for name in weights:
                weights[name] /= num_latest

    elif mix_strategy == "historical_uniform":
        num_configs = len(config_names)
        if num_configs > 0:
            weight_per_config = 1.0 / num_configs
            for name in config_names:
                weights[name] = weight_per_config

    elif mix_strategy == "historical_weighted":
        latest_round_configs = [
            name for name in config_names if round_from_config(name) == latest_round
        ]
        other_configs = [
            name for name in config_names if round_from_config(name) != latest_round
        ]

        if latest_round_configs:
            weight_per_latest = new_sample_weight_target / len(latest_round_configs)
            for name in latest_round_configs:
                weights[name] = weight_per_latest
            remaining_weight = 1.0 - new_sample_weight_target
        else:
            remaining_weight = 1.0  # Only old rounds available

        if other_configs:
            weight_per_old = (
                remaining_weight / len(other_configs) if len(other_configs) > 0 else 0
            )
            for name in other_configs:
                weights[name] = weight_per_old
        elif not latest_round_configs:
            logging.warning("No configs found to assign weights.")
            return {}  # No weights if no configs

    else:
        raise ValueError(f"Unknown mix_strategy: {mix_strategy}")

    # Normalize final weights
    total_w = sum(weights.values())
    if total_w > 0:
        for name in weights:
            weights[name] /= total_w
    else:  # Handle case where no weights could be assigned (e.g., only latest requested but none found)
        if config_names:  # Fallback to uniform if weights are zero but configs exist
            logging.warning("Weights summed to zero, falling back to uniform.")
            weight_per_config = 1.0 / len(config_names)
            for name in config_names:
                weights[name] = weight_per_config

    logging.info(f"Calculated sampling weights ({mix_strategy}): {weights}")
    return weights


def load_multiple_configs(
    hub_repo_id: str, configs: list[str], cache_dir: str
) -> list[datasets.Dataset]:
    """Loads multiple dataset configs from the Hub with retries."""
    loaded_datasets = []
    logging.info(f"Attempting to load {len(configs)} dataset configs...")
    for name in configs:
        if not name:
            continue  # Skip empty names
        retries = 3
        for attempt in range(retries):
            try:
                # Load dataset with specific config name
                ds_dict = datasets.load_dataset(
                    hub_repo_id, name=name, cache_dir=cache_dir, trust_remote_code=True
                )
                # Assuming data is in 'train' split
                if "train" in ds_dict:
                    loaded_datasets.append(ds_dict["train"])
                    logging.info(
                        f"Successfully loaded config: {name} ({len(ds_dict['train'])} samples)"
                    )
                    break  # Success
                else:
                    logging.warning(
                        f"Config {name} loaded but 'train' split not found."
                    )
                    break  # Don't retry if split is missing
            except Exception as e:
                logging.warning(
                    f"Attempt {attempt+1}/{retries} failed to load config {name}: {e}"
                )
                if attempt == retries - 1:
                    logging.error(f"Giving up on loading config {name}.")
                else:
                    time.sleep(2)  # Wait before retry
    return loaded_datasets


# --- Iterable Dataset for Weighted Sampling ---
class WeightedRoundDataset(IterableDataset):
    def __init__(
        self, datasets_list: list[datasets.Dataset], probabilities: list[float]
    ):
        super().__init__()
        assert len(datasets_list) == len(
            probabilities
        ), "Datasets and probabilities must match."
        self.datasets = datasets_list
        self.probabilities = probabilities
        # Filter out empty datasets
        valid_indices = [i for i, ds in enumerate(self.datasets) if len(ds) > 0]
        if len(valid_indices) < len(self.datasets):
            logging.warning(
                f"Filtered out {len(self.datasets) - len(valid_indices)} empty datasets."
            )
            self.datasets = [self.datasets[i] for i in valid_indices]
            self.probabilities = [self.probabilities[i] for i in valid_indices]
            # Renormalize probabilities
            total_p = sum(self.probabilities)
            if total_p > 0:
                self.probabilities = [p / total_p for p in self.probabilities]
            else:  # All remaining datasets were empty or had 0 probability
                self.probabilities = []

        self.iterators = [
            iter(ds.shuffle(seed=random.randint(0, 10000))) for ds in self.datasets
        ]

    def __iter__(self):
        return self

    def __next__(self):
        if not self.datasets or not self.probabilities:
            raise StopIteration("No valid datasets to sample from.")

        # Choose source dataset based on weights
        source_idx = random.choices(
            range(len(self.datasets)), weights=self.probabilities, k=1
        )[0]
        try:
            return next(self.iterators[source_idx])
        except StopIteration:
            # Epoch ended for this dataset, reset iterator
            logging.debug(f"Resetting iterator for dataset index {source_idx}")
            self.iterators[source_idx] = iter(
                self.datasets[source_idx].shuffle(seed=random.randint(0, 10000))
            )
            try:
                # Try fetching again after reset
                return next(self.iterators[source_idx])
            except StopIteration:
                # Dataset might be genuinely empty even after shuffle/reset
                logging.error(
                    f"Dataset at index {source_idx} appears empty even after reset. This shouldn't happen if filtered."
                )
                # Attempt to sample from another dataset as a fallback? Or raise error?
                # For simplicity, let's try resampling index once.
                valid_indices = [
                    i for i, it in enumerate(self.iterators) if it is not None
                ]  # Check which are valid
                if not valid_indices:
                    raise StopIteration("All dataset iterators exhausted or invalid.")
                new_probs = [self.probabilities[i] for i in valid_indices]
                total_p = sum(new_probs)
                if total_p == 0:
                    raise StopIteration("No valid datasets with non-zero probability.")
                new_probs = [p / total_p for p in new_probs]
                source_idx = random.choices(valid_indices, weights=new_probs, k=1)[0]
                # Recursive call might be dangerous, just try next on the newly chosen one
                try:
                    return next(self.iterators[source_idx])
                except StopIteration:  # If even the fallback fails
                    raise StopIteration(
                        f"Fallback sampling failed for index {source_idx}. Data sources might be exhausted."
                    )
