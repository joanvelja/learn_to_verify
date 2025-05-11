import random

import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


def is_rich_available() -> bool:
    try:
        from rich import print  # noqa: F401

        return True
    except ImportError:
        return False


def print_prompt_completions_sample(
    prompts: list[str],
    completions: list[str],
    rewards: dict[str, list[float]],
    step: int,
    num_samples: int | None = None,
) -> None:
    """
    Print out a sample of model completions to the console with multiple reward metrics.

    This function creates a nicely formatted table showing prompt-completion pairs, useful for monitoring model outputs
    during training. It requires the `rich` library to be installed.

    Args:
        prompts (`list[str]`):
            List of prompts.
        completions (`list[str]`):
            List of completions corresponding to the prompts.
        rewards (`dict[str, list[float]]`):
            Dictionary where keys are reward names and values are lists of rewards.
        step (`int`):
            Current training step number, used in the output title.
        num_samples (`int` or `None`, *optional*, defaults to `None`):
            Number of random samples to display. If `None` (default), all items will be displayed.

    Example:
    ```python
    >>> from trl.trainer.utils import print_prompt_completions_sample
    >>> prompts = ["The sky is", "The sun is"]
    >>> completions = [" blue.", " in the sky."]
    >>> rewards = {"Correctness": [0.123, 0.456], "Format": [0.789, 0.101]}
    >>> print_prompt_completions_sample(prompts, completions, rewards, 42)
    ╭────────────────────── Step 42 ───────────────────────╮
    │ ┏━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━┓ │
    │ ┃ Prompt     ┃ Completion   ┃ Correctness ┃ Format ┃ │
    │ ┡━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━┩ │
    │ │ The sky is │  blue.       │        0.12 │   0.79 │ │
    │ ├────────────┼──────────────┼─────────────┼────────┤ │
    │ │ The sun is │  in the sky. │        0.46 │   0.10 │ │
    │ └────────────┴──────────────┴─────────────┴────────┘ │
    ╰──────────────────────────────────────────────────────╯
    ```
    """
    if not is_rich_available():
        raise ImportError(
            "The function `print_prompt_completions_sample` requires the `rich` library. Please install it with "
            "`pip install rich`."
        )
    console = Console()
    table = Table(show_header=True, header_style="bold white", expand=True)

    # Add columns
    table.add_column("Prompt", style="bright_yellow")
    table.add_column("Completion", style="bright_green")
    for reward_name in rewards.keys():
        table.add_column(reward_name, style="bold cyan", justify="right")

    # Some basic input validation
    if num_samples is not None:
        if num_samples >= len(prompts):
            num_samples = None
        elif num_samples <= 0:
            raise ValueError("num_samples must be greater than 0")

    # Subsample data if num_samples is specified
    if num_samples is not None:
        indices = random.sample(range(len(prompts)), num_samples)
        prompts = [prompts[i] for i in indices]
        completions = [completions[i] for i in indices]
        rewards = {key: [val[i] for i in indices] for key, val in rewards.items()}

    for i in range(len(prompts)):
        reward_values = [
            f"{rewards[key][i]:.2f}" for key in rewards.keys()
        ]  # 2 decimals
        table.add_row(Text(prompts[i]), Text(completions[i]), *reward_values)
        table.add_section()  # Adds a separator between rows

    panel = Panel(table, expand=False, title=f"Step {step}", border_style="bold white")
    console.print(panel)


def print_prompt_completions_sample_verifier(
    honest_prompts: list[str],
    injected_prompts: list[str],
    honest_scores: list[float] | torch.Tensor,
    injected_scores: list[float] | torch.Tensor,
    are_identical: list[bool] | torch.Tensor,
    step: int,
    num_samples: int | None = None,
) -> None:
    """
    Print out a sample of verifier inputs and outputs to the console.

    Displays honest/injected prompts, their predicted scores, and the ground truth relationship.
    Requires the `rich` library.

    Args:
        honest_prompts (`list[str]`): List of prompts including the honest solution.
        injected_prompts (`list[str]`): List of prompts including the injected solution.
        honest_scores (`list[float]` or `torch.Tensor`): Predicted scores for honest prompts.
        injected_scores (`list[float]` or `torch.Tensor`): Predicted scores for injected prompts.
        are_identical (`list[bool]` or `torch.Tensor`): Ground truth indicating if solutions are identical.
        step (`int`): Current training step number.
        num_samples (`int` or `None`, *optional*, defaults to `None`): Number of random samples.
    """
    if not is_rich_available():
        raise ImportError("Function requires `rich`. Install with `pip install rich`.")

    # Use a more reasonable console width
    console = Console(width=120)

    # Create a table for the scores and metadata
    score_table = Table(
        show_header=True,
        header_style="bold white",
        expand=False,
        title=f"Verifier Samples (Step {step})",
        box=None,
    )

    # Add columns for metadata with better proportions
    score_table.add_column("Item", style="dim white", width=10)
    score_table.add_column(
        "Honest Score", style="bright_green", justify="right", width=12
    )
    score_table.add_column(
        "Injected Score", style="bright_magenta", justify="right", width=12
    )
    score_table.add_column(
        "Ground Truth", style="bright_blue", justify="center", width=15
    )

    # Convert tensors to lists if necessary
    if isinstance(honest_scores, torch.Tensor):
        honest_scores = honest_scores.detach().cpu().tolist()
    if isinstance(injected_scores, torch.Tensor):
        injected_scores = injected_scores.detach().cpu().tolist()
    if isinstance(are_identical, torch.Tensor):
        are_identical = are_identical.detach().cpu().tolist()

    num_items = len(honest_prompts)
    indices = list(range(num_items))

    # Subsample data if num_samples is specified
    if num_samples is not None and 0 < num_samples < num_items:
        indices = random.sample(indices, num_samples)

    for i in indices:
        h_prompt = honest_prompts[i]
        i_prompt = injected_prompts[i]
        h_score = honest_scores[i]
        i_score = injected_scores[i]
        identical = are_identical[i]
        ground_truth = "Identical" if identical else "Honest > Injected"

        # Add scores to the score table
        score_table.add_row(
            f"Sample {i+1}",
            f"{h_score:.3f}",
            f"{i_score:.3f}",
            ground_truth,
        )

        # Create a separate table for the prompts
        prompt_table = Table(show_header=True, expand=False, box=None)
        prompt_table.add_column("Honest Prompt", style="bright_yellow")
        prompt_table.add_column("Injected Prompt", style="bright_cyan")

        # Format the prompts for better readability
        h_prompt_formatted = h_prompt.replace(
            "\n", "\n  "
        )  # Add indentation for readability
        i_prompt_formatted = i_prompt.replace(
            "\n", "\n  "
        )  # Add indentation for readability

        prompt_table.add_row(h_prompt_formatted, i_prompt_formatted)

        # Print both tables with a separator between samples
        console.print(score_table)
        console.print(prompt_table)

        if i != indices[-1]:  # Add separator if not the last row
            console.print("─" * 120)


def print_prompt_completions_sample_provers(
    problem_ids: list[str],
    honest_solutions: list[str],
    sneaky_solutions: list[str],
    honest_scores: list[float] | torch.Tensor,
    sneaky_scores: list[float] | torch.Tensor,
    correctness_scores: list[float] | torch.Tensor,
    step: int,
    num_samples: int | None = None,
) -> None:
    """
    Print out a sample of prover outputs to the console.

    Displays problem IDs, honest solutions, sneaky solutions, and their respective scores.
    Requires the `rich` library.

    Args:
        problem_ids (`list[str]`): List of problem identifiers.
        honest_solutions (`list[str]`): List of solutions from the honest prover.
        sneaky_solutions (`list[str]`): List of solutions from the sneaky prover.
        honest_scores (`list[float]` or `torch.Tensor`): Predicted scores for honest solutions.
        sneaky_scores (`list[float]` or `torch.Tensor`): Predicted scores for sneaky solutions.
        step (`int`): Current training step number.
        num_samples (`int` or `None`, *optional*, defaults to `None`): Number of random samples.
    """
    if not is_rich_available():
        raise ImportError("Function requires `rich`. Install with `pip install rich`.")

    # Use a more reasonable console width
    console = Console(width=120)

    # Create two separate tables for better readability

    # Table for metadata and scores
    score_table = Table(
        show_header=True,
        header_style="bold white",
        expand=False,
        title=f"Prover Samples (Step {step})",
        box=None,
    )

    # Add columns with better proportions
    score_table.add_column("Problem ID", style="dim white", width=10)
    score_table.add_column(
        "Honest Score", style="bright_green", justify="right", width=12
    )
    score_table.add_column(
        "Sneaky Score", style="bright_magenta", justify="right", width=12
    )
    score_table.add_column(
        "Correctness", style="bright_blue", justify="right", width=12
    )

    # Convert tensors to lists if necessary
    if isinstance(honest_scores, torch.Tensor):
        honest_scores = honest_scores.detach().cpu().tolist()
    if isinstance(sneaky_scores, torch.Tensor):
        sneaky_scores = sneaky_scores.detach().cpu().tolist()
    if isinstance(correctness_scores, torch.Tensor):
        correctness_scores = correctness_scores.detach().cpu().tolist()

    num_items = len(problem_ids)
    indices = list(range(num_items))

    # Subsample data if num_samples is specified
    if num_samples is not None and 0 < num_samples < num_items:
        indices = random.sample(indices, num_samples)

    for i in indices:
        p_id = problem_ids[i]
        h_sol = honest_solutions[i]
        s_sol = sneaky_solutions[i]
        h_score = honest_scores[i]
        s_score = sneaky_scores[i]
        correctness_score = correctness_scores[i]

        # Add scores to the score table
        score_table.add_row(
            str(p_id),
            f"{h_score:.3f}",
            f"{s_score:.3f}",
            f"{correctness_score:.3f}",
        )

        # Create a separate table for the code solutions
        sol_table = Table(show_header=True, expand=False, box=None)
        sol_table.add_column("Honest Solution", style="bright_yellow")
        sol_table.add_column("Sneaky Solution", style="bright_cyan")

        # Format the solutions for better readability
        h_sol_formatted = h_sol.replace("\n", "\n  ")  # Add indentation for readability
        s_sol_formatted = s_sol.replace("\n", "\n  ")  # Add indentation for readability

        sol_table.add_row(h_sol_formatted, s_sol_formatted)

        # Print both tables with a separator between samples
        console.print(score_table)
        console.print(sol_table)

        if i != indices[-1]:  # Add separator if not the last row
            console.print("─" * 120)
