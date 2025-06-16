import torch


# Edited version to handle 2D tensors
def nanstd(
    tensor: torch.Tensor,
    dim: int | None = None,
    keepdim: bool = False,
    correction: int = 1,
) -> torch.Tensor:
    """
    Compute the standard deviation of a tensor along a given dimension, ignoring NaNs.

    Args:
        tensor (`torch.Tensor`): Input tensor.
        dim (`int`, *optional*): Dimension along which to compute the standard deviation.
                                  If None, compute over the entire tensor.
        keepdim (`bool`): Whether the output tensor has `dim` retained or not.
        correction (`int`): Difference between the sample size and sample degrees of freedom.
                             Defaults to 1 (Bessel's correction).

    Returns:
        `torch.Tensor`: Standard deviation of the tensor, ignoring NaNs.
    """
    # Calculate mean, keeping dim for broadcasting
    mean = torch.nanmean(tensor, dim=dim, keepdim=True)

    # Calculate squared deviations, propagating NaNs correctly
    # Where tensor is NaN, deviation should be NaN
    # Where tensor is not NaN, deviation is (tensor - mean)**2
    squared_dev = torch.where(torch.isnan(tensor), torch.nan, (tensor - mean) ** 2)

    # Calculate variance (mean of squared deviations), keeping dim temporarily for correction calculation
    variance = torch.nanmean(
        squared_dev, dim=dim, keepdim=True
    )  # Always keepdim=True here

    # Adjust for Bessel's correction if needed
    if correction != 0:
        # Count non-NaN elements along the dimension
        # Need keepdim=True for broadcasting with variance
        count = torch.sum(~torch.isnan(tensor), dim=dim, keepdim=True)
        # Ensure we don't divide by zero or negative numbers
        n = count.clamp(min=correction)
        factor = n / (n - correction).clamp(min=1e-8)  # Shape will have dim kept
        variance = (
            variance * factor
        )  # Both variance and factor have dim kept, broadcasting works

    # Apply final sqrt
    std_dev = torch.sqrt(variance.clamp(min=0))

    # Squeeze dimension only at the end if keepdim was False
    if dim is not None and not keepdim:
        std_dev = std_dev.squeeze(dim)

    return std_dev


def compute_entropy(
    logits: torch.Tensor,
    reduce: bool = True,
) -> torch.Tensor:
    """
    Calculate entropy from logits: H(p) = -sum(p_i * log(p_i))
    Args:
        logits: Raw logits from model (before softmax), shape (batch_size, sequence_length, vocab_size)
        mask: Optional mask to apply (for ignoring padding tokens), shape (batch_size, sequence_length)
        reduce: If True, return mean entropy; if False, return per-token entropy tensor
    Returns:
        If reduce=True: Mean entropy as scalar tensor
        If reduce=False: Per-token entropy tensor of shape (batch_size, sequence_length)
    """
    # Use log_softmax for better numerical stability
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)
    entropy = -(probs * log_probs).sum(dim=-1)  # More stable computation

    if not reduce:
        return entropy

    return entropy.mean()


def nanmin(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the minimum value of a tensor, ignoring NaNs. This function only supports 1D tensors.

    Args:
        tensor (`torch.Tensor`): Input tensor of shape `(N,)`.

    Returns:
        `torch.Tensor`: Minimum value of the tensor, ignoring NaNs. Returns NaN if all values are NaN.
    """
    if torch.isnan(tensor).all():
        return torch.tensor(float("nan"), dtype=tensor.dtype, device=tensor.device)
    return torch.min(tensor[~torch.isnan(tensor)])


def nanmax(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the maximum value of a tensor, ignoring NaNs. This function only supports 1D tensors.

    Args:
        tensor (`torch.Tensor`): Input tensor of shape `(N,)`.

    Returns:
        `torch.Tensor`: Maximum value of the tensor, ignoring NaNs. Returns NaN if all values are NaN.
    """
    if torch.isnan(tensor).all():
        return torch.tensor(float("nan"), dtype=tensor.dtype, device=tensor.device)
    return torch.max(tensor[~torch.isnan(tensor)])
