from transformers import PreTrainedModel


def check_model_equivalence(model1: PreTrainedModel, model2: PreTrainedModel) -> bool:
    """
    Check if two models have the same backbone (e.g. Qwen2.5-0.5B-Instruct and Qwen2.5-0.5B-Instruct)
    Args:
        model1: First model to compare
        model2: Second model to compare
    Returns:
        True if the models have the same backbone, False otherwise
    """
    return model1.config._name_or_path == model2.config._name_or_path
