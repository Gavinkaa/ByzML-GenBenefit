import torch


def aggregate(gradients: list) -> list:
    """Aggregates the gradients by averaging them.

    Args:
        gradients (list): The list of gradients to aggregate

    Returns:
        list: The aggregated gradients
    """
    aggregated_gradients = []
    for i, grad_list in enumerate(zip(*gradients)):
        # Compute the element-wise average of the tensors in grad_list
        avg_tensor = torch.mean(torch.stack(grad_list), dim=0)
        aggregated_gradients.append(avg_tensor)
    return aggregated_gradients
