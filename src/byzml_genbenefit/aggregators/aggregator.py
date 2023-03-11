from typing import Protocol


class Aggregator(Protocol):
    def aggregate(self, gradients: list, f: int) -> list:
        """Aggregates the gradients, where f is the number of Byzantine nodes.

        Args:
            gradients (list): The list of gradients to aggregate.
            f (int): The number of Byzantine nodes

        Returns:
            list: The aggregated gradients, ready to be applied to the model. (list of torch.Tensor)
        """
        ...
