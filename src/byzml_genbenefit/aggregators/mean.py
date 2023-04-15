import torch

from byzml_genbenefit.aggregators.aggregator import Aggregator


class MeanAggregator(Aggregator):

    def __repr__(self):
        return 'MeanAggregator'

    @staticmethod
    def __call__(gradients: list, f: int) -> list:
        """Aggregates the gradients by averaging them.

        Args:
            gradients (list): The list of gradients to aggregate
            f (int): The number of Byzantine nodes

        Returns:
            list: The aggregated gradients
        """
        aggregated_gradients = []
        for i, grad_list in enumerate(zip(*gradients)):
            # Compute the element-wise average of the tensors in grad_list
            avg_tensor = torch.nanmean(torch.stack(grad_list), dim=0)
            aggregated_gradients.append(avg_tensor)
        return aggregated_gradients
