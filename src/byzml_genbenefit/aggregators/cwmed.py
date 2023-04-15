import torch

from byzml_genbenefit.aggregators import aggregator


class CWMedAggregator(aggregator.Aggregator):

    def __repr__(self):
        return 'CWMedAggregator'

    @staticmethod
    def __call__(gradients: list, f: int) -> list:
        """Aggregates the gradients using the CWMed (coordinate-wise median) algorithm.

        Args:
            gradients (list): The list of gradients to aggregate
            f (int): The number of Byzantine nodes

        Returns:
            list: The aggregated gradients
        """

        aggregated_gradients = []
        for i, grad_list in enumerate(zip(*gradients)):
            # Compute the element-wise median_tensor of the tensors in grad_list
            median_tensor, _ = torch.nanmedian(torch.stack(grad_list), dim=0)
            aggregated_gradients.append(median_tensor)
        return aggregated_gradients
