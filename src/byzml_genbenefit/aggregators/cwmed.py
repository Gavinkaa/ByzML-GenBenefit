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
            # Compute the element-wise average of the tensors in grad_list
            avg_tensor = torch.median(torch.stack(grad_list), dim=0)
            aggregated_gradients.append(avg_tensor)
        return aggregated_gradients
