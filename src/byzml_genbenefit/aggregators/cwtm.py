import torch

from byzml_genbenefit.aggregators import aggregator


class CWTMAggregator(aggregator.Aggregator):

    def __repr__(self):
        return 'CWTMAggregator'

    @staticmethod
    def __call__(gradients: list, f: int) -> list:
        """Aggregates the gradients using the CWTM (coordinate-wise trimmed mean) algorithm.
            We remove the f lowest and f the highest values, and then take the mean of the remaining ones.

        Args:
            gradients (list): The list of gradients to aggregate
            f (int): The number of Byzantine nodes

        Returns:
            list: The aggregated gradients
        """

        aggregated_gradients = []
        for i, grad_list in enumerate(zip(*gradients)):
            # Compute the element-wise average of the tensors in grad_list
            stacked_tensor = torch.stack(grad_list)

            # Remove the f lowest and f highest values in dimension 0
            stacked_tensor, _ = torch.sort(stacked_tensor, dim=0)
            stacked_tensor = stacked_tensor[f:-f]

            avg_tensor = torch.mean(stacked_tensor, dim=0)

            aggregated_gradients.append(avg_tensor)
        return aggregated_gradients
