import torch

from byzml_genbenefit.aggregators import aggregator


class CWTMAggregator(aggregator.Aggregator):

    def __repr__(self):
        return 'CWTMAggregator'

    @staticmethod
    def __call__(gradients: list, f: int) -> list:
        """Aggregates the gradients using the CWTM (coordinate-wise trimmed mean) algorithm.

        Args:
            gradients (list): The list of gradients to aggregate
            f (int): The number of Byzantine nodes

        Returns:
            list: The aggregated gradients
        """

        gradients_as_tensor, shapes = CWTMAggregator._gradient_as_tensor(gradients)

        # for each coordinate, sort the gradients in ascending order, and take the average of the ones
        # between the [f+1, n-f] positions
        trimmed_mean = torch.zeros_like(gradients_as_tensor[0])
        for i in range(len(gradients_as_tensor)):
            sorted_gradients, _ = torch.sort(gradients_as_tensor[i])
            trimmed_mean[i] = torch.mean(sorted_gradients[f:len(gradients_as_tensor) - f])

        return CWTMAggregator._tensor_as_gradient([trimmed_mean], shapes)[0]

