import torch

from byzml_genbenefit.aggregators import aggregator as aggregator_module


class NNMAggregator(aggregator_module.Aggregator):

    def __init__(self, aggregator: aggregator_module.Aggregator):
        self._aggregator = aggregator

    def __repr__(self):
        return 'NNMAggregator({})'.format(self._aggregator)

    @staticmethod
    def _apply_nnm(gradients: list, f: int) -> list:
        """Replace each gradient by the average of its n-f nearest neighbors.

        Args:
            gradients (list): The list of gradients to aggregate
            f (int): The number of Byzantine nodes

        Returns:
            list: The new list of gradients
        """

        gradients_as_tensor, shapes = NNMAggregator._gradient_as_tensor(gradients)

        # Compute the distance matrix
        distance_matrix = NNMAggregator._compute_distance_matrix(gradients_as_tensor)

        # For each gradient, replace it by the average of its n - f nearest neighbors
        new_gradients = []
        for i in range(len(gradients)):
            _, indices = torch.sort(distance_matrix[i])
            indices = indices[:len(gradients) - f]
            new_gradients.append(torch.nanmean(torch.stack([gradients_as_tensor[j] for j in indices]), dim=0))

        return NNMAggregator._tensor_as_gradient(new_gradients, shapes)

    def __call__(self, gradients: list, f: int) -> list:
        """Aggregates the gradients using the NNM algorithm before applying the given aggregator.

        Args:
            gradients (list): The list of gradients to aggregate
            f (int): The number of Byzantine nodes

        Returns:
            list: The aggregated gradients
        """

        # Apply the NNM algorithm
        gradients = NNMAggregator._apply_nnm(gradients, f)

        # Apply the given aggregator
        return self._aggregator(gradients, f)
