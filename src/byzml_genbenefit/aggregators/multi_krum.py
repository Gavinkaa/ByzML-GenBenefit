import torch

from byzml_genbenefit.aggregators import aggregator


class MultiKrumAggregator(aggregator.Aggregator):

    def __repr__(self):
        return 'MultiKrumAggregator'

    @staticmethod
    def __call__(gradients: list, f: int) -> list:
        """Aggregates the gradients using the Multi-Krum algorithm.
        Note that in this implementation we set m = n - f.

        Args:
            gradients (list): The list of gradients to aggregate
            f (int): The number of Byzantine nodes

        Returns:
            list: The aggregated gradients
        """

        gradients_as_tensor, shapes = MultiKrumAggregator._gradient_as_tensor(gradients)

        m = len(gradients) - f

        # Compute the distance matrix
        distance_matrix = MultiKrumAggregator._compute_distance_matrix(gradients_as_tensor)

        # For each gradient, compute the sum of the distances to its n - f nearest neighbors
        distances_to_neighbors = MultiKrumAggregator._compute_distance_to_nearest_neighbors(distance_matrix, f)

        # Find the m gradients that are the nearest to their neighbors
        _, indices = torch.sort(distances_to_neighbors)
        indices = indices[:m]

        # Return the average of the m gradients that are the nearest to their neighbors
        average = torch.nanmean(torch.stack([gradients_as_tensor[i] for i in indices]), dim=0)

        return MultiKrumAggregator._tensor_as_gradient([average], shapes)[0]
