import torch

from byzml_genbenefit.aggregators import aggregator


class KrumAggregator(aggregator.Aggregator):

    def __repr__(self):
        return 'KrumAggregator'

    @staticmethod
    def _compute_distance_to_nearest_neighbors(distance_matrix: torch.Tensor, f: int) -> torch.Tensor:
        """Computes the distance to the n - f nearest neighbors of each gradient.

        Args:
            distance_matrix (torch.Tensor): The distance matrix
            f (int): The number of Byzantine nodes

        Returns:
            torch.Tensor: The distance to the n - f nearest neighbors of each gradient
        """
        n = len(distance_matrix)

        distance_to_nearest_neighbors = torch.zeros(n)
        for i, row in enumerate(distance_matrix):
            # Sort the row in ascending order
            sorted_row, _ = torch.sort(row)

            # Sum the n - f nearest neighbors
            distance_to_nearest_neighbors[i] = torch.nansum(sorted_row[:n - f])

        return distance_to_nearest_neighbors

    @staticmethod
    def __call__(gradients: list, f: int) -> list:
        """Aggregates the gradients using the Krum algorithm.

        Args:
            gradients (list): The list of gradients to aggregate
            f (int): The number of Byzantine nodes

        Returns:
            list: The aggregated gradients
        """
        # SOURCE: https://arxiv.org/pdf/2302.01772.pdf
        # ----------------------------------------------
        # Essentially, given the input vectors
        # x1, ... , xn, Krum outputs the vector that is the nearest to its neighbors upon discarding f
        # furthest vectors. Specifically, we denote by Nj the set the of indices of the n − f nearest neighbors of xj
        # in {x1, ... , xn}, with ties arbitrarily broken. Krum outputs the vector x_k∗ such that
        # k∗ ∈ arg min_{j∈[n]} sum_{i∈Nj} ‖xj − xi‖^2,
        # with ties arbitrarily broken if the set of minimizers above includes more than one element.
        # ----------------------------------------------

        gradients_as_tensor, shapes = KrumAggregator._gradient_as_tensor(gradients)

        # Compute the distance matrix
        distance_matrix = KrumAggregator._compute_distance_matrix(gradients_as_tensor)

        # For each gradient, compute the sum of the distances to its n - f nearest neighbors
        distances_to_neighbors = KrumAggregator._compute_distance_to_nearest_neighbors(distance_matrix, f)

        # Find the index of the gradient that is the nearest to its neighbors
        _, index = torch.min(distances_to_neighbors, dim=0)

        # This return is similar to the one below, but it's less efficient
        # return KrumAggregator._tensor_as_gradient(gradients_as_tensor, shapes)[index]

        return KrumAggregator._tensor_as_gradient([gradients_as_tensor[index]], shapes)[0]
