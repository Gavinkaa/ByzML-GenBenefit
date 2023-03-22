import torch

from byzml_genbenefit.aggregators import aggregator


class GMAggregator(aggregator.Aggregator):
    max_iter = 20
    epsilon = 1e-6

    def __repr__(self):
        return 'GMAggregator'

    @staticmethod
    def __call__(gradients: list, f: int) -> list:
        """Aggregates the gradients using the GM (geometric median) algorithm.

        Args:
            gradients (list): The list of gradients to aggregate
            f (int): The number of Byzantine nodes

        Returns:
            list: The aggregated gradients
        """

        gradients_as_tensor, shapes = GMAggregator._gradient_as_tensor(gradients)

        # Geometric mean is the vector that minimizes the sum of the distances to the other vectors

        # Weiszfeld's algorithm

        x = torch.mean(torch.stack(gradients_as_tensor), dim=0)

        for _ in range(GMAggregator.max_iter):

            weights = [1 / torch.norm(x - g) for g in gradients_as_tensor]

            # x_new = torch.zeros(x.shape)
            # for i, g in enumerate(gradients_as_tensor):
            #     x_new += weights[i] * g
            #
            # x_new /= sum(weights)

            x_new = torch.sum(torch.stack([weights[i] * g for i, g in enumerate(gradients_as_tensor)]), dim=0) / sum(
                weights)

            if torch.norm(x_new - x) < GMAggregator.epsilon:
                break

            x = x_new

        return GMAggregator._tensor_as_gradient([x], shapes)[0]
