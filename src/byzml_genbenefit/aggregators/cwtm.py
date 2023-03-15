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
        # gradients_as_tensor is a list of tensors, each tensor is a list coordinates

        # TODO: implement this

        # trimmed_means = torch.zeros_like(gradients_as_tensor[0])
        # for i in range(gradients_as_tensor[0].shape[0]):
        #     tensor_coordinates = [g[i] for g in gradients_as_tensor]
        #     tensor_coordinates.sort()
        #     trimmed_mean = torch.mean(torch.stack(tensor_coordinates[f:-f]), dim=0)
        #     trimmed_means[i] = trimmed_mean
        #
        # return CWTMAggregator._tensor_as_gradient([trimmed_means], shapes)[0]

