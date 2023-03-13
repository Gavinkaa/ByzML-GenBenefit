from abc import ABC, abstractmethod
import torch


class Aggregator(ABC):
    """Abstract class for aggregators."""

    @staticmethod
    @abstractmethod
    def __call__(gradients: list, f: int) -> list:
        """Aggregates the gradients, where f is the number of Byzantine nodes.

        Args:
            gradients (list): The list of gradients to aggregate.
            f (int): The number of Byzantine nodes

        Returns:
            list: The aggregated gradients, ready to be applied to the model. (list of torch.Tensor)
        """
        ...

    @staticmethod
    def _gradient_as_tensor(gradients: list) -> tuple:
        """Converts a list of gradients (list of list of torch.Tensor) to a tuple of
            flattened tensors and their original shapes.

        Args:
            gradients (list): The list of gradients to transform

        Returns:
            tuple: A tuple of the flattened tensors and their original shapes. Shapes are
                stored as a list of torch.Size
        """
        flattened_tensors = []
        shapes = [tsr.shape for tsr in gradients[0]]

        for gradient in gradients:
            tmp_gradient = []
            for tsr in gradient:
                # Flatten the tensor
                tmp_gradient.append(tsr.view(-1, 1))
            # Concatenate the flattened tensors
            flattened_tensor = torch.cat(tmp_gradient, dim=0)
            flattened_tensors.append(flattened_tensor)

        return flattened_tensors, shapes

    @staticmethod
    def _tensor_as_gradient(flattened_tensors: list, shapes: list) -> list:
        """Converts a tuple of flattened tensors and their original shapes to a list of gradients
            (list of list of torch.Tensor)

        Args:
            flattened_tensors (list): The list of flattened tensors to transform
            shapes (list): The list of original shapes of the tensors. (list of torch.Size)

        Returns:
            list: A list of gradients (list of list of torch.Tensor)
        """

        gradients = []
        for tsr in flattened_tensors:
            gradient = []
            start = 0
            for shape in shapes:
                # shape.numel() returns the number of elements in the tensor
                end = start + shape.numel()
                gradient.append(tsr[start:end].view(shape))
                start = end
            gradients.append(gradient)

        return gradients
