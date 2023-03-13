import torch
import random

from byzml_genbenefit.aggregators.aggregator import Aggregator


class TestGradients:

    def test_gradient_as_tensor_with_random_gradients(self):
        for i in range(5):
            # creation of the input (list of list of tensors)
            expected_shapes = [(random.randint(1, 100), random.randint(1, 100)) for _ in range(5)]
            gradients = [[torch.randn(*shape) for shape in expected_shapes] for _ in range(3)]

            flattened_tensors, shapes = Aggregator._gradient_as_tensor(gradients)

            assert isinstance(flattened_tensors, list)
            assert len(flattened_tensors) == len(gradients)

            assert all([isinstance(tensor, torch.Tensor) for tensor in flattened_tensors])
            assert shapes == expected_shapes

    def test_tensor_as_gradient_is_the_inverse_of_gradient_as_tensor(self):
        for i in range(5):
            # creation of the input (list of list of tensors)
            expected_shapes = [(random.randint(1, 100), random.randint(1, 100)) for _ in range(5)]
            gradients = [[torch.randn(*shape) for shape in expected_shapes] for _ in range(3)]

            flattened_tensors, shapes = Aggregator._gradient_as_tensor(gradients)
            result = Aggregator._tensor_as_gradient(flattened_tensors, shapes)

            # check format and shape of the result
            assert isinstance(result, list)
            assert len(result) == len(gradients)
            assert all([isinstance(list_of_tensors, list) for list_of_tensors in result])
            assert all([len(list_of_tensors) == len(expected_gradients) for list_of_tensors, expected_gradients in
                        zip(result, gradients)])
            assert all([isinstance(tensor, torch.Tensor) for list_of_tensors in result for tensor in list_of_tensors])
            assert all([tensor.shape == expected_shape for list_of_tensors in
                        result for tensor, expected_shape in zip(list_of_tensors, expected_shapes)])

            # check that the tensors are the same
            assert all([torch.allclose(tensor, expected_tensor) for list_of_tensors, expected_list_of_tensors in
                        zip(result, gradients) for tensor, expected_tensor in
                        zip(list_of_tensors, expected_list_of_tensors)])
