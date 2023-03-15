import torch

from byzml_genbenefit.aggregators.cwtm import CWTMAggregator

aggregator = CWTMAggregator()


def check_consistency(gradients, f, expected_output):
    output = aggregator(gradients, f)
    assert all([isinstance(tensor, torch.Tensor) for tensor in output])
    assert all([tensor.shape == expected_tensor.shape for tensor, expected_tensor in
                zip(output, gradients[0])])
    assert all([torch.allclose(tensor, expected_tensor) for tensor, expected_tensor in
                zip(output, expected_output)])


def test_aggregate_1():
    gradients = [
        [torch.Tensor([1, 2, 3]), torch.Tensor([4, 5, 6]), torch.Tensor([7, 8, 9])],
        [torch.Tensor([2, 4, 6]), torch.Tensor([8, 10, 12]), torch.Tensor([14, 16, 18])],
        [torch.Tensor([3, 6, 9]), torch.Tensor([12, 15, 18]), torch.Tensor([21, 24, 27])],
    ]
    f = 1
    expected_output = [
        torch.Tensor([2, 4, 6]),
        torch.Tensor([8, 10, 12]),
        torch.Tensor([14, 16, 18]),
    ]
    check_consistency(gradients, f, expected_output)


def test_aggregate_2():
    gradients = [
        [torch.Tensor([1, 2, 3]), torch.Tensor([4, 5, 6]), torch.Tensor([7, 8, 9])],
        [torch.Tensor([2, 4, 6]), torch.Tensor([8, 10, 12]), torch.Tensor([14, 16, 18])],
        [torch.Tensor([3, 6, 9]), torch.Tensor([12, 15, 18]), torch.Tensor([21, 24, 27])],
        [torch.Tensor([4, 8, 12]), torch.Tensor([16, 20, 24]), torch.Tensor([28, 32, 36])],
    ]
    f = 1
    expected_output = [
        torch.Tensor([2.5, 5, 7.5]),
        torch.Tensor([10, 12.5, 15]),
        torch.Tensor([17.5, 20, 22.5]),
    ]

    check_consistency(gradients, f, expected_output)
