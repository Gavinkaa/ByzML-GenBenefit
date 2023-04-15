import torch

from byzml_genbenefit.aggregators.cwtm import CWTMAggregator
from test_byzml_genbenefit.test_aggregators.utils import check_consistency_list_tensor

aggregator = CWTMAggregator()


def test_aggregate_1():
    gradients = [
        [torch.Tensor([1, 2, 3]), torch.Tensor([8, 10, 12]), torch.Tensor([7, 8, 9])],
        [torch.Tensor([2, 4, 6]), torch.Tensor([4, 5, 6]), torch.Tensor([21, 24, 27])],
        [torch.Tensor([3, 6, 9]), torch.Tensor([12, 15, 18]), torch.Tensor([14, 16, 18])],
    ]
    f = 1
    expected_output = [
        torch.Tensor([2, 4, 6]),
        torch.Tensor([8, 10, 12]),
        torch.Tensor([14, 16, 18]),
    ]
    check_consistency_list_tensor(aggregator, gradients, f, expected_output)


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

    check_consistency_list_tensor(aggregator, gradients, f, expected_output)


def test_aggregate_3():
    gradients = [
        [torch.Tensor([1, 2, 3]), torch.Tensor([4, 5, 6]), torch.Tensor([7, 8, 9])],
    ]
    f = 1
    expected_output = [
        torch.Tensor([1, 2, 3]),
        torch.Tensor([4, 5, 6]),
        torch.Tensor([7, 8, 9]),
    ]

    check_consistency_list_tensor(aggregator, gradients, f, expected_output)
