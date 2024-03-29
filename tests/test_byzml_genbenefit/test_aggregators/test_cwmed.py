import torch

from byzml_genbenefit.aggregators.cwmed import CWMedAggregator
from test_byzml_genbenefit.test_aggregators.utils import check_consistency_list_tensor

aggregator = CWMedAggregator()


def test_aggregate_1():
    gradients = [
        [torch.Tensor([1, 2, 3]), torch.Tensor([4, 5, 6]), torch.Tensor([7, 8, 9])],
        [torch.Tensor([2, 4, 6]), torch.Tensor([8, 10, 12]), torch.Tensor([14, 16, 18])],
        [torch.Tensor([3, 6, 9]), torch.Tensor([12, 15, 18]), torch.Tensor([21, 24, 27])],
    ]
    expected_output = [
        torch.Tensor([2, 4, 6]),
        torch.Tensor([8, 10, 12]),
        torch.Tensor([14, 16, 18]),
    ]
    check_consistency_list_tensor(aggregator, gradients, 0, expected_output)


def test_aggregate_2():
    gradients = [
        [torch.Tensor([1, 2, 3]), torch.Tensor([12, 15, 18]), torch.Tensor([21, 24, 27])],
        [torch.Tensor([2, 4, 6]), torch.Tensor([8, 10, 12]), torch.Tensor([14, 16, 18])],
        [torch.Tensor([3, 6, 9]), torch.Tensor([4, 5, 6]), torch.Tensor([28, 32, 36])],
        [torch.Tensor([4, 8, 12]), torch.Tensor([16, 20, 24]), torch.Tensor([7, 8, 9])],
    ]
    expected_output = [
        torch.Tensor([2, 4, 6]),
        torch.Tensor([8, 10, 12]),
        torch.Tensor([14, 16, 18]),
    ]

    check_consistency_list_tensor(aggregator, gradients, 0, expected_output)


def test_aggregate_3():
    gradients = [
        [torch.Tensor([1, 2, 3]), torch.Tensor([4, 5, 6]), torch.Tensor([7, 8, 9])]
    ]
    expected_output = [
        torch.Tensor([1, 2, 3]),
        torch.Tensor([4, 5, 6]),
        torch.Tensor([7, 8, 9]),
    ]
    check_consistency_list_tensor(aggregator, gradients, 0, expected_output)
