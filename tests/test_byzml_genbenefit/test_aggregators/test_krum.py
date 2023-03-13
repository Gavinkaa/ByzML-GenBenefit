import pytest
import torch

from byzml_genbenefit.aggregators.krum import KrumAggregator

aggregator = KrumAggregator()


def test_aggregate():
    gradients = [[torch.tensor([.1, .2, .3]), torch.tensor([.2, .3, .4])],
                 [torch.tensor([.2, .3, .4]), torch.tensor([.3, .4, .5])],
                 [torch.tensor([.3, .4, .5]), torch.tensor([.4, .5, .6])],
                 [torch.tensor([-.9, -.9, -.9]), torch.tensor([-.9, -.9, -.9])]]

    expected_aggregated_gradient = [torch.tensor([.2, .3, .4]), torch.tensor([.3, .4, .5])]

    aggregated_gradient = aggregator(gradients, 1)

    # aggregated_gradient should be a list of tensors

    assert all([isinstance(tensor, torch.Tensor) for tensor in aggregated_gradient])
    assert all([tensor.shape == expected_tensor.shape for tensor, expected_tensor in
                zip(aggregated_gradient, expected_aggregated_gradient)])
    assert all([torch.allclose(tensor, expected_tensor) for tensor, expected_tensor in
                zip(aggregated_gradient, expected_aggregated_gradient)])
