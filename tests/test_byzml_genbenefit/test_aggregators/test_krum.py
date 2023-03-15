import torch

from byzml_genbenefit.aggregators.krum import KrumAggregator
from test_byzml_genbenefit.test_aggregators.utils import check_consistency_list_tensor

aggregator = KrumAggregator()


def test_aggregate():
    gradients = [[torch.tensor([.1, .2, .3]), torch.tensor([.2, .3, .4])],
                 [torch.tensor([.2, .3, .4]), torch.tensor([.3, .4, .5])],
                 [torch.tensor([.3, .4, .5]), torch.tensor([.4, .5, .6])],
                 [torch.tensor([-.9, -.9, -.9]), torch.tensor([-.9, -.9, -.9])]]

    expected_aggregated_gradient = [torch.tensor([.2, .3, .4]), torch.tensor([.3, .4, .5])]

    check_consistency_list_tensor(aggregator, gradients, 1, expected_aggregated_gradient)
