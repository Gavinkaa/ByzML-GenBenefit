import torch

from byzml_genbenefit.aggregators.multi_krum import MultiKrumAggregator
from test_byzml_genbenefit.test_aggregators.utils import check_consistency_list_tensor

aggregator = MultiKrumAggregator()


def test_aggregate_1():
    gradients = [[torch.tensor([.1, .2, .3]), torch.tensor([.2, .3, .4])],
                 [torch.tensor([.2, .3, .4]), torch.tensor([.3, .4, .5])],
                 [torch.tensor([.3, .4, .5]), torch.tensor([.4, .5, .6])],
                 [torch.tensor([-.9, -.9, -.9]), torch.tensor([-.9, -.9, -.9])]]

    expected_aggregated_gradient = [torch.tensor([.2, .3, .4]), torch.tensor([.3, .4, .5])]

    check_consistency_list_tensor(aggregator, gradients, 1, expected_aggregated_gradient)


def test_aggregate_2():
    gradients = [[torch.tensor([.0, .0, .0]), torch.tensor([.0, .0, .0])],
                 [torch.tensor([.2, .3, .4]), torch.tensor([.3, .4, .5])],
                 [torch.tensor([.4, .3, .5]), torch.tensor([.0, .2, .7])],
                 [torch.tensor([-.9, -.9, -.9]), torch.tensor([-.9, -.9, -.9])]]

    expected_aggregated_gradient = [torch.tensor([.2, .2, .3]), torch.tensor([.1, .2, .4])]

    check_consistency_list_tensor(aggregator, gradients, 1, expected_aggregated_gradient)


def test_aggregate_3():
    gradients = [[torch.tensor([.1, .2, .3]), torch.tensor([.2, .3, .4])],
                 [torch.tensor([.2, .3, .4]), torch.tensor([.3, .4, .5])],
                 [torch.tensor([.3, .10, .5]), torch.tensor([.4, .5, .6])],
                 [torch.tensor([-.9, -.9, -.9]), torch.tensor([-.9, -.9, -.9])]]

    expected_aggregated_gradient = [torch.tensor([.15, .25, .35]), torch.tensor([.25, .35, .45])]

    check_consistency_list_tensor(aggregator, gradients, 2, expected_aggregated_gradient)


def test_aggregate_4():
    gradients = [[torch.tensor([.0, .0, .0]), torch.tensor([.0, .0, .0])],
                 [torch.tensor([.2, .2, .2]), torch.tensor([.2, .2, .2])],
                 [torch.tensor([.4, .3, .5]), torch.tensor([.10, .10, .7])],
                 [torch.tensor([-.9, -.9, -.9]), torch.tensor([-.9, -.9, -.9])],
                 [torch.tensor([-.9, -.9, -.9]), torch.tensor([-.9, -.9, -.9])]]

    expected_aggregated_gradient = [torch.tensor([-.9, -.9, -.9]), torch.tensor([-.9, -.9, -.9])]

    check_consistency_list_tensor(aggregator, gradients, 3, expected_aggregated_gradient)


def test_aggregate_5():
    gradients = [[torch.tensor([.0, .0, .0]), torch.tensor([.0, .0, .0])],
                 [torch.tensor([.2, .2, .2]), torch.tensor([.2, .2, .2])],
                 [torch.tensor([.4, .3, .5]), torch.tensor([.10, .10, .7])],
                 [torch.tensor([-.9, -.9, -.9]), torch.tensor([-.9, -.9, -.9])],
                 [torch.tensor([-.9, .9, -.9]), torch.tensor([-.9, .9, -.9])]]

    expected_aggregated_gradient = [torch.tensor([.1, .1, .1]), torch.tensor([.1, .1, .1])]

    check_consistency_list_tensor(aggregator, gradients, 3, expected_aggregated_gradient)
