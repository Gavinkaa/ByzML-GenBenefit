import torch

from byzml_genbenefit.aggregators.mean import MeanAggregator
from byzml_genbenefit.aggregators.nnm import NNMAggregator


def check_consistency_list_list_tensor(outputs, expected_outputs):
    print(outputs)
    print(expected_outputs)

    for idx, output in enumerate(outputs):
        expected_output = expected_outputs[idx]

        assert all([isinstance(tensor, torch.Tensor) for tensor in output])
        assert all([tensor.shape == expected_tensor.shape for tensor, expected_tensor in
                    zip(output, expected_outputs[0])])
        assert all([torch.allclose(tensor, expected_tensor) for tensor, expected_tensor in
                    zip(output, expected_output)])


aggregator = NNMAggregator(MeanAggregator())


def test_apply_nnm_1():
    gradients = [[torch.tensor([.1, .2, .3]), torch.tensor([.2, .3, .4])],
                 [torch.tensor([.2, .3, .4]), torch.tensor([.3, .4, .5])],
                 [torch.tensor([.3, .4, .5]), torch.tensor([.4, .5, .6])],
                 [torch.tensor([-.9, -.9, -.9]), torch.tensor([-.9, -.9, -.9])]]

    expected_aggregated_gradient = [[torch.tensor([.1, .2, .3]), torch.tensor([.2, .3, .4])],
                                    [torch.tensor([.2, .3, .4]), torch.tensor([.3, .4, .5])],
                                    [torch.tensor([.3, .4, .5]), torch.tensor([.4, .5, .6])],
                                    [torch.tensor([-.9, -.9, -.9]), torch.tensor([-.9, -.9, -.9])]]

    outputs = aggregator._apply_nnm(gradients, 3)

    check_consistency_list_list_tensor(outputs, expected_aggregated_gradient)


def test_apply_nnm_2():
    gradients = [[torch.tensor([.1, .2, .3]), torch.tensor([.2, .3, .4])],
                 [torch.tensor([.2, .3, .4]), torch.tensor([.3, .4, .5])],
                 [torch.tensor([.5, .5, .5]), torch.tensor([.5, .5, .5])],
                 [torch.tensor([-.9, -.9, -.9]), torch.tensor([-.9, -.9, -.9])]]

    expected_aggregated_gradient = [[torch.tensor([.15, .25, .35]), torch.tensor([.25, .35, .45])],
                                    [torch.tensor([.15, .25, .35]), torch.tensor([.25, .35, .45])],
                                    [torch.tensor([.35, .4, .45]), torch.tensor([.4, .45, .5])],
                                    [torch.tensor([-.4, -.35, -.3]), torch.tensor([-.35, -.3, -.25])]]

    outputs = aggregator._apply_nnm(gradients, 2)

    check_consistency_list_list_tensor(outputs, expected_aggregated_gradient)


def test_apply_nnm_3():
    gradients = [[torch.tensor([.1, .2, .3]), torch.tensor([.2, .3, .4])],
                 [torch.tensor([.2, .3, .4]), torch.tensor([.3, .4, .5])],
                 [torch.tensor([.5, .5, .5]), torch.tensor([.5, .5, .5])],
                 [torch.tensor([-.9, -.9, -.9]), torch.tensor([-.9, -.9, -.9])]]

    expected_aggregated_gradient = [[torch.tensor([.15, .25, .35]), torch.tensor([.25, .35, .45])],
                                    [torch.tensor([.15, .25, .35]), torch.tensor([.25, .35, .45])],
                                    [torch.tensor([.35, .4, .45]), torch.tensor([.4, .45, .5])],
                                    [torch.tensor([-.4, -.35, -.3]), torch.tensor([-.35, -.3, -.25])]]

    outputs = aggregator._apply_nnm(gradients, 2)

    check_consistency_list_list_tensor(outputs, expected_aggregated_gradient)


def test_apply_nnm_4():
    gradients = [[torch.tensor([.1, .2, .3]), torch.tensor([.2, .3, .4])],
                 [torch.tensor([.2, .3, .4]), torch.tensor([.3, .4, .5])],
                 [torch.tensor([.5, .5, .5]), torch.tensor([.5, .5, .5])],
                 [torch.tensor([-.9, -.9, -.9]), torch.tensor([-.9, -.9, -.9])]]

    expected_aggregated_gradient = [[torch.tensor([.8/3, 1.0/3, 1.2/3]), torch.tensor([1.0/3, 1.2/3, 1.4/3])],
                                    [torch.tensor([.8/3, 1.0/3, 1.2/3]), torch.tensor([1.0/3, 1.2/3, 1.4/3])],
                                    [torch.tensor([.8/3, 1.0/3, 1.2/3]), torch.tensor([1.0/3, 1.2/3, 1.4/3])],
                                    [torch.tensor([-.6/3, -.4/3, -.2/3]), torch.tensor([-.4/3, -.2/3, 0])]]

    outputs = aggregator._apply_nnm(gradients, 1)

    check_consistency_list_list_tensor(outputs, expected_aggregated_gradient)
