import torch

from byzml_genbenefit.aggregators import mean
from test_byzml_genbenefit.test_aggregators.utils import check_consistency_list_tensor

aggregator = mean.MeanAggregator()


def test_aggregate():
    # Create some fake gradient tensors to pass to the function
    grad1 = torch.tensor([1, 2, 3], dtype=torch.float32)
    grad2 = torch.tensor([2, 4, 6], dtype=torch.float32)
    grad3 = torch.tensor([3, 6, 9], dtype=torch.float32)

    # Call the function with the fake gradients
    gradients = [[grad1, grad2], [grad2, grad3], [grad1, grad3]]
    result = aggregator(gradients, 0)

    # Check that the output has the expected shape
    assert len(result) == 2
    assert result[0].shape == torch.Size([3])
    assert result[1].shape == torch.Size([3])

    expected_output = [torch.tensor([(1 + 2 + 1) / 3, (2 + 4 + 2) / 3, (3 + 6 + 3) / 3]),
                       torch.tensor([(2 + 3 + 3) / 3, (4 + 6 + 6) / 3, (6 + 9 + 9) / 3])]
    check_consistency_list_tensor(aggregator, gradients, 0, expected_output)


def test_aggregate_with_empty_list():
    # Call the function with an empty list
    gradients = []
    result = aggregator(gradients, 0)

    # Check that the output is an empty list
    assert result == []
