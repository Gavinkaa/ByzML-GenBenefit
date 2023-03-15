import torch


def check_consistency_list_tensor(aggregator, gradients, f, expected_output):
    output = aggregator(gradients, f)
    assert all([isinstance(tensor, torch.Tensor) for tensor in output])
    assert all([tensor.shape == expected_tensor.shape for tensor, expected_tensor in
                zip(output, gradients[0])])
    assert all([torch.allclose(tensor, expected_tensor) for tensor, expected_tensor in
                zip(output, expected_output)])
