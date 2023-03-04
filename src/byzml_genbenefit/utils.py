import torch


def compute_accuracy(test_data_loader: torch.utils.data.DataLoader, model: torch.nn.Module) -> tuple[float, int, int]:
    """Computes the accuracy of the model on the test data set

    Args:
        test_data_loader (torch.utils.data.DataLoader): The test data loader
        model (torch.nn.Module): The model to test

    Returns:
        float: The accuracy of the model
        int: The number of correct predictions'
        int: The number of samples
    """
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():  # reduce memory consumption
        for x, y in test_data_loader:
            # TODO check if we can send the data to the GPU
            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    return float(num_correct) / float(num_samples), num_correct, num_samples
