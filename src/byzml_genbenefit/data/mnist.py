import os
import torch
import torchvision
import torchvision.transforms as transforms

DATA_FOLDER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../..', 'data')


def get_dataset():
    """Returns the MNIST dataset as a tuple of (train_dataset, test_dataset)

    Returns:
        train_dataset (torchvision.datasets.MNIST): The training dataset
        test_dataset (torchvision.datasets.MNIST): The test dataset
    """
    train_dataset = torchvision.datasets.MNIST(root=DATA_FOLDER_PATH, train=True, transform=transforms.ToTensor(),
                                               download=True)
    test_dataset = torchvision.datasets.MNIST(root=DATA_FOLDER_PATH, train=False, transform=transforms.ToTensor(),
                                              download=True)
    return train_dataset, test_dataset


def get_data_loader(batch_size: int, shuffle_train: bool = True, shuffle_test: bool = False):
    """Returns the MNIST dataset as a tuple of (train_loader, test_loader)

    Args:
        batch_size (int): The batch size to use for the data loader
        shuffle_train (bool): Whether to shuffle the training data. Default: True
        shuffle_test (bool): Whether to shuffle the test data. Default: False

    Returns:
        train_loader (torch.utils.data.DataLoader): The training data loader
        test_loader (torch.utils.data.DataLoader): The test data loader
    """
    train_dataset, test_dataset = get_dataset()
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=shuffle_train)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=shuffle_test)
    return train_loader, test_loader
