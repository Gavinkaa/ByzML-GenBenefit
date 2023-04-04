import os
import torch
import torchvision
import torchvision.transforms as transforms

DATA_FOLDER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../..', 'data')


def get_dataset():
    """Returns the CIFAR10 dataset as a tuple of (train_dataset, test_dataset). This dataset is normalized.

    Returns:
        train_dataset (torchvision.datasets.CIFAR10): The training dataset
        test_dataset (torchvision.datasets.CIFAR10): The test dataset
    """

    # values where first taken from https://github.com/kuangliu/pytorch-cifar/issues/19 and then
    # verified using the code below (which is commented out).
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    train_dataset = torchvision.datasets.CIFAR10(root=DATA_FOLDER_PATH, train=True, transform=transform,
                                                 download=True)
    test_dataset = torchvision.datasets.CIFAR10(root=DATA_FOLDER_PATH, train=False, transform=transform,
                                                download=True)

    # print(train_dataset.data.shape)
    # print(train_dataset.data.mean(axis=(0, 1, 2)) / 255)
    # print(train_dataset.data.std(axis=(0, 1, 2)) / 255)
    # >>> [0.49139968 0.48215841 0.44653091]
    # >>> [0.24703223 0.24348513 0.26158784]

    return train_dataset, test_dataset


def get_data_loader(batch_size: int, shuffle_train: bool = True, shuffle_test: bool = False):
    """Returns the CIFAR10 dataset as a tuple of (train_loader, test_loader)

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
