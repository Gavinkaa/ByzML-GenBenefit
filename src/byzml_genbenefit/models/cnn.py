import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_MNIST(nn.Module):
    def __init__(self, device):
        super(CNN_MNIST, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)

        self.fc1 = nn.Linear(in_features=64 * 5 * 5, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

        self.device = device

    def forward(self, x):
        # [50, 1, 28, 28]
        x = F.relu(self.conv1(x))
        # [50, 32, 26, 26]
        x = F.max_pool2d(x, 2)
        # [50, 32, 13, 13]
        x = F.relu(self.conv2(x))
        # [50, 64, 11, 11]
        x = F.max_pool2d(x, 2)
        # [50, 64, 5, 5]

        x = torch.flatten(x, 1)
        # [50, 64 * 5 * 5]
        x = F.relu(self.fc1(x))
        x = F.log_softmax(self.fc2(x), dim=1)
        return x


class CNN_CIFAR10(nn.Module):
    def __init__(self, device):
        super(CNN_CIFAR10, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1)

        self.fc1 = nn.Linear(in_features=128 * 2 * 2, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=10)

        self.device = device

    def forward(self, x):
        # [50, 3, 32, 32]
        x = F.relu(self.conv1(x))
        # [50, 32, 30, 30]
        x = F.max_pool2d(x, 2)
        # [50, 32, 15, 15]
        x = F.relu(self.conv2(x))
        # [50, 64, 13, 13]
        x = F.max_pool2d(x, 2)
        # [50, 64, 6, 6]
        x = F.relu(self.conv3(x))
        # [50, 128, 4, 4]
        x = F.max_pool2d(x, 2)
        # [50, 128, 2, 2]

        x = torch.flatten(x, 1)
        # [50, 512]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)
        return x
