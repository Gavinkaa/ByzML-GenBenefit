import torch.nn as nn
import torch.utils.data
from tqdm import tqdm


def train(model: nn.Module, optimizer: torch.optim.Optimizer, loss_fn: callable,
          train_loader: torch.utils.data.DataLoader, nb_epochs: int, show_tqdm: bool = True):
    """Trains the model on the training data, with classical gradient descent.

    Args:
        model (torch.nn.Module): The model to train
        optimizer (torch.optim.Optimizer): The optimizer to use for training
        loss_fn (callable): The loss function to use for training
        train_loader (torch.utils.data.DataLoader): The training data loader
        nb_epochs (int): The number of epochs to train the model for
        show_tqdm (bool): Whether to use tqdm for progress bar. Default: True
    """

    model.train()

    for epoch in tqdm(range(nb_epochs), disable=not show_tqdm):
        for batch_idx, (data, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(data)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()


def train_with_aggregation(model: nn.Module, optimizer: torch.optim.Optimizer, loss_fn: callable,
                           train_loader: torch.utils.data.DataLoader, nb_epochs: int, aggregate_fn: callable,
                           nb_simulated_byzantine_nodes: int, show_tqdm: bool = True):
    """Trains the model on the training data, aggregating the gradients by the specified function.

    Args:
        model (torch.nn.Module): The model to train
        optimizer (torch.optim.Optimizer): The optimizer to use for training
        loss_fn (callable): The loss function to use for training
        train_loader (torch.utils.data.DataLoader): The training data loader
        nb_epochs (int): The number of epochs to train the model for
        aggregate_fn (callable): The function to aggregate the gradients
        nb_simulated_byzantine_nodes (int): The number of simulated Byzantine nodes
        show_tqdm (bool): Whether to use tqdm to display the progress bar
    """

    model.train()

    for epoch in tqdm(range(nb_epochs), disable=not show_tqdm):
        gradients = []  # Create an empty list to store the gradients
        for batch_idx, (data, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(data)
            loss = loss_fn(outputs, targets)
            loss.backward()
            gradients.append([param.grad for param in model.parameters()])  # Append the gradients to the list

            if batch_idx % nb_simulated_byzantine_nodes == 0:
                # Simulate Byzantine nodes
                aggregated_gradients = aggregate_fn(gradients)
                for i, param in enumerate(model.parameters()):
                    param.grad = aggregated_gradients[i]

                optimizer.step()
                gradients = []  # Reset the list of gradients
                continue

        if len(gradients) > 0:
            # Simulate Byzantine nodes
            aggregated_gradients = aggregate_fn(gradients)
            for i, param in enumerate(model.parameters()):
                param.grad = aggregated_gradients[i]
            optimizer.step()
