import torch
from torch.nn import functional as F
from tqdm import tqdm

from byzml_genbenefit import utils
from byzml_genbenefit.aggregators.aggregator import Aggregator
from byzml_genbenefit.aggregators.cwmed import CWMedAggregator
from byzml_genbenefit.aggregators.cwtm import CWTMAggregator
from byzml_genbenefit.aggregators.gm import GMAggregator
from byzml_genbenefit.aggregators.krum import KrumAggregator
from byzml_genbenefit.data.mnist import get_data_loader
from byzml_genbenefit.models.cnn import CNN_MNIST as CNN
from byzml_genbenefit.models.nn import NN_MNIST as NN
from byzml_genbenefit.train.trainer import train, train_with_aggregation
import argparse

if __name__ == '__main__':

    # --- Argument parser ---

    aggregators = ['cwmed', 'cwtm', 'gm', 'krum', 'none']

    parser = argparse.ArgumentParser(description='Argument parser for Byzantine Machine Learning')
    parser.add_argument('--aggregator', choices=aggregators, required=False, default=None,
                        help='the type of aggregator to use')
    parser.add_argument('--epochs', type=int, required=False, default=100,
                        help='the number of epochs')
    parser.add_argument('--batch_size', type=int, required=False, default=100,
                        help='the batch size')
    parser.add_argument('--nb_of_nodes', type=int, required=False, default=5,
                        help='the number of nodes')
    parser.add_argument('--nb_of_byzantine_nodes', type=int, required=False, default=1,
                        help='the number of byzantine nodes')
    args = parser.parse_args()
    # ------------------------

    # --- Hyper-parameters ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNN(device)
    # model = NN()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = F.cross_entropy

    # choose the aggregator with a match
    match args.aggregator:
        case 'cwmed':
            aggregate_fn: Aggregator = CWMedAggregator()
        case 'cwtm':
            aggregate_fn: Aggregator = CWTMAggregator()
        case 'gm':
            aggregate_fn: Aggregator = GMAggregator()
        case 'krum':
            aggregate_fn: Aggregator = KrumAggregator()
        case _:
            aggregate_fn: Aggregator = None

    nb_epochs = args.epochs

    # Size of a batch, with or without aggregation, this represents the number
    # of data points that will be used to compute one gradient descent
    batch_size = args.batch_size

    # Number of simulated nodes (workers), they can be malicious or not
    nb_of_nodes = args.nb_of_nodes

    # Number of byzantine nodes (malicious), this number is (by def) <= nb_of_nodes
    nb_of_byzantine_nodes = args.nb_of_byzantine_nodes

    assert nb_of_byzantine_nodes <= nb_of_nodes
    # ------------------------

    # print the hyper-parameters
    print('Hyper-parameters:')
    print(f'  - nb_of_nodes: {nb_of_nodes}')
    print(f'  - nb_of_byzantine_nodes: {nb_of_byzantine_nodes}')
    print(f'  - nb_epochs: {nb_epochs}')
    print(f'  - batch_size: {batch_size}')
    print(f'  - aggregator: {aggregate_fn}')

    if aggregate_fn is None:
        train_loader, test_loader = get_data_loader(batch_size=batch_size, shuffle_train=True, shuffle_test=False)
    else:
        train_loader, test_loader = get_data_loader(batch_size=batch_size // nb_of_nodes, shuffle_train=True,
                                                    shuffle_test=False)

    accuracies_train = []
    accuracies_test = []

    model.to(device)

    for epoch in tqdm(range(nb_epochs)):
        if aggregate_fn is None:
            train(model, optimizer, loss_fn, train_loader, 1, show_tqdm=False)
        else:
            train_with_aggregation(model, optimizer, loss_fn, train_loader, 1, aggregate_fn, nb_of_nodes,
                                   nb_of_byzantine_nodes, show_tqdm=False)

        accuracy, _, _ = utils.compute_accuracy(train_loader, model)
        accuracies_train.append(accuracy)

        accuracy, _, _ = utils.compute_accuracy(test_loader, model)
        accuracies_test.append(accuracy)

    # plot accuracies
    # utils.plot_accuracies(accuracies_train, accuracies_test, accuracy_range=(0.9, 1.0),
    #                       title=f'Accuracy on MNIST dataset, using {aggregate_fn},'
    #                             f'\n{nb_of_nodes} nodes, '
    #                             f'{nb_of_byzantine_nodes} byzantine nodes and {batch_size} batch size', save=False)
    print(f'Final accuracy on test set: {accuracies_test[-1]}')

    # # Save the accuracies in a csv file
    # filename = f'./results/nodes_{nb_of_nodes}_byz_{nb_of_byzantine_nodes}_batch_' \
    #            f'{batch_size}_epochs_{nb_epochs}_agg_{aggregate_fn}.csv'
    # utils.save_accuracies_to_csv(accuracies_train, accuracies_test, filename)
