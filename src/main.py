import torch
from torch.nn import functional as F
from tqdm import tqdm

from byzml_genbenefit import utils
from byzml_genbenefit.aggregators import mean
from byzml_genbenefit.aggregators.aggregator import Aggregator
from byzml_genbenefit.data.mnist import get_data_loader
from byzml_genbenefit.models.cnn import CNN_MNIST as CNN
from byzml_genbenefit.models.nn import NN_MNIST as NN
from byzml_genbenefit.train.trainer import train, train_with_aggregation

# --- Hyper-parameters ---
# model = CNN()
model = NN()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
loss_fn = F.cross_entropy
aggregate_fn: Aggregator = mean.MeanAggregator()
num_epochs = 10
nb_simulated_byzantine_nodes = 10
batch_size = 1000
# ------------------------

train_loader, test_loader = get_data_loader(batch_size=batch_size, shuffle_train=True, shuffle_test=False)
# train(model, optimizer, loss_fn, train_loader, num_epochs)
# train_with_aggregation(model, optimizer, loss_fn, train_loader, num_epochs, aggregate_fn,
#                        nb_simulated_byzantine_nodes, show_tqdm=False)

accuracies_train = []
accuracies_test = []

for epoch in tqdm(range(num_epochs)):
    # train_with_aggregation(model, optimizer, loss_fn, train_loader, 1, aggregate_fn,
    #                        nb_simulated_byzantine_nodes, show_tqdm=False)
    train(model, optimizer, loss_fn, train_loader, 1, show_tqdm=False)
    accuracy, _, _ = utils.compute_accuracy(train_loader, model)
    accuracies_train.append(accuracy)

    accuracy, _, _ = utils.compute_accuracy(test_loader, model)
    accuracies_test.append(accuracy)

# plot accuracies
utils.plot_accuracies(accuracies_train, accuracies_test, (0.9, 1.0))
print(f'Final accuracy on test set: {accuracies_test[-1]}')
