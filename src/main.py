import torch
from torch.nn import functional as F

from byzml_genbenefit import utils
from byzml_genbenefit.aggregator import mean
from byzml_genbenefit.data.mnist import get_data_loader
from byzml_genbenefit.models.cnn import CNN_MNIST as CNN
from byzml_genbenefit.models.nn import NN_MNIST as NN
from byzml_genbenefit.train.trainer import train, train_with_aggregation

# --- Hyper-parameters ---
# model = CNN()
model = NN()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
loss_fn = F.cross_entropy
aggregate_fn = mean.aggregate
num_epochs = 3
nb_simulated_byzantine_nodes = 10
batch_size = 100
# ------------------------

train_loader, test_loader = get_data_loader(batch_size=batch_size, shuffle_train=True, shuffle_test=False)
# train(model, optimizer, loss_fn, train_loader, num_epochs)
train_with_aggregation(model, optimizer, loss_fn, train_loader, num_epochs, aggregate_fn, nb_simulated_byzantine_nodes)

accuracy, num_correct, num_samples = utils.compute_accuracy(test_loader, model)
print(f'Accuracy: {accuracy} ({num_correct}/{num_samples})')
