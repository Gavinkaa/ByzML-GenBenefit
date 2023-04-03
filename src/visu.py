import matplotlib.pyplot as plt
import csv
import os
from pathlib import Path

# ------ SETTINGS ------
INPUT_FOLDER = '../results/MNIST/batch_1100/'
OUTPUT_FOLDER = '../results/MNIST/batch_1100/'


def plot(nb_of_nodes: list[str], batch_size: list[str], nb_epochs: list[str], aggregators: list[str], files: list[str]):
    plt.rcParams['figure.figsize'] = (10, 8)

    for file in files:
        if file.endswith('.csv'):
            file_labels = file.split('_')
            file_nb_nodes = file_labels[1]
            file_nb_byz = file_labels[3]
            file_batch_size = file_labels[5]
            file_nb_epochs = file_labels[7]
            aggregator = file_labels[9].split('Aggregator')[0]
            # file_lr = file_labels[11]
            # file_seed = file_labels[13].split('.csv')[0]

            if file_nb_nodes not in nb_of_nodes or \
                    file_batch_size not in batch_size or \
                    file_nb_epochs not in nb_epochs or \
                    aggregator not in aggregators:
                continue

            accuracies_train = []
            accuracies_test = []
            with open(Path(INPUT_FOLDER, file), 'r') as csvfile:
                plots = csv.reader(csvfile, delimiter=',')
                # skip header
                next(plots)
                for row in plots:
                    accuracies_train.append(float(row[1]))
                    accuracies_test.append(float(row[2]))

            # plot the moving average
            accuracies_train_ma = []
            accuracies_test_ma = []
            ma_window = 10
            for i in range(len(accuracies_train)):
                if i < ma_window:
                    accuracies_train_ma.append(sum(accuracies_train[:i]) / (i + 1))
                    accuracies_test_ma.append(sum(accuracies_test[:i]) / (i + 1))
                else:
                    accuracies_train_ma.append(sum(accuracies_train[i - ma_window:i]) / ma_window)
                    accuracies_test_ma.append(sum(accuracies_test[i - ma_window:i]) / ma_window)

            # plot train in dashed line
            # for test use the same color as the train

            plt.plot(accuracies_train_ma, label=f'TRAIN nodes: {file_nb_nodes}, byz: {file_nb_byz}, agg: {aggregator}',
                     linestyle='dashed')

            plt.plot(accuracies_test_ma, label=f'TEST nodes: {file_nb_nodes}, byz: {file_nb_byz}, agg: {aggregator}',
                     linestyle='solid',
                     color=plt.gca().lines[-1].get_color())

    # check if there is at least one plot
    if len(plt.gca().lines) == 0:
        return

    plt.ylim(0.95, 1)
    plt.legend()
    plt.title(f'Accuracy evolution of {[x for x in aggregators if x != "None"][0]} with {nb_of_nodes} nodes, '
              f'{nb_epochs} epochs and batch size {batch_size}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    filename = f'nodes_{nb_of_nodes}_epochs_{nb_epochs}_batch_{batch_size}_agg_' \
               f'{[a for a in aggregators if a != "None"][0]}.png'

    plt.savefig(Path(OUTPUT_FOLDER, filename))
    # plt.show()
    plt.clf()


def main():
    # first we list all different values of the parameters
    files = os.listdir(Path(INPUT_FOLDER))
    # make sure we have a predictable order
    files.sort()
    # if filename contains 'None' put it at the beginning
    files = [file for file in files if 'None' in file] + [file for file in files if
                                                          'None' not in file]

    nb_nodes = set()
    batch_sizes = set()
    epochs = set()
    agg = set()
    # lr = set()
    # seed = set()

    for file in files:
        if file.endswith('.csv'):
            labels = file.split('_')
            nb_nodes.add(labels[1])
            batch_sizes.add(labels[5])
            epochs.add(labels[7])
            agg.add(labels[9].split('Aggregator')[0])
            # lr.add(labels[11])
            # seed.add(labels[13].split('.csv')[0])

    agg.remove('None')

    for nb_of_nodes in nb_nodes:
        for batch_size in batch_sizes:
            for nb_epochs in epochs:
                for aggregator in agg:
                    if aggregator != 'None':
                        plot([nb_of_nodes], [batch_size], [nb_epochs], ['None', aggregator], files)


if __name__ == '__main__':
    main()
