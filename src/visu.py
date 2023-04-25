import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# ------ SETTINGS ------
INPUT_FOLDER = '../results/CIFAR-10'
OUTPUT_FOLDER = '../results/CIFAR-10'


def plot():
    plt.rcParams['figure.figsize'] = (10, 8)

    files = os.listdir(Path(INPUT_FOLDER))

    df = pd.DataFrame()

    for file in files:
        if not file.endswith('.csv'):
            continue
        file_labels = file.split('_')
        file_nb_nodes = file_labels[1]
        file_nb_byz = file_labels[3]
        file_batch_size = file_labels[5]
        file_nb_epochs = file_labels[7]
        aggregator = file_labels[9].split('Aggregator')[0]
        file_lr = file_labels[11]
        file_seed = file_labels[13].split('.csv')[0]

        local_df = pd.read_csv(Path(INPUT_FOLDER, file))
        local_df['nb_nodes'] = file_nb_nodes
        local_df['nb_byz'] = file_nb_byz
        local_df['batch_size'] = file_batch_size
        local_df['nb_epochs'] = file_nb_epochs
        local_df['aggregator'] = aggregator
        local_df['lr'] = file_lr
        local_df['seed'] = file_seed

        df = pd.concat([df, local_df])

    # Group the data by the tuple
    grouped = df.groupby(['nb_nodes', 'batch_size', 'nb_epochs', 'aggregator', 'lr'])

    # Loop over each group and plot the mean accuracy over epochs
    for name, group in grouped:
        nb_nodes = name[0]
        batch_size = name[1]
        nb_epochs = name[2]
        aggregator = name[3]
        lr = name[4]

        if aggregator == 'None':
            continue

        plt.figure()

        for nb_byz in group['nb_byz'].unique():
            group_byz = group[group['nb_byz'] == nb_byz]

            acc_test_mean = group_byz.groupby('epoch')['accuracy_test'].mean()
            acc_test_std = group_byz.groupby('epoch')['accuracy_test'].std()

            acc_train_mean = group_byz.groupby('epoch')['accuracy_train'].mean()
            acc_train_std = group_byz.groupby('epoch')['accuracy_train'].std()

            plt.plot(acc_train_mean.index, acc_train_mean.values, label=f'TRAIN - byz: {nb_byz}, agg: {aggregator}',
                     linestyle='dashed')
            plt.fill_between(acc_train_mean.index, acc_train_mean.values - acc_train_std.values,
                             acc_train_mean.values + acc_train_std.values, alpha=0.2)
            plt.plot(acc_test_mean.index, acc_test_mean.values, label=f'TEST - byz: {nb_byz}, agg: {aggregator}',
                     linestyle='solid', color=plt.gca().lines[-1].get_color())
            plt.fill_between(acc_test_mean.index, acc_test_mean.values - acc_test_std.values,
                             acc_test_mean.values + acc_test_std.values, alpha=0.2,
                             color=plt.gca().lines[-1].get_color())

        # add None aggregator
        group_none = df[
            (df['nb_nodes'] == nb_nodes) & (df['batch_size'] == batch_size) & (df['nb_epochs'] == nb_epochs) & (
                    df['aggregator'] == 'None') & (df['lr'] == lr)]
        acc_test_mean_none = group_none.groupby('epoch')['accuracy_test'].mean()
        acc_test_std_none = group_none.groupby('epoch')['accuracy_test'].std()

        acc_train_mean_none = group_none.groupby('epoch')['accuracy_train'].mean()
        acc_train_std_none = group_none.groupby('epoch')['accuracy_train'].std()

        plt.plot(acc_train_mean_none.index, acc_train_mean_none.values, label='TRAIN - agg: None',
                 linestyle='dashed')
        plt.fill_between(acc_train_mean_none.index, acc_train_mean_none.values - acc_train_std_none.values,
                         acc_train_mean_none.values + acc_train_std_none.values, alpha=0.2)
        plt.plot(acc_test_mean_none.index, acc_test_mean_none.values, label='TEST - agg: None',
                 linestyle='solid', color=plt.gca().lines[-1].get_color())
        plt.fill_between(acc_test_mean_none.index, acc_test_mean_none.values - acc_test_std_none.values,
                         acc_test_mean_none.values + acc_test_std_none.values, alpha=0.2,
                         color=plt.gca().lines[-1].get_color())

        plt.ylim(0.6, 1)
        plt.legend()

        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')

        plt.title(f'Accuracy evolution of {aggregator} with {nb_nodes} nodes, '
                  f'{nb_epochs} epochs and batch size {batch_size}')

        filename = f'z-nodes_{nb_nodes}_epochs_{nb_epochs}_batch_{batch_size}_agg_' \
                   f'{aggregator}.png'

        # plt.show()
        print((Path(OUTPUT_FOLDER, filename)))
        plt.savefig(Path(OUTPUT_FOLDER, filename))
        plt.close()
        # plt.clf()


def main():
    plot()


if __name__ == '__main__':
    main()
