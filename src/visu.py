import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# ------ SETTINGS ------
INPUT_FOLDER = '../results/CIFAR-10/raw_data'
OUTPUT_FOLDER = '../results/CIFAR-10/graphs'


def plot():
    plt.rcParams['figure.figsize'] = (10, 8 * 3)

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

        fig, ax = plt.subplots(3, 1)

        # add None aggregator for accuracy
        group_none = df[
            (df['nb_nodes'] == nb_nodes) & (df['batch_size'] == batch_size) & (df['nb_epochs'] == nb_epochs) & (
                    df['aggregator'] == 'None') & (df['lr'] == lr)]
        acc_test_mean_none = group_none.groupby('epoch')['accuracy_test'].mean()
        acc_test_std_none = group_none.groupby('epoch')['accuracy_test'].std()

        acc_train_mean_none = group_none.groupby('epoch')['accuracy_train'].mean()
        acc_train_std_none = group_none.groupby('epoch')['accuracy_train'].std()

        ax[0].plot(acc_train_mean_none.index, acc_train_mean_none.values, label='TRAIN - agg: None',
                   linestyle='dashed')
        ax[0].fill_between(acc_train_mean_none.index, acc_train_mean_none.values - acc_train_std_none.values,
                           acc_train_mean_none.values + acc_train_std_none.values, alpha=0.2)
        ax[0].plot(acc_test_mean_none.index, acc_test_mean_none.values, label='TEST - agg: None',
                   linestyle='solid', color=ax[0].get_lines()[-1].get_color())
        ax[0].fill_between(acc_test_mean_none.index, acc_test_mean_none.values - acc_test_std_none.values,
                           acc_test_mean_none.values + acc_test_std_none.values, alpha=0.2,
                           color=ax[0].get_lines()[-1].get_color())

        # generalization_gap = accuracy_test - accuracy_train
        generalization_gap_mean = group_none.groupby('epoch') \
            .apply(lambda x: (x['accuracy_train'] - x['accuracy_test']).mean())
        generalization_gap_std = group_none.groupby('epoch') \
            .apply(lambda x: (x['accuracy_train'] - x['accuracy_test']).std())

        ax[1].plot(generalization_gap_mean.index, generalization_gap_mean.values,
                   label='GAP - agg: None',
                   linestyle='solid')

        ax[1].fill_between(generalization_gap_mean.index,
                           generalization_gap_mean.values - generalization_gap_std.values,
                           generalization_gap_mean.values + generalization_gap_std.values, alpha=0.2)

        # add None aggregator for loss
        loss_test_mean_none = group_none.groupby('epoch')['loss_test'].mean()
        loss_test_std_none = group_none.groupby('epoch')['loss_test'].std()

        loss_train_mean_none = group_none.groupby('epoch')['loss_train'].mean()
        loss_train_std_none = group_none.groupby('epoch')['loss_train'].std()

        ax[2].plot(loss_train_mean_none.index, loss_train_mean_none.values, label='TRAIN - agg: None',
                   linestyle='dashed')
        ax[2].fill_between(loss_train_mean_none.index, loss_train_mean_none.values - loss_train_std_none.values,
                           loss_train_mean_none.values + loss_train_std_none.values, alpha=0.2)
        ax[2].plot(loss_test_mean_none.index, loss_test_mean_none.values, label='TEST - agg: None',
                   linestyle='solid', color=ax[0].get_lines()[-1].get_color())
        ax[2].fill_between(loss_test_mean_none.index, loss_test_mean_none.values - loss_test_std_none.values,
                           loss_test_mean_none.values + loss_test_std_none.values, alpha=0.2,
                           color=ax[0].get_lines()[-1].get_color())

        for nb_byz in group['nb_byz'].sort_values().unique():
            group_byz = group[group['nb_byz'] == nb_byz]

            # accuracy

            acc_test_mean = group_byz.groupby('epoch')['accuracy_test'].mean()
            acc_test_std = group_byz.groupby('epoch')['accuracy_test'].std()

            acc_train_mean = group_byz.groupby('epoch')['accuracy_train'].mean()
            acc_train_std = group_byz.groupby('epoch')['accuracy_train'].std()

            ax[0].plot(acc_train_mean.index, acc_train_mean.values, label=f'TRAIN - byz: {nb_byz}, agg: {aggregator}',
                       linestyle='dashed')
            ax[0].fill_between(acc_train_mean.index, acc_train_mean.values - acc_train_std.values,
                               acc_train_mean.values + acc_train_std.values, alpha=0.2)
            ax[0].plot(acc_test_mean.index, acc_test_mean.values, label=f'TEST - byz: {nb_byz}, agg: {aggregator}',
                       linestyle='solid', color=ax[0].get_lines()[-1].get_color())
            ax[0].fill_between(acc_test_mean.index, acc_test_mean.values - acc_test_std.values,
                               acc_test_mean.values + acc_test_std.values, alpha=0.2,
                               color=ax[0].get_lines()[-1].get_color())

            # accuracy_test - accuracy_train
            generalization_gap_mean = group_byz.groupby('epoch') \
                .apply(lambda x: (x['accuracy_train'] - x['accuracy_test']).mean())
            generalization_gap_std = group_byz.groupby('epoch') \
                .apply(lambda x: (x['accuracy_train'] - x['accuracy_test']).std())

            ax[1].plot(generalization_gap_mean.index, generalization_gap_mean.values,
                       label=f'GAP - byz: {nb_byz}, agg: {aggregator}',
                       linestyle='solid')

            ax[1].fill_between(generalization_gap_mean.index,
                               generalization_gap_mean.values - generalization_gap_std.values,
                               generalization_gap_mean.values + generalization_gap_std.values, alpha=0.2)

            # loss
            loss_test_mean = group_byz.groupby('epoch')['loss_test'].mean()
            loss_test_std = group_byz.groupby('epoch')['loss_test'].std()

            loss_train_mean = group_byz.groupby('epoch')['loss_train'].mean()
            loss_train_std = group_byz.groupby('epoch')['loss_train'].std()

            ax[2].plot(loss_train_mean.index, loss_train_mean.values, label=f'TRAIN - byz: {nb_byz}, agg: {aggregator}',
                       linestyle='dashed')
            ax[2].fill_between(loss_train_mean.index, loss_train_mean.values - loss_train_std.values,
                               loss_train_mean.values + loss_train_std.values, alpha=0.2)
            ax[2].plot(loss_test_mean.index, loss_test_mean.values, label=f'TEST - byz: {nb_byz}, agg: {aggregator}',
                       linestyle='solid', color=ax[0].get_lines()[-1].get_color())
            ax[2].fill_between(loss_test_mean.index, loss_test_mean.values - loss_test_std.values,
                               loss_test_mean.values + loss_test_std.values, alpha=0.2,
                               color=ax[0].get_lines()[-1].get_color())

        ax[0].set_ylim([0.6, 1.0])
        ax[1].set_ylim([-0.05, 0.3])
        ax[0].legend()
        ax[1].legend()
        ax[2].legend()

        ax[0].set_xlabel('Epoch')
        ax[0].set_ylabel('Accuracy')
        ax[1].set_xlabel('Epoch')
        ax[1].set_ylabel('Generalization gap')
        ax[2].set_xlabel('Epoch')
        ax[2].set_ylabel('Loss')

        ax[0].set_title(f'Accuracy evolution of {aggregator} with {nb_nodes} nodes, '
                        f'{nb_epochs} epochs and batch size {batch_size}')
        ax[1].set_title(f'Generalization gap evolution of {aggregator} with {nb_nodes} nodes, '
                        f'{nb_epochs} epochs and batch size {batch_size}')
        ax[2].set_title(f'Loss evolution of {aggregator} with {nb_nodes} nodes, '
                        f'{nb_epochs} epochs and batch size {batch_size}')

        filename = f'nodes_{nb_nodes}_epochs_{nb_epochs}_batch_{batch_size}_agg_' \
                   f'{aggregator}.png'

        # plt.show()
        print((Path(OUTPUT_FOLDER, filename)))
        plt.savefig(Path(OUTPUT_FOLDER, filename), dpi=300)
        plt.close()
        # plt.clf()


def main():
    plot()


if __name__ == '__main__':
    main()
