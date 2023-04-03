import matplotlib.pyplot as plt
import csv
import os

# ------ SETTINGS ------
SETTINGS_ACTIVE = False
SELECTED_AGGREGATORS = ['None', 'Krum']
SELECTED_NB_NODES = ['5']
SELECTED_NB_EPOCHS = '200'


# ----------------------


def main():
    plt.rcParams['figure.figsize'] = (10, 8)

    for file in os.listdir('../results/MNIST'):
        if file.endswith('.csv'):
            labels = file.split('_')
            nb_nodes = labels[1]
            nb_byz = labels[3]
            # batch_size = labels[5]
            nb_epochs = labels[7]
            agg = labels[9].split('Aggregator')[0]
            # lr = labels[11]
            # seed = labels[13].split('.csv')[0]

            if nb_epochs != SELECTED_NB_EPOCHS or nb_nodes not in SELECTED_NB_NODES or agg not in SELECTED_AGGREGATORS:
                continue

            accuracies_train = []
            accuracies_test = []
            with open('../results/MNIST/' + file, 'r') as csvfile:
                plots = csv.reader(csvfile, delimiter=',')
                # skip header
                next(plots)
                for row in plots:
                    accuracies_train.append(float(row[1]))
                    accuracies_test.append(float(row[2]))

            # filename = f'./results/nodes_{nb_of_nodes}_byz_{nb_of_byzantine_nodes}_batch_' \
            #                f'{batch_size}_epochs_{nb_epochs}_agg_{aggregate_fn}.csv'

            # plt.plot(accuracies_test, label=f'nodes: {nb_nodes}, byz: {nb_byz}, agg: {agg}')

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

            plt.plot(accuracies_train_ma, label=f'TRAIN nodes: {nb_nodes}, byz: {nb_byz}, agg: {agg}',
                     linestyle='dashed')

            color = plt.gca().lines[-1].get_color()

            plt.plot(accuracies_test_ma, label=f'TEST nodes: {nb_nodes}, byz: {nb_byz}, agg: {agg}', linestyle='solid',
                     color=color)

    # check if there is at least one plot
    if len(plt.gca().lines) == 0:
        return

    plt.ylim(0.95, 1)
    plt.legend()
    plt.title(f'Accuracy evolution of {[x for x in SELECTED_AGGREGATORS if x != "None"][0]}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    # plot in big
    # plt.show()
    print(
        f'nodes_{nb_nodes}_epochs_{SELECTED_NB_EPOCHS}_agg_'
        f'{[agg for agg in SELECTED_AGGREGATORS if agg != "None"][0]}.png')
    plt.savefig(
        f'nodes_{nb_nodes}_epochs_{SELECTED_NB_EPOCHS}_agg_'
        f'{[agg for agg in SELECTED_AGGREGATORS if agg != "None"][0]}.png')
    plt.show()


if __name__ == '__main__':
    if SETTINGS_ACTIVE is not True:
        aggregators = ['Krum', 'CWMed', 'GM', 'CWTM']

        for nodes in ['5', '11']:
            SELECTED_NB_NODES = nodes
            for epoch in ['300', '600', '1500']:
                SELECTED_NB_EPOCHS = str(epoch)
                for agg in aggregators:
                    if epoch == '1500' and 'Krum' not in agg:
                        continue
                    SELECTED_AGGREGATORS = [agg, 'None']
                    main()

    else:
        main()
