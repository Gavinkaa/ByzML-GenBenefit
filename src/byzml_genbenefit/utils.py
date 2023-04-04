import torch
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score
import numpy as np


def compute_stats(test_data_loader: torch.utils.data.DataLoader, model: torch.nn.Module, loss_fn: callable) -> \
        dict[str, float]:
    """Computes the accuracy, loss, precision, recall and F1 score of the model on the test data set

    Args:
        test_data_loader (torch.utils.data.DataLoader): The test data loader
        model (torch.nn.Module): The model to test
        loss_fn (callable): The loss function to use for training

    Returns:
        dict[str, float]: A dictionary containing: ['accuracy', 'loss', 'precision', 'recall', 'f1_score']

    """

    scores_list = []
    true_labels_list = []

    model.eval()

    with torch.no_grad():  # reduce memory consumption
        for x, y in test_data_loader:
            x, y = x.to(model.device), y.to(model.device)
            scores = model(x)

            scores_list.append(scores.cpu())
            true_labels_list.append(y.cpu())

    scores = torch.cat(scores_list)
    predictions_list = torch.argmax(scores, dim=1)

    report = classification_report(torch.cat(true_labels_list), predictions_list, output_dict=True,
                                   labels=np.unique(predictions_list))
    accuracy = accuracy_score(torch.cat(true_labels_list), predictions_list)
    precision = report['weighted avg']['precision']
    recall = report['weighted avg']['recall']
    f1_score = report['weighted avg']['f1-score']
    loss = loss_fn(scores, torch.cat(true_labels_list))

    return {
        'accuracy': accuracy,
        'loss': loss,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }


def save_stats_to_csv(stats_train: dict[str, list], stats_test: dict[str, list], filename: str):
    """Saves the stats to a csv file

    Args:
        stats_train (dict[str, list]): The stats of the training data
        stats_test (dict[str, list]): The stats of the test data
        filename (str): The name of the file
    """
    with open(filename, 'w') as f:
        f.write('epoch,accuracy_train,accuracy_test,loss_train,loss_test,precision_train,precision_test,'
                'recall_train,recall_test,f1_score_train,f1_score_test\n')
        for i in range(len(stats_train['accuracy'])):
            f.write(f'{i + 1},{stats_train["accuracy"][i]},{stats_test["accuracy"][i]},'
                    f'{stats_train["loss"][i]},{stats_test["loss"][i]},'
                    f'{stats_train["precision"][i]},{stats_test["precision"][i]},'
                    f'{stats_train["recall"][i]},{stats_test["recall"][i]},'
                    f'{stats_train["f1_score"][i]},{stats_test["f1_score"][i]}\n')


def compute_accuracy(test_data_loader: torch.utils.data.DataLoader, model: torch.nn.Module) -> tuple[float, int, int]:
    """Computes the accuracy of the model on the test data set

    Args:
        test_data_loader (torch.utils.data.DataLoader): The test data loader
        model (torch.nn.Module): The model to test

    Returns:
        float: The accuracy of the model
        int: The number of correct predictions'
        int: The number of samples
    """
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():  # reduce memory consumption
        for x, y in test_data_loader:
            x, y = x.to(model.device), y.to(model.device)
            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    return float(num_correct) / float(num_samples), num_correct, num_samples


def plot_accuracies(accuracies_train: list[float], accuracies_test: list[float], title: str = 'Accuracy evolution',
                    accuracy_range: tuple[float, float] = None, save: bool = False):
    """Plots the accuracies of the training and test data

    Args:
        accuracies_train (list[float]): The accuracies of the training data
        accuracies_test (list[float]): The accuracies of the test data
        title (str): The title of the plot. Default: 'Accuracy evolution'
        accuracy_range (tuple[float, float]): The range of the y-axis. Default: None
        save (bool): Whether to save the plot. Default: False
    """

    plt.plot(accuracies_train, label='train')
    plt.plot(accuracies_test, label='test')
    plt.legend()
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    if accuracy_range is not None:
        plt.ylim(accuracy_range)

    if save:
        fig_name = title.replace('\n', ' ').replace(' ', '_') + '.png'
        plt.savefig(fig_name)
    else:
        plt.show()


def save_accuracies_to_csv(accuracies_train: list[float], accuracies_test: list[float], filename: str):
    """Saves the accuracies of the training and test data to a csv file

    Args:
        accuracies_train (list[float]): The accuracies of the training data
        accuracies_test (list[float]): The accuracies of the test data
        filename (str): The name of the file
    """
    with open(filename, 'w') as f:
        f.write('epoch,train,test\n')
        for i in range(len(accuracies_train)):
            f.write(f'{i + 1},{accuracies_train[i]},{accuracies_test[i]}\n')
