# Generalization Benefits of Byzantine ML

This repository is the implementation of my Semester Project at EPFL.
The report of this project can be requested if you are interested.

## Abstract
Distributed machine learning algorithms face challenges in ensuring integrity and robustness against malicious agents. Byzantine-robust machine learning algorithms aim to address these challenges while the impact on generalization remains unexplored. This project investigates the generalization benefits by comparing Byzantine-robust ML algorithms to non-robust counterparts. The experiments employ different robust aggregation methods and compare them to a standard arithmetic mean aggregator using popular datasets, MNIST and CIFAR-10. The results reveal variations in accuracy, convergence speed, and generalization gap across aggregation methods. Notably, the Krum method consistently outperforms the mean aggregator, showcasing improvements in accuracy and generalization. The incorporation of the pre-aggregation technique, Nearest Neighbor Mixing (NNM), demonstrates potential to enhance accuracy and improve generalization for certain aggregation methods. Overall, this paper sheds light on the potential generalization benefits of Byzantine-robust ML algorithms, providing insights into their performance in distributed learning scenarios.

## Installation and Usage

1. Install [Poetry](https://python-poetry.org/docs/).
2. Run `poetry install` to install dependencies.
3. Add the environment as interpreter in your IDE.
