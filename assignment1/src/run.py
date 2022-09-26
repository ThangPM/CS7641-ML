import argparse
from datetime import datetime
import logging
import numpy as np

from data_loader import WiscosinBreastCancer, CreditCardFraud
from experiment import Experiment

from models import DecisionTree, AdaBoost, kNearestNeighbor, SupportVectorMachine, NeuralNetwork


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_experiment(model, dataset, dataset_name, timings, verbose, threads, seed):
    t = datetime.now()

    experiment = Experiment(model=model, dataset=dataset, dataset_name=dataset_name, threads=threads, seed=seed, verbose=verbose)
    experiment.perform_experiment()

    logger.info("Running {} experiment: {}".format(model.get_name(), dataset_name))
    t_d = datetime.now() - t
    timings[model.get_name()] = t_d.seconds


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform some SL experiments')
    parser.add_argument('--threads', type=int, default=1, help='Number of threads (defaults to 1, -1 for auto)')
    parser.add_argument('--seed', type=int, help='A random seed to set, if desired')
    parser.add_argument('--nn', action='store_true', help='Run the Neural Network experiment')
    parser.add_argument('--adaboost', action='store_true', help='Run the Adaboost experiment')
    parser.add_argument('--dt', action='store_true', help='Run the Decision Tree experiment')
    parser.add_argument('--knn', action='store_true', help='Run the k-Nearest Neighbor experiment')
    parser.add_argument('--svm', action='store_true', help='Run the SVM experiment')
    parser.add_argument('--all', action='store_true', help='Run all experiments')
    parser.add_argument('--verbose', action='store_true', help='If true, provide verbose output')
    args = parser.parse_args()

    verbose = args.verbose
    threads = args.threads

    seed = args.seed
    if seed is None:
        seed = np.random.randint(0, (2 ** 32) - 1, dtype='uint64')
        print("Using seed {}".format(seed))

    print("Loading data...")

    dataset1 = {'data': WiscosinBreastCancer(verbose=verbose, seed=seed), 'name': 'WiscosinBreastCancer'}
    dataset2 = {'data': CreditCardFraud(verbose=verbose, seed=seed), 'name': 'CreditCardFraud'}

    print("Running experiments...")

    timings = {}
    datasets = [dataset1, dataset2]
    experiment_details = []

    for ds in datasets:
        data = ds['data']
        data.load_and_process()
        data.build_train_test_split()
        data.scale_standard()

        models = []
        if args.dt or args.all:
            models.append(DecisionTree())
        if args.adaboost or args.all:
            models.append(AdaBoost())
        if args.knn or args.all:
            models.append(kNearestNeighbor())
        if args.svm or args.all:
            models.extend([SupportVectorMachine(kernel="linear"), SupportVectorMachine(kernel="rbf")])
        # if args.nn or args.all or args.all:
        #     models.append(NeuralNetwork())

        for model in models:
            run_experiment(model, dataset=data, dataset_name=ds['name'], timings=timings, verbose=verbose, threads=threads, seed=seed)

    print(timings)

