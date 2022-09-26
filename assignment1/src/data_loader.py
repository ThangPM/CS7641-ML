import os
import copy
import logging
import pandas as pd
import numpy as np

from collections import Counter
from sklearn.model_selection import train_test_split
from scipy.sparse import isspmatrix
from sklearn.preprocessing import StandardScaler

from abc import ABC, abstractmethod


OUTPUT_DIRECTORY = '../results'

if not os.path.exists(OUTPUT_DIRECTORY):
    os.makedirs(OUTPUT_DIRECTORY)
if not os.path.exists('{}/images'.format(OUTPUT_DIRECTORY)):
    os.makedirs('{}/images'.format(OUTPUT_DIRECTORY))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Adapted from https://stats.stackexchange.com/questions/239973/a-general-measure-of-data-set-imbalance
def is_balanced(seq):
    n = len(seq)
    classes = [(clas, float(count)) for clas, count in Counter(seq).items()]
    k = len(classes)

    H = -sum([(count/n) * np.log((count/n)) for clas, count in classes])
    return H/np.log(k) > 0.75


class DataLoader(ABC):
    def __init__(self, path, verbose, seed):
        self._path = path
        self._verbose = verbose
        self._seed = seed

        self.features = None
        self.classes = None

        self.training_x = None
        self.training_y = None

        self.testing_x = None
        self.testing_y = None

        self.binary = False
        self.balanced = False
        self._data = pd.DataFrame()

    def load_and_process(self, data=None, preprocess=True):
        """
        Load data from the given path and perform any initial processing required. This will populate the
        features and classes and should be called before any processing is done.

        :return: Nothing
        """
        if data is not None:
            self._data = data
        else:
            self._load_data()
        self.log("Processing {} Path: {}, Dimensions: {}", self.data_name(), self._path, self._data.shape)

        if self._verbose:
            old_max_rows = pd.options.display.max_rows
            pd.options.display.max_rows = 10
            self.log("Data Sample:\n{}", self._data)
            pd.options.display.max_rows = old_max_rows

        if preprocess:
            self.log("Will pre-process data")
            self._preprocess_data()

        self.get_features()
        self.get_classes()

        class_dist = np.histogram(self.classes)[0]
        class_dist = class_dist[np.nonzero(class_dist)]

        self.binary = True if len(class_dist) == 2 else False
        self.balanced = is_balanced(self.classes)

        if self._verbose:
            self.log("Feature dimensions: {}", self.features.shape)
            self.log("Classes dimensions: {}", self.classes.shape)
            self.log("Class values: {}", np.unique(self.classes))
            self.log("Class distribution: {}", class_dist)
            self.log("Class distribution (%): {}", (class_dist / self.classes.shape[0]) * 100)
            self.log("Sparse? {}", isspmatrix(self.features))
            self.log("Binary? {}", self.binary)
            self.log("Balanced? {}", self.balanced)

    def scale_standard(self):
        self.features = StandardScaler().fit_transform(self.features)
        if self.training_x is not None:
            self.training_x = StandardScaler().fit_transform(self.training_x)

        if self.testing_x is not None:
            self.testing_x = StandardScaler().fit_transform(self.testing_x)

    def build_train_test_split(self, test_size=0.3):
        if not self.training_x and not self.training_y and not self.testing_x and not self.testing_y:
            self.training_x, self.testing_x, self.training_y, self.testing_y = train_test_split(self.features, self.classes, test_size=test_size, random_state=self._seed, stratify=self.classes)

    def get_features(self, force=False):
        if self.features is None or force:
            self.log("Pulling features")
            self.features = np.array(self._data.iloc[:, 0:-1])

        return self.features

    def get_classes(self, force=False):
        if self.classes is None or force:
            self.log("Pulling classes")
            self.classes = np.array(self._data.iloc[:, -1])

        return self.classes

    @abstractmethod
    def _load_data(self):
        pass

    @abstractmethod
    def _preprocess_data(self):
        pass

    @abstractmethod
    def data_name(self):
        pass

    def reload_from_hdf(self, hdf_path, hdf_ds_name, preprocess=True):
        self.log("Reloading from HDF {}".format(hdf_path))
        loader = copy.deepcopy(self)

        df = pd.read_hdf(hdf_path, hdf_ds_name)
        loader.load_and_process(data=df, preprocess=preprocess)
        loader.build_train_test_split()

        return loader

    def log(self, msg, *args):
        """
        If the learner has verbose set to true, log the message with the given parameters using string.format
        :param msg: The log message
        :param args: The arguments
        :return: None
        """
        if self._verbose:
            logger.info(msg.format(*args))


class WiscosinBreastCancer(DataLoader):
    def __init__(self, path='../data/breast_cancer_wisconsin_wdbc.csv', verbose=False, seed=1):
        super().__init__(path, verbose, seed)

    def _load_data(self):
        self._data = pd.read_csv(self._path, header=0).fillna(0)

    def _preprocess_data(self):
        pass

    def data_name(self):
        return 'WisconsinBreastCancer'


class CreditCardFraud(DataLoader):
    def __init__(self, path='../data/creditcard.csv', verbose=False, seed=1):
        super().__init__(path, verbose, seed)

    def _load_data(self):
        self._data = pd.read_csv(self._path, header=0).fillna(0)
        self._data_fraud = self._data[self._data.iloc[:, -1] == 1.0]
        self._data_not_fraud = self._data[self._data.iloc[:, -1] == 0.0].sample(n=len(self._data_fraud))
        self._data = pd.concat([self._data_fraud, self._data_not_fraud]).sample(frac=1)

    def _preprocess_data(self):
        pass

    def data_name(self):
        return 'CreditCardFraud'


if __name__ == '__main__':
    wbc_data = WiscosinBreastCancer(verbose=True)
    wbc_data.load_and_process()

    ccf_data = CreditCardFraud(verbose=True)
    ccf_data.load_and_process()
