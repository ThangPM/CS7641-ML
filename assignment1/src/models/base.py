import logging

from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BaseModel(ABC, BaseEstimator):
    """
    Base model for classification-based learning
    """
    def __init__(self, verbose):
        self._verbose = verbose

    @property
    @abstractmethod
    def learner(self):
        pass

    def get_name(self):
        pass

    def get_params(self, deep=True):
        return self.learner().get_params(deep)

    def set_params(self, **params):
        return self.learner().set_params(**params)

    def fit(self, training_data, classes):
        if self.learner() is None:
            return None

        return self.learner().fit(training_data, classes)

    def predict(self, data):
        if self.learner() is None:
            return None

        return self.learner().predict(data)

    def get_tuning_hyperparameters(self):
        pass

    def get_plotting_params(self):
        pass

    def log(self, msg, *args):
        if self._verbose:
            logger.info(msg.format(*args))

