import numpy as np

from .base import BaseModel
from .decision_tree import DecisionTree
from sklearn.ensemble import AdaBoostClassifier


class AdaBoost(BaseModel):
    def __init__(self, n_estimators=50, learning_rate=1., algorithm="SAMME", base_estimator=None, verbose=False, **kwargs):
        super().__init__(verbose)
        self.base_estimator = DecisionTree(criterion='entropy', class_weight='balanced') if base_estimator is None else base_estimator
        self._learner = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate,
                                           algorithm=algorithm, base_estimator=self.base_estimator, **kwargs)

    def learner(self):
        return self._learner

    def get_name(self):
        return "AdaBoost"

    def get_tuning_hyperparameters(self):
        max_depths = np.arange(1, 11, 1)

        params = {f'{self.get_name()}__n_estimators': [5, 10, 20, 40, 60, 80, 100],
                  f'{self.get_name()}__learning_rate': [(2 ** x) / 100 for x in range(7)] + [1],
                  f'{self.get_name()}__base_estimator__max_depth': max_depths
                  }

        return params

    def get_plotting_params(self):
        # For Iter Curve
        iteration_details = {'params': {f'{self.get_name()}__n_estimators': [5, 10, 20, 40, 60, 80, 100]}}

        # For Model Curve
        # complexity_param = {'name': f'{self.get_name()}__learning_rate',
        #                     'display_name': 'Learning rate',
        #                     'x_scale': 'log',
        #                     'values': [(2 ** x) / 100 for x in range(7)] + [1]}

        # complexity_param = {'name': f'{self.get_name()}__n_estimators',
        #                     'display_name': 'Number of weak learners',
        #                     'x_scale': 'log',
        #                     'values': [5, 10, 20, 40, 60, 80, 100]}

        complexity_params = [{'name': f'{self.get_name()}__learning_rate',
                              'display_name': 'Learning rate',
                              'x_scale': 'log',
                              'values': [(2 ** x) / 100 for x in range(7)] + [1]},

                             {'name': f'{self.get_name()}__n_estimators',
                              'display_name': 'Number of weak learners',
                              'x_scale': 'log',
                              'values': [5, 10, 20, 40, 60, 80, 100]}]

        # For Time Curve
        timing_params = None

        return iteration_details, complexity_params, timing_params
