import numpy as np

from .base import BaseModel
from sklearn.tree import DecisionTreeClassifier


class DecisionTree(BaseModel):
    def __init__(self, criterion="gini", splitter="best", max_depth=None, class_weight=None, ccp_alpha=0.0, verbose=False, **kwargs):
        super().__init__(verbose)
        self._learner = DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=max_depth,
                                               class_weight=class_weight, ccp_alpha=ccp_alpha, **kwargs)

    def learner(self):
        return self._learner

    @property
    def classes_(self):
        return self._learner.classes_

    @property
    def n_classes_(self):
        return self._learner.n_classes_

    def get_name(self):
        return "DecisionTree"

    def fit(self, x, y, sample_weight=None, check_input=True):
        return self._learner.fit(x, y, sample_weight=sample_weight, check_input=check_input)

    def get_tuning_hyperparameters(self):
        max_depths = np.arange(1, 21, 1)

        params = {f'{self.get_name()}__criterion': ['gini', 'entropy'],
                  f'{self.get_name()}__class_weight': ['balanced', None],
                  f'{self.get_name()}__max_depth': max_depths,
                  f'{self.get_name()}__ccp_alpha': np.arange(0, 1, 0.1).tolist()}

        return params

    def get_plotting_params(self):
        iteration_details, timing_params = None, None
        max_depths = np.arange(1, 21, 1)

        # complexity_param = {'name': f'{self.get_name()}__max_depth',
        #                     'display_name': 'Max Depth',
        #                     'values': max_depths}

        # complexity_param = {'name': f'{self.get_name()}__ccp_alpha',
        #                     'display_name': 'ccp_alpha',
        #                     'values': np.arange(0, 1, 0.1).tolist()}

        complexity_params = [{'name': f'{self.get_name()}__ccp_alpha',
                              'display_name': 'ccp_alpha',
                              'values': np.arange(0, 1, 0.1).tolist()},

                             {'name': f'{self.get_name()}__max_depth',
                              'display_name': 'Max Depth',
                              'values': max_depths}]

        return iteration_details, complexity_params, timing_params
