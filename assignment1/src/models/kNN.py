import numpy as np

from .base import BaseModel
from sklearn.neighbors import KNeighborsClassifier


class kNearestNeighbor(BaseModel):
    def __init__(self, n_neighbors=5, weights='uniform', metric='minkowski', verbose=False, **kwargs):
        super().__init__(verbose)
        self._learner = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, metric=metric, **kwargs)

    def learner(self):
        return self._learner

    def get_name(self):
        return "kNearestNeighbor"

    def get_tuning_hyperparameters(self):
        params = {f'{self.get_name()}__metric': ['manhattan', 'euclidean', 'minkowski'],
                  f'{self.get_name()}__n_neighbors': np.arange(1, 51, 5),
                  f'{self.get_name()}__weights': ['uniform', 'distance']}

        return params

    def get_plotting_params(self):
        iteration_details, timing_params = None, None

        # complexity_param = {'name': f'{self.get_name()}__n_neighbors',
        #                     'display_name': 'Neighbor count',
        #                     'values': np.arange(1, 51, 5)}

        complexity_params = [{'name': f'{self.get_name()}__n_neighbors',
                              'display_name': 'Neighbor count',
                              'values': np.arange(1, 51, 5)},

                             {'name': f'{self.get_name()}__metric',
                              'display_name': 'Metric',
                              'values': ['manhattan', 'euclidean', 'minkowski']}]

        return iteration_details, complexity_params, timing_params
