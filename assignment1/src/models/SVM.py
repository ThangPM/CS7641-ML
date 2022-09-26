import numpy as np

from .base import BaseModel
from sklearn.svm import SVC


class SupportVectorMachine(BaseModel):
    def __init__(self, kernel='rbf', C=1.0, gamma='auto', tol=1e-3, class_weight=None, verbose=False, **kwargs):
        super().__init__(verbose)
        self._learner = SVC(kernel=kernel, C=C, gamma=gamma, tol=tol, class_weight=class_weight, verbose=verbose, **kwargs)
        self.kernel = kernel

    def learner(self):
        return self._learner

    def get_name(self):
        return f"SupportVectorMachine_{self.kernel}"

    def get_tuning_hyperparameters(self):
        # iters = [-1, int((1e6 / samples) / .8) + 1]
        # tols = np.arange(1e-8, 1e-1, 0.01)
        # C_values = np.arange(0.001, 2.5, 0.25)
        # gamma_fracs = np.arange(1 / features, 2.1, 0.2)

        tols = np.arange(1e-8, 1e-1, 0.01)
        C_values = [0.001, 0.01, 0.1, 1, 10, 100]
        gamma_fracs = [0.0091, 0.001, 0.01, 0.1, 1]

        params = {f'{self.get_name()}__tol': tols,
                  f'{self.get_name()}__class_weight': ['balanced', None],
                  f'{self.get_name()}__C': C_values,
                  f'{self.get_name()}__gamma': gamma_fracs}

        return params

    def get_plotting_params(self):
        C_values = [0.001, 0.01, 0.1, 1, 10, 100]   # np.arange(0.001, 2.5, 0.1)

        timing_params = None
        # complexity_param = {'name': f'{self.get_name()}__C',
        #                     'display_name': 'Regularization parameter',
        #                     'values': C_values}

        complexity_params = [{'name': f'{self.get_name()}__C',
                             'display_name': 'Regularization parameter',
                             'values': C_values},

                             {'name': f'{self.get_name()}__tol',
                              'display_name': 'Tolerance for stopping criterion',
                              'values': C_values}]

        iteration_details = {
            'x_scale': 'log',
            'params': {f'{self.get_name()}__max_iter': [2 ** x for x in range(12)]},
        }

        return iteration_details, complexity_params, timing_params

