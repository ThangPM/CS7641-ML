import numpy as np

from .base import BaseModel
from sklearn.neural_network import MLPClassifier


class NeuralNetwork(BaseModel):
    def __init__(self, hidden_layer_sizes=(100,),
                 activation="relu",
                 solver='adam',
                 alpha=0.0001,
                 batch_size='auto',
                 learning_rate="constant",
                 max_iter=1000,
                 verbose=False, **kwargs):

        super().__init__(verbose)
        self._learner = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,
                                      activation=activation,
                                      solver=solver,
                                      alpha=alpha,
                                      batch_size=batch_size,
                                      learning_rate=learning_rate,
                                      max_iter=max_iter,
                                      **kwargs)

    def learner(self):
        return self._learner

    def get_name(self):
        return "NeuralNetwork"

    def get_tuning_hyperparameters(self):
        alphas = [0.0001, 0.001, 0.01, 0.1, 0.5, 1]
        optimizers = ['sgd', 'adam']
        activations = ['tanh', 'relu', 'logistic']
        learning_rates = ['invscaling', 'adaptive']     # No need for 'constant'
        hidden_layer_sizes = [(h,) * l for l in np.arange(1, 4, 1) for h in [5, 10, 20]]

        params = {f'{self.get_name()}__alpha': alphas,
                  f'{self.get_name()}__solver': optimizers,
                  # f'{self.get_name()}__activation': activations,
                  # f'{self.get_name()}__learning_rate': learning_rates,
                  f'{self.get_name()}__hidden_layer_sizes': hidden_layer_sizes}

        return params

    def get_plotting_params(self):
        alphas = [0.0001, 0.001, 0.01, 0.1, 0.5, 1]
        hidden_layer_sizes = [(h,) * l for l in np.arange(1, 4, 1) for h in [5, 10, 20]]

        # complexity_param = {'name': f'{self.get_name()}__alpha',
        #                     'display_name': 'Alpha',
        #                     'x_scale': 'log',
        #                     'values': alphas}

        # complexity_param = {'name': f'{self.get_name()}__hidden_layer_sizes',
        #                     'display_name': 'Hidden layer sizes',
        #                     'x_scale': 'log',
        #                     'values': hidden_layer_sizes}

        complexity_params = [{'name': f'{self.get_name()}__hidden_layer_sizes',
                              'display_name': 'Hidden layer sizes',
                              'x_scale': 'log',
                              'values': hidden_layer_sizes},

                             {'name': f'{self.get_name()}__alpha',
                              'display_name': 'Alpha',
                              'x_scale': 'log',
                              'values': alphas}]

        timing_params = {f'{self.get_name()}__early_stopping': False}
        iteration_details = {
            'x_scale': 'log',
            'params': {f'{self.get_name()}__max_iter': [2 ** x for x in range(12)]},
            'pipe_params': timing_params
        }

        return iteration_details, complexity_params, timing_params
