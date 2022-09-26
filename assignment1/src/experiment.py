import datetime
import os
from os.path import exists

from collections import defaultdict
import time

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, validation_curve, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from metrics import *
from plotting import *


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

OUTPUT_DIRECTORY = '../results'

if not os.path.exists(OUTPUT_DIRECTORY):
    os.makedirs(OUTPUT_DIRECTORY)
if not os.path.exists('{}/images'.format(OUTPUT_DIRECTORY)):
    os.makedirs('{}/images'.format(OUTPUT_DIRECTORY))


class Experiment(object):
    def __init__(self, model, dataset, dataset_name, threads=-1, seed=42, verbose=False):
        self.model = model
        self.model_name = model.get_name()
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.threads = threads
        self.seed = seed
        self.verbose = verbose

        dataset_splits = train_test_split(dataset.features, dataset.classes, test_size=0.2, random_state=self.seed, shuffle=True, stratify=dataset.classes)
        self.training_x, self.test_x, self.training_y, self.test_y = dataset_splits

        self.params = self.model.get_tuning_hyperparameters()

        best_params_file = '{}/{}_{}_best_params.csv'.format(OUTPUT_DIRECTORY, self.model_name, self.dataset_name)
        self.best_params = pd.read_csv(best_params_file).to_dict() if exists(best_params_file) else None

        self.iteration_details, self.complexity_params, self.timing_params = self.model.get_plotting_params()
        # self.iteration_details, self.complexity_params, self.timing_params = None, None, None

    def tune_best_model(self, pipeline, params, best_params=None, balanced_dataset=False):

        logger.info("Tuning best results for {} ({} thread(s))".format(self.model_name, self.threads))

        if self.model_name is None or self.dataset_name is None:
            raise Exception('clf_type and dataset are required')

        if self.seed is not None:
            np.random.seed(self.seed)

        curr_scorer = scorer
        if not balanced_dataset:
            curr_scorer = f1_scorer

        if best_params:
            pipeline.fit(self.training_x, self.training_y)
            pred_y = pipeline.predict(self.test_x)
            test_score = balanced_accuracy(self.test_y, pred_y)
            cv = pipeline
        else:
            cv = GridSearchCV(pipeline, n_jobs=self.threads, param_grid=params, refit=True, verbose=10, cv=5, scoring=curr_scorer)
            cv.fit(self.training_x, self.training_y)

            reg_table = pd.DataFrame(cv.cv_results_)
            reg_table.to_csv('{}/{}_{}_reg.csv'.format(OUTPUT_DIRECTORY, self.model_name, self.dataset_name), index=False)
            test_score = cv.score(self.test_x, self.test_y)

            final_estimator = cv.best_estimator_._final_estimator
            best_params = final_estimator.get_params()
            grid_best_params = pd.DataFrame([best_params])
            grid_best_params.to_csv('{}/{}_{}_best_params.csv'.format(OUTPUT_DIRECTORY, self.model_name, self.dataset_name), index=False)
            logger.info(" - Grid search complete")

            with open('{}/all_test_results.csv'.format(OUTPUT_DIRECTORY), 'a') as f:
                ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
                f.write('"{}",{},{},{},"{}"\n'.format(ts, self.model_name, self.dataset_name, test_score, best_params))

        # ----------------------------------------------------------------------------------------------------------------
        # Draw Learning Curve here
        # ----------------------------------------------------------------------------------------------------------------

        n = self.training_y.shape[0]

        train_size_fracs = np.linspace(0, 1.0, 21, endpoint=True)[1:]    # 0 should not be included
        logger.info(" - n: {}, train_sizes: {}".format(n, train_size_fracs))
        
        train_sizes, train_scores, test_scores = learning_curve(
            cv if best_params is not None else cv.best_estimator_, self.training_x, self.training_y,
            cv=5, train_sizes=train_size_fracs, scoring=curr_scorer, verbose=10, n_jobs=self.threads, random_state=self.seed)

        logger.info(" - n: {}, train_sizes: {}".format(n, train_sizes))
        curve_train_scores = pd.DataFrame(index=train_sizes, data=train_scores)
        curve_test_scores = pd.DataFrame(index=train_sizes, data=test_scores)
        curve_train_scores.to_csv('{}/{}_{}_LC_train.csv'.format(OUTPUT_DIRECTORY, self.model_name, self.dataset_name))
        curve_test_scores.to_csv('{}/{}_{}_LC_test.csv'.format(OUTPUT_DIRECTORY, self.model_name, self.dataset_name))

        plt = plot_learning_curve('Learning Curve: {} - {}'.format(self.model_name, self.dataset_name), [frac*100 for frac in train_size_fracs], train_scores, test_scores)
        plt.savefig('{}/images/{}_{}_LC.png'.format(OUTPUT_DIRECTORY, self.model_name, self.dataset_name), format='png', dpi=150)
        logger.info(" - Learning curve complete")

        # ----------------------------------------------------------------------------------------------------------------

        return cv

    def perform_experiment(self, iteration_lc_only=False):

        logger.info("Experimenting on {} with classifier {}.".format(self.dataset_name, self.model_name))

        pipeline = Pipeline([('Scale', StandardScaler()), (self.model_name, self.model)])
        final_params = None

        if not iteration_lc_only:
            ds_clf = self.tune_best_model(pipeline, self.params, self.best_params, balanced_dataset=self.dataset.balanced)

            if self.best_params is not None:
                final_params = self.best_params
            else:
                final_params = ds_clf.best_params_
                pipeline.set_params(**final_params)

            if self.verbose:
                logger.info("final_params: {}".format(final_params))

            if self.complexity_params is not None:
                for complexity_param in self.complexity_params:
                    param_display_name = complexity_param['name']
                    x_scale = 'linear'
                    if 'display_name' in complexity_param:
                        param_display_name = complexity_param['display_name']
                    if 'x_scale' in complexity_param:
                        x_scale = complexity_param['x_scale']

                    self.make_complexity_curve(complexity_param['name'], param_display_name, complexity_param['values'],
                                               ds_clf.best_estimator_ if self.best_params is None else ds_clf,
                                               x_scale, balanced_dataset=self.dataset.balanced)

            if self.timing_params is not None:
                pipeline.set_params(**self.timing_params)

            self.make_timing_curve(ds_clf.best_estimator_ if self.best_params is None else ds_clf)

        if self.iteration_details is not None:
            x_scale = 'linear'
            if 'pipe_params' in self.iteration_details:
                pipeline.set_params(**self.iteration_details['pipe_params'])
            if 'x_scale' in self.iteration_details:
                x_scale = self.iteration_details['x_scale']

            self.iteration_lc(pipeline, self.iteration_details['params'], balanced_dataset=self.dataset.balanced, x_scale=x_scale)

        # Return the best params found, if we have any
        return final_params

    def iteration_lc(self, clf, params, balanced_dataset=False, x_scale='linear'):
        logger.info("Building iteration learning curve for params {} ({} threads)".format(params, self.threads))

        if self.model_name is None or self.dataset_name is None:
            raise Exception('model_name and dataset_name are required')
        if self.seed is not None:
            np.random.seed(self.seed)

        curr_scorer = scorer
        acc_method = balanced_accuracy
        if not balanced_dataset:
            curr_scorer = f1_scorer
            acc_method = f1_accuracy

        cv = GridSearchCV(clf, n_jobs=self.threads, param_grid=params, refit=True, verbose=10, cv=5, scoring=curr_scorer)
        cv.fit(self.training_x, self.training_y)
        reg_table = pd.DataFrame(cv.cv_results_)
        reg_table.to_csv('{}/ITER_base_{}_{}.csv'.format(OUTPUT_DIRECTORY, self.model_name, self.dataset_name), index=False)
        d = defaultdict(list)
        name = list(params.keys())[0]

        for value in list(params.values())[0]:
            d['param_{}'.format(name)].append(value)
            clf.set_params(**{name: value})
            clf.fit(self.training_x, self.training_y)
            pred = clf.predict(self.training_x)
            d['train acc'].append(acc_method(self.training_y, pred))
            clf.fit(self.training_x, self.training_y)
            pred = clf.predict(self.test_x)
            d['test acc'].append(acc_method(self.test_y, pred))
            logger.info(' - {}'.format(value))
        d = pd.DataFrame(d)
        d.to_csv('{}/ITERtestSET_{}_{}.csv'.format(OUTPUT_DIRECTORY, self.model_name, self.dataset_name), index=False)

        plt = plot_learning_curve('{} - {} ({})'.format(self.model_name, self.dataset_name, name), d['param_{}'.format(name)],
                                  d['train acc'], d['test acc'], multiple_runs=False, x_scale=x_scale, x_label='Value')
        plt.savefig('{}/images/{}_{}_ITER_LC.png'.format(OUTPUT_DIRECTORY, self.model_name, self.dataset_name), format='png', dpi=150)
        logger.info(" - Iteration learning curve complete")

        return cv

    def make_timing_curve(self, clf):
        logger.info("Building timing curve")

        out_file = '{}/{}_{}_timing.csv'.format(OUTPUT_DIRECTORY, self.model_name, self.dataset_name)
        sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        tests = 5

        if not exists(out_file):
            out = dict()
            out['train'] = np.zeros(shape=(len(sizes), tests))
            out['test'] = np.zeros(shape=(len(sizes), tests))
            for i, frac in enumerate(sizes):
                for j in range(tests):
                    np.random.seed(self.seed)
                    x_train, x_test, y_train, y_test = train_test_split(self.dataset.features, self.dataset.classes, test_size=1-frac, random_state=self.seed)

                    st = time.time()
                    clf.fit(x_train, y_train)
                    out['train'][i, j] = (time.time() - st)

                    st = time.time()
                    clf.predict(x_test)
                    out['test'][i, j] = (time.time() - st)

                    logger.info(" - {} {} {}".format(self.model_name, self.dataset_name, frac))

            train_df = pd.DataFrame(out['train'], index=sizes)
            test_df = pd.DataFrame(out['test'], index=sizes)
            mean_std_axis = 1

            out = pd.DataFrame(index=sizes)
            out['train'] = np.mean(train_df, axis=1)
            out['test'] = np.mean(test_df, axis=1)
            out.to_csv()
        else:
            out = pd.read_csv(out_file).to_dict()
            train_df = pd.DataFrame(out['train'], index=sizes)
            test_df = pd.DataFrame(out['test'], index=sizes)
            mean_std_axis = 0

        plt = plot_model_timing('{} - {}'.format(self.model_name, self.dataset_name), np.array(sizes) * 100, train_df, test_df, mean_std_axis=mean_std_axis)
        plt.savefig('{}/images/{}_{}_TC.png'.format(OUTPUT_DIRECTORY, self.model_name, self.dataset_name), format='png', dpi=150)

        logger.info(" - Timing curve complete")

    def make_complexity_curve(self, param_name, param_display_name, param_values, clf, x_scale, balanced_dataset=False):
        logger.info("Building model complexity curve")
        curr_scorer = scorer
        if not balanced_dataset:
            curr_scorer = f1_scorer

        train_scores, test_scores = validation_curve(clf, self.training_x, self.training_y, param_name=param_name, param_range=param_values,
                                                     cv=5, verbose=self.verbose, scoring=curr_scorer, n_jobs=self.threads)

        curve_train_scores = pd.DataFrame(index=param_values, data=train_scores)
        curve_test_scores = pd.DataFrame(index=param_values, data=test_scores)

        curve_train_scores.to_csv('{}/{}_{}_{}_MC_train.csv'.format(OUTPUT_DIRECTORY, self.model_name, self.dataset_name, param_name))
        curve_test_scores.to_csv('{}/{}_{}_{}_MC_test.csv'.format(OUTPUT_DIRECTORY, self.model_name, self.dataset_name, param_name))

        # For plotting only
        if isinstance(param_values[0], tuple):
            param_values = [str(x) for x in param_values]

        plt = plot_model_complexity_curve('Model Complexity: {} - {} ({})'.format(self.model_name, self.dataset_name, param_display_name),
                                          param_values, train_scores, test_scores, x_scale=x_scale, x_label=param_display_name)

        plt.savefig('{}/images/{}_{}_{}_MC.png'.format(OUTPUT_DIRECTORY, self.model_name, self.dataset_name, param_name), format='png', dpi=150)
        logger.info(" - Model complexity curve complete")



