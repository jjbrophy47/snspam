"""
This class gridsearch model specifically designed for EGGS models.
"""
import numpy as np
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.model_selection import StratifiedKFold
from itertools import product
from EGGS.eggs import EGGS
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from copy import copy
from multiprocessing import Pool


class GridSearch:
    """
    Class to perform gridsearch for the EGGS model.
    """

    def __init__(self, estimator, param_grid, scoring='average_precision', cv=3, n_jobs=1, random_state=None):
        """
        Initialization of EGGS gridsearch model.

        Parameters
        ----------
        estimator : object
            Classifier assumed to be an EGGS model.
        param_grid : dict
            Parameter values to try during gridsearch.
        scoring : str (default='average_precision')
            Scoring method to use for model selection.
        cv : int (default=3)
            Number of cross-validation folds to use for model selection.
        n_jobs : int (default=1)
            Number of jobs to run in parallel.
        """
        self.estimator = estimator
        self.param_grid = param_grid
        self.scoring = scoring
        self.cv = cv
        self.n_jobs = n_jobs
        self.random_state = random_state

        if scoring == 'accuracy':
            self.scoring = accuracy_score
        elif scoring == 'roc_auc':
            self.scoring = roc_auc_score
        else:
            self.scoring = average_precision_score

    def fit(self, X, y, target_col):
        X, y = check_X_y(X, y)
        if y.dtype == np.float and not np.all(np.mod(y, 1) == 0):
            raise ValueError('Unknown label type: ')
        self.n_feats_ = X.shape[1]
        self.classes_ = np.unique(y)
        self.closed_params = self.estimator.get_params()

        # get list of dicts containing parameter fits
        params_list = self._params_list()
        kf = StratifiedKFold(self.cv, shuffle=True, random_state=self.random_state)
        print('[GridSearch]: total number of fits: %d' % (len(params_list) * self.cv))

        # fit each parameter setting for each fold
        all_scores = []
        n_fit = 0
        for fold, (train_index, test_index) in enumerate(kf.split(X, y)):
            X_train, y_train, target_col_train = X[train_index], y[train_index], target_col[train_index]
            X_test, y_test, target_col_test = X[test_index], y[test_index], target_col[test_index]

            # fit each parameter setting
            fold_scores = []

            if self.n_jobs > 1:
                self.async_results_ = []
                p = Pool(self.n_jobs)

                for i, params in enumerate(params_list):
                    n_fit += 1
                    args = (n_fit, params, X_train, y_train, target_col_train, X_test, y_test, target_col_test)
                    p.apply_async(self._fit_predict, args=args, callback=self._save_row)
                p.close()
                p.join()

                results = self.async_results_.copy()  # list of (score, setting) tuples
                results = sorted(results, key=lambda x: x[2])
                fold_scores = [score for score, params, fit_id in results]
                all_scores.append(fold_scores)

            else:

                for i, params in enumerate(params_list):
                    n_fit += 1

                    # create eggs model with fit parameters
                    print('[Gridsearch]: fit %d %s' % (n_fit, params))
                    model = EGGS(self.estimator).set_params(self.closed_params).set_params(params)

                    # train and score this model
                    fit_model = model.fit(X_train, y_train, target_col_train)
                    y_hat = fit_model.predict_proba(X_test, target_col_test)[:, 1]
                    fold_scores.append(self.scoring(y_test, y_hat))

                all_scores.append(fold_scores)

        mean_scores = np.array(all_scores).mean(axis=0)

        # compute average best score
        score_params = list(zip(mean_scores, params_list))
        self.best_score_, self.best_params_ = sorted(score_params, key=lambda x: x[0], reverse=True)[0]
        self.best_estimator_ = EGGS(self.estimator).set_params(self.closed_params).set_params(self.best_params_)
        self.fit_model_ = self.best_estimator_.fit(X, y, target_col)

        return self

    def predict_proba(self, X, target_col):
        X = check_array(X)
        if X.shape[1] != self.n_feats_:
            raise ValueError('X does not have the same number of features!')
        check_is_fitted(self, 'fit_model_')

        y_hat = self.fit_model_.predict_proba(X, target_col)
        return y_hat

    def predict(self, X, target_col):
        X = check_array(X)
        check_is_fitted(self, 'fit_model_')

        y_hat = self.fit_model_.predict_proba(X, target_col)
        return self.classes_[np.argmax(y_hat, axis=1)]

    # private
    def _fit_predict(self, n_fit, params, X_train, y_train, target_col_train, X_test, y_test, target_col_test):

        # create eggs model with fit parameters
        print('[Gridsearch]: fit %d %s' % (n_fit, params))
        model = EGGS(self.estimator).set_params(self.closed_params).set_params(params)

        # train and score this model
        fit_model = model.fit(X_train, y_train, target_col_train)
        y_hat = fit_model.predict_proba(X_test, target_col_test)[:, 1]
        score = self.scoring(y_test, y_hat)

        return (score, params, n_fit)

    def _params_list(self):
        """Build list of dicts containing parameters to use for model selection."""

        params_list = []
        keys, values = zip(*sorted(self.param_grid.items()))
        for v in product(*values):
            params_list.append(dict(zip(keys, v)))

        # prune redundant settings
        settings_to_remove = []
        new_settings = []

        # reduce settings where sgl_method=None and joint_model=None to one setting
        # reduce settings where sgl_method=None and joint_model=PGM and stacks=[1,2,...] to one setting
        relations_set = set()
        pgm_set = set()
        if 'sgl_method' in params_list[0] and 'joint_model' in params_list[0]:
            for i, setting in enumerate(params_list):

                if setting['sgl_method'] is None and setting['joint_model'] is None:
                        settings_to_remove.append(i)

                if setting['sgl_method'] is None and setting['joint_model'] is not None:
                    pgm_set.add(setting['joint_model'])

                    if 'relations' in setting:
                        relations_set.add('|'.join(setting['relations']))
                        settings_to_remove.append(i)

            new_setting = copy(params_list[-1])
            new_setting['sgl_method'] = None
            new_setting['joint_model'] = None
            new_settings.append(new_setting)

            for pgm in list(pgm_set):
                for relations in list(relations_set):
                    new_setting = copy(params_list[-1])
                    new_setting['sgl_method'] = None
                    new_setting['joint_model'] = pgm
                    new_setting['relations'] = relations.split('|')
                    new_settings.append(new_setting)

        params_list = [x for i, x in enumerate(params_list) if i not in settings_to_remove]
        params_list += new_settings

        return params_list

    def _save_row(self, result):
        if result is not None:
            self.async_results_.append(result)
