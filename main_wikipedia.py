"""
This script uses EGGS to model a wikipedia spam dataset.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.model_selection import KFold

from EGGS.eggs import EGGS
from xgboost import XGBClassifier
from data.wikipedia.features.relational import pseudo_relational as pr
from data.wikipedia.features.relational import pgm

from collections import defaultdict
from EGGS import print_utils
from EGGS.gridsearch import GridSearch


def main(random_state=None):

    # settings
    data_fname = 'data/wikipedia/features/independent/data.csv'
    folds = 10
    invert = False
    spammer_percentage = 0.5

    data_df = pd.read_csv(data_fname)
    data_df = data_df.fillna(0)

    # create unbalanced data with mostly benign users
    if spammer_percentage < 0.5:
        benign_df = data_df[data_df['label'] == 0]
        spammer_df = data_df[data_df['label'] == 1].sample(frac=spammer_percentage, random_state=random_state)
        data_df = pd.concat([benign_df, spammer_df])

    # shuffle dataset
    data_df = data_df.sample(frac=1, random_state=random_state)

    # get list of features
    X_cols = list(data_df.columns)
    X_cols.remove('user_id')
    X_cols.remove('label')

    # convert data to numpy
    X = data_df[X_cols].to_numpy()
    y = data_df['label'].to_numpy()
    target_col = data_df['user_id'].to_numpy()

    # setup models
    xgb = XGBClassifier()
    eggs_param_grid = {'sgl_method': [None], 'stacks': [1, 2], 'joint_model': [None, 'mrf'],
                       'relations': [['burst_id'], ['burst_id', 'link_id']]}
    metrics = [('roc_auc', roc_auc_score), ('aupr', average_precision_score), ('accuracy', accuracy_score)]
    models = [
        ('xgb', EGGS(estimator=xgb), None),
        ('eggs', EGGS(estimator=xgb, sgl_method=None, sgl_func=pr.pseudo_relational_features, joint_model='mrf',
                      pgm_func=pgm.create_files, relations=['burst_id'],
                      validation_size=0.2, verbose=1), eggs_param_grid)
        ]

    # setup score containers
    scores = defaultdict(list)
    all_scores = defaultdict(list)
    predictions = {name: y.copy().astype(float) for name, _, _ in models}
    predictions_binary = {name: y.copy().astype(float) for name, _, _ in models}

    # test models using cross-validation
    kf = KFold(n_splits=folds, random_state=random_state, shuffle=True)
    for fold, (train_index, test_index) in enumerate(kf.split(X)):
        print('\nfold %d...' % fold)

        # flip the training and test sets
        if invert:
            temp = train_index
            train_index = test_index
            test_index = temp

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        target_col_train, target_col_test = target_col[train_index], target_col[test_index]

        for name, model, param_grid in models:
            print('%s...' % name)

            if param_grid is not None:
                model = GridSearch(model, param_grid, scoring='average_precision', random_state=random_state, n_jobs=4)

            model = model.fit(X_train, y_train, target_col_train)
            y_hat = model.predict_proba(X_test, target_col_test)[:, 1]
            y_hat_binary = model.predict(X_test, target_col_test)

            if hasattr(model, 'best_params_'):
                print('best params: %s' % model.best_params_)

            np.put(predictions[name], test_index, y_hat)
            np.put(predictions_binary[name], test_index, y_hat_binary)

            for metric, scorer in metrics:
                score = scorer(y_test, y_hat_binary) if metric == 'accuracy' else scorer(y_test, y_hat)
                scores[name + '|' + metric].append(score)

    # compute single score using predictions from all folds
    for name, model, _ in models:
        for metric, scorer in metrics:
            score = scorer(y, predictions_binary[name]) if metric == 'accuracy' else scorer(y, predictions[name])
            all_scores[name + '|' + metric].append(score)

    print_utils.print_scores(scores, models, metrics)
    print_utils.print_scores(all_scores, models, metrics)

if __name__ == '__main__':
    main()
