"""
Module containing various utility methods.
"""
import numpy as np
import pandas as pd
from time import time
from sklearn.base import clone
from sklearn.model_selection import PredefinedSplit
from sklearn.model_selection import GridSearchCV
from scipy import sparse


def cross_val_predict(X, y, model, param_grid=None, group_ids=None, num_cvfolds=10, num_tunefolds=3, seed=69,
                      verbose=0):
    """Generates predictions for all instances in X using cross-validation."""

    cv_folds = generate_folds(X, group_ids=group_ids, num_folds=num_cvfolds, seed=seed)

    p = y.copy().astype(float)  # predictions

    for i, (train_index, test_index) in enumerate(PredefinedSplit(cv_folds).split()):
        t2 = time()

        X_train, y_train = X[train_index], y[train_index]
        X_test = X[test_index]

        if param_grid is not None:
            tune_folds = generate_folds(X_train, num_folds=num_tunefolds, seed=seed)
            model = GridSearchCV(clone(model), cv=PredefinedSplit(tune_folds), param_grid=param_grid, verbose=0)

        clf = clone(model).fit(X_train, y_train)
        y_score = clf.predict_proba(X_test)[:, 1]
        np.put(p, test_index, y_score)

        print('  fold %d...%s' % (i, time(t2))) if verbose > 1 else None

    assert len(p) == len(y)
    return p


def generate_folds(X, group_ids=None, num_folds=10, seed=69):
    """
    Partitions rows in a dataframe into different groups.

    Parameters
    ----------
    X: data
        2D array containing rows of data.
    group_ids: list
        Group id for each image.
    num_folds : int, default: 10
        Number of groups to seperate rows into.
    seed : int, default: 69
        Random seed generator for reproducibility.

    Returns
    -------
    List of folds ids grouping instance from the same journal together.
    """
    if group_ids is None or pd.isnull(group_ids).all():
        np.random.seed(seed)
        folds = np.random.randint(num_folds, size=len(X))
    else:
        df = pd.DataFrame(X).reset_index().rename(columns={'index': 'image_id'})
        df['group'] = pd.Series(group_ids).replace('nan', np.nan)

        np.random.seed(seed)
        g = df[pd.isnull(df['group'])].groupby('image_id').size().reset_index()
        g['group_other'] = np.random.randint(num_folds, size=len(g))
        g = g[['image_id', 'group_other']]
        df = df.merge(g, on='image_id', how='left')

        np.random.seed(seed)
        g = df.groupby('group').size().reset_index()
        g['group_temp'] = np.random.randint(num_folds, size=len(g))
        g = g[['group', 'group_temp']]
        df = df.merge(g, on='group', how='left')
        df['fold'] = df['group_other'].fillna(df['group_temp'])

        df['fold'] = df['fold'].apply(int)
        folds = df['fold']
    return folds


def hstack(blocks):
    """Horizontally stacks sparse or non-sparse blocks."""

    # checks if all blocks are sparse
    sparse_blocks = 0
    for block in blocks:
        if sparse.issparse(block):
            sparse_blocks += 1

    # stacks the blocks together
    if sparse_blocks == len(blocks):
        result = sparse.hstack(blocks)

    elif sparse_blocks == 0:
        result = np.hstack(blocks)

    else:
        raise ValueError('Sparse and non-sparse blocks present!')

    return result


def array_split(X, splits):
    """Splits sparse array into equal-sized pieces."""

    assert splits > 1

    array_list = []
    n = X.shape[0]
    incrementer = int(n / splits)

    for i in range(1, splits + 1):

        if i == splits:
            array_fragment = X[(i - 1) * incrementer:]
        else:
            array_fragment = X[(i - 1) * incrementer: i * incrementer]

        array_list.append(array_fragment)

    return np.array(array_list)
