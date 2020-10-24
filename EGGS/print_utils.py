"""
Informational print statements for logging purposes.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm


def print_datasets(X_cols, X_train, y_train, X_test, y_test=None, max_cols=100):
    """Display basic train and test set statistics."""

    if len(X_cols) < max_cols:
        print('indicators (%d): %s' % (len(X_cols), X_cols))
    else:
        print('%d indicators' % len(X_cols))

    train_pos = len(X_train[np.where(y_train == 1)])
    train_neg = len(X_train[np.where(y_train == 0)])
    train_pos_pct = round(train_pos / len(X_train) * 100)
    train_neg_pct = round(train_neg / len(X_train) * 100)
    s = '\ntraining set: %d images; %d manipulated (%d%%), %d non-manipulated (%d%%)'
    print(s % (len(X_train), train_pos, train_pos_pct, train_neg, train_neg_pct))

    s = 'test set: %d images'
    t = (len(X_test),)
    if y_test is not None:
        test_pos = len(X_test[np.where(y_test == 1)])
        test_neg = len(X_test[np.where(y_test == 0)])
        t += (test_pos, round(test_pos / len(X_test) * 100))
        t += (test_neg, round(test_neg / len(X_test) * 100))
        s += '; %d manipulated (%d%%), %d non-manipulated (%d%%)'
    print(s % t)


def print_model(model, X_cols):
    """Display components of fitted model."""

    name = str(model).split('(')[0]

    if name == 'RandomForestClassifier' or name == 'LGBMClassifier' or name == 'XGBClassifier':
        importances = model.feature_importances_
        indices = np.argsort(importances)

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(range(len(indices)), importances[indices], align='center', color='g')
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([X_cols[i] for i in indices])
        ax.set_xlabel('Relative Importance')
        ax.set_title(name.lower())
        ax.xaxis.grid(b=True, which='major')
        plt.show()

    if name == 'LogisticRegression':
        coefficients = model.coef_[0]
        indices = np.argsort(coefficients)

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(range(len(indices)), coefficients[indices], align='center', color='b')
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([X_cols[i] for i in indices])
        ax.set_xlabel('Coefficients')
        ax.set_title('Logistic Regression, intercept: %s' % model.intercept_)
        ax.xaxis.grid(b=True, which='major')
        plt.show()

    if name == 'GaussianNB':
        means, stds = model.theta_, model.sigma_

        ncols = 4
        nrows = int(means.shape[1] / ncols) + (1 if means.shape[1] % ncols > 0 else 0)
        height = 4 * nrows
        fig, axs = plt.subplots(nrows, ncols, figsize=(18, height))
        axs = axs.reshape(-1)

        for i in range(len(means)):  # each class
            class_means, class_stds = means[i], stds[i]

            for j in range(len(class_means)):  # each feature
                feat_mean, feat_std = class_means[j], class_stds[j]

                x = np.linspace(feat_mean - 3 * feat_std, feat_mean + 3 * feat_std, 100)
                axs[j].plot(x, norm.pdf(x, loc=feat_mean, scale=feat_std), label='label=%d' % i)
                axs[j].set_title(X_cols[j])
                axs[j].legend()
        plt.show()


def print_scores(scores, models, metrics, to_csv=True, out_dir='output/scores/'):
    """Display the metric results for each model."""

    rows = []
    print()

    for name, _, _ in models:
        row = [name]
        s = name

        for metric, _ in metrics:
            key = name + '|' + metric
            mean_score = np.mean(scores[key])
            mean_std = np.std(scores[key])
            score_str = '%.3f +/- %.2f' % (mean_score, mean_std)
            s += ' (%s): ' % metric
            s += score_str
            row.append(score_str)
        rows.append(row)
        print(s)

    if to_csv:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        metric_names = [metric_name for metric_name, _ in metrics]
        df = pd.DataFrame(rows, columns=['model'] + metric_names)
        df.to_csv('%sscores.csv' % out_dir, index=None)
