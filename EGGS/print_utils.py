"""
Informational print statements for logging purposes.
"""
import sys
import logging
import numpy as np
import pandas as pd


def print_scores(scores, models, metrics, logger):
    """Display the metric results for each model."""

    rows = []
    logger.info('\n')

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
        logger.info('{}'.format(s))

    metric_names = [metric_name for metric_name, _ in metrics]
    df = pd.DataFrame(rows, columns=['model'] + metric_names)

    return df


def get_logger(filename=''):
    """
    Return a logger object to easily save textual output.
    """

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler(sys.stdout)
    log_handler = logging.FileHandler(filename, mode='w')
    formatter = logging.Formatter('%(message)s')

    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    log_handler.setLevel(logging.INFO)
    log_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(log_handler)

    return logger


def remove_logger(logger):
    """
    Remove handlers from logger.
    """
    logger.handlers = []
