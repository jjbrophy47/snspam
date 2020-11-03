"""
This script uses EGGS to model a YouTube spam dataset.
"""
import os
import argparse
import numpy as np
from datetime import datetime

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

from EGGS import print_utils


def main(featureset='full', setting='inductive+transductive', stacks=0, joint_model='mrf'):

    assert featureset in ['limited', 'full']
    assert setting in ['inductive', 'inductive+transductive']
    assert stacks >= 0
    assert joint_model in [None, 'mrf', 'psl']
    assert args.rs >= 0

    # setup output directory
    stacks_str = '{}'.format(stacks)
    joint_str = joint_model if joint_model is not None else 'None'
    out_dir = os.path.join('output', 'scores', 'youtube',
                           'rs{}'.format(args.rs), featureset, setting,
                           stacks_str, joint_str)
    os.makedirs(out_dir, exist_ok=True)

    # create logger
    logger = print_utils.get_logger(os.path.join(out_dir, 'log.txt'))
    logger.info(args)
    logger.info('timestamp: {}'.format(datetime.now()))

    # setup score containers
    y_hat_list = []
    y_true_list = []

    # combine predictions from all folds
    for fold in args.fold:
        logger.info('getting predictions from fold {}...'.format(fold))

        in_dir = os.path.join('output', 'youtube', 'rs{}'.format(args.rs),
                              'fold{}'.format(fold), featureset, setting,
                              stacks_str, joint_str)

        # save predictions
        y_hat = np.load(os.path.join(in_dir, 'predictions.npy'))
        y_true = np.load(os.path.join(in_dir, 'y_true.npy'))

        y_hat_list.append(y_hat)
        y_true_list.append(y_true)

    # combine predictions from all folds and generate scores
    logger.info('combining predictions...')
    y_hat = np.hstack(y_hat_list)
    y_true = np.hstack(y_true_list)

    auc = roc_auc_score(y_true, y_hat)
    ap = average_precision_score(y_true, y_hat)

    logger.info('AUC: {:.5f}, AP: {:.5f}'.format(auc, ap))


if __name__ == '__main__':

    # read in commandline args
    parser = argparse.ArgumentParser(description='EGGS: Extended Group-based Graphical models for Spam', prog='run')
    parser.add_argument('--featureset', default='full', help='independent features, default: %(default)s')
    parser.add_argument('--setting', default='inductive+transductive', help='network setting, default: %(default)s')
    parser.add_argument('--stacks', default=0, type=int, help='number of SGL stacks, default: %(default)s')
    parser.add_argument('--joint', default=None, help='joint inference model (mrf or psl), default: %(default)s')
    parser.add_argument('--fold', default=list(range(10)), type=int, nargs='+', help='dataset partitions.')
    parser.add_argument('--rs', default=1, type=int, help='random state.')
    args = parser.parse_args()

    # run experiment
    main(featureset=args.featureset, setting=args.setting, stacks=args.stacks, joint_model=args.joint)
