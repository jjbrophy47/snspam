"""
This script uses EGGS to model a YouTube spam dataset.
"""
import os
import time
import argparse
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.linear_model import LogisticRegression
from scipy.sparse import load_npz

from EGGS.eggs import EGGS
from EGGS import print_utils


def main(args):

    # extract arguments
    featureset = args.featureset
    setting = args.setting
    stacks = args.stacks
    joint_model = args.joint
    dataset = args.dataset
    fold = args.fold

    # validate arguments
    assert featureset in ['limited', 'full']
    assert setting in ['inductive', 'inductive+transductive']
    assert stacks >= 0
    assert joint_model in [None, 'mrf', 'psl']
    assert fold >= 0
    assert args.rs >= 0

    # dataset specific settings
    if dataset == 'youtube':
        relations = ['text_id', 'user_id']
        from snspam_data.youtube.features.relational import pseudo_relational as pr
        from snspam_data.youtube.features.relational import pgm

    elif dataset == 'twitter':
        relations = ['text_id', 'user_id', 'hashuser_id']
        from snspam_data.twitter.features.relational import pseudo_relational as pr
        from snspam_data.twitter.features.relational import pgm

    elif dataset == 'soundcloud':
        relations = ['text_id', 'user_id', 'link_id']
        from snspam_data.soundcloud.features.relational import pseudo_relational as pr
        from snspam_data.soundcloud.features.relational import pgm

    else:
        raise ValueError('unknown dataset {}'.format(dataset))

    # setup output directory
    stacks_str = '{}'.format(stacks)
    joint_str = joint_model if joint_model is not None else 'None'
    out_dir = os.path.join('output', 'youtube', 'rs{}'.format(args.rs),
                           'fold{}'.format(fold), featureset, setting,
                           stacks_str, joint_str)
    os.makedirs(out_dir, exist_ok=True)

    # create logger
    logger = print_utils.get_logger(os.path.join(out_dir, 'log.txt'))
    logger.info(args)
    logger.info('timestamp: {}'.format(datetime.now()))

    # setup input directories
    logger.info('\nFOLD {}\n'.format(fold))
    data_dir = 'snspam_data/youtube/processed/folds/'
    train_dir = 'snspam_data/youtube/features/independent/%s/training_data/' % featureset
    test_dir = 'snspam_data/youtube/features/independent/%s/%s/' % (featureset, setting)

    # read in feature data
    start = time.time()
    X_train = load_npz('%strain_data_%d.npz' % (train_dir, fold)).tocsr()
    X_val = load_npz('%sval_data_%d.npz' % (train_dir, fold)).tocsr()
    X_test = load_npz('%stest_data_%d.npz' % (test_dir, fold)).tocsr()

    # read in label data
    train_df = pd.read_csv('%strain_%d.csv' % (data_dir, fold), usecols=['com_id', 'label'])
    val_df = pd.read_csv('%sval_%d.csv' % (data_dir, fold), usecols=['com_id', 'label'])
    test_df = pd.read_csv('%stest_%d.csv' % (data_dir, fold), usecols=['com_id', 'label'])

    if setting == 'inductive':
        indices_df = pd.read_csv('%stest_indices_%d.csv' % (test_dir, fold))
        test_df = test_df[test_df.index.isin(indices_df['index'])]

    # extract label data
    y_train = train_df['label'].to_numpy()
    y_val = val_df['label'].to_numpy()
    y_test = test_df['label'].to_numpy()

    # extract identifier data
    target_col_train = train_df['com_id'].to_numpy()
    target_col_val = val_df['com_id'].to_numpy()
    target_col_test = test_df['com_id'].to_numpy()

    logger.info('reading in data...{:.3f}s'.format(time.time() - start))

    # setup hyperparameters, metrics, and models
    lr = LogisticRegression(solver='liblinear', C=1000, class_weight='balanced', random_state=1)
    model = EGGS(estimator=lr, sgl_method='holdout', stacks=stacks, sgl_func=pr.pseudo_relational_features,
                 joint_model=joint_model, pgm_func=pgm.create_files, relations=relations,
                 logger=logger)

    # train
    start = time.time()
    logger.info('\nTRAINING\n')
    model = model.fit(X_train, y_train, target_col_train, X_val, y_val, target_col_val, fold=args.fold)
    logger.info('train time: {:.3f}s'.format(time.time() - start))

    # predict
    start = time.time()
    logger.info('\nTESTING\n')
    y_hat = model.predict_proba(X_test, target_col_test, fold=args.fold)[:, 1]
    logger.info('test time: {:.3f}s'.format(time.time() - start))

    # save predictions
    logger.info('\nsaving predictions to {}...'.format(out_dir))
    np.save(os.path.join(out_dir, 'predictions.npy'), y_hat)
    np.save(os.path.join(out_dir, 'y_true.npy'), y_test)


if __name__ == '__main__':

    # read in commandline args
    parser = argparse.ArgumentParser(description='EGGS: Extended Group-based Graphical models for Spam', prog='run')
    parser.add_argument('--dataset', default='youtube', help='dataset: %(default)s')
    parser.add_argument('--featureset', default='full', help='independent features, default: %(default)s')
    parser.add_argument('--setting', default='inductive+transductive', help='network setting, default: %(default)s')
    parser.add_argument('--stacks', default=0, type=int, help='number of SGL stacks, default: %(default)s')
    parser.add_argument('--joint', default=None, help='joint inference model (mrf or psl), default: %(default)s')
    parser.add_argument('--fold', default=0, type=int, help='dataset partition.')
    parser.add_argument('--rs', default=1, type=int, help='random state.')
    args = parser.parse_args()

    # run experiment
    main(args)
