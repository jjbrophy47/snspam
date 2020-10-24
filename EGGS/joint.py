"""
This module contains high-level APIs implementing joint inference using PGMs.
"""
import os
import uuid
import shutil
from .mrf import MRF
from .psl import PSL
from sklearn.utils.validation import check_is_fitted


class Joint:
    """
    High-level class with multiple pgm implementations.
    """

    def __init__(self, relations, relations_func, pgm_type='psl', working_dir='.temp/'):
        """
        Initialization of joint inference class.

        Parameters
        ----------
        relations : list (default=None)
            Relations to use for relational modeling.
        relations_func : func (default=None)
            Domain-dependent helper method to generate pgm files.
        pgm_type : str (default='psl') {'psl', 'mrf'}
            Type of PGM to use for joint inference.
        working_dir : str (default='.temp/')
            Temporary directory to store intermediate files.
        """
        self.relations = relations
        self.relations_func = relations_func
        self.pgm_type = pgm_type
        self.working_dir = working_dir + str(uuid.uuid4()) + '/'

        # clear the working directory
        if os.path.exists(self.working_dir):
            shutil.rmtree(self.working_dir)
        os.makedirs(self.working_dir)

    def fit(self, y, y_hat, target_col, fold=None):
        """Trains a PGM model.
            y: true labels for target nodes. shape: (n_samples,).
            y_hat: priors for target nodes. shape: (n_samples,).
            target_col: list of target_ids. shape: (n_samples,).
        """

        pgm_class = PSL if self.pgm_type == 'psl' else MRF
        pgm = pgm_class(self.relations, self.relations_func, self.working_dir)
        self.pgm_ = pgm.fit(y, y_hat, target_col, fold=fold)
        return self

    def inference(self, y_hat, target_col, fold=None):
        """Performs joint inference.
            y_hat: priors for target nodes. shape: (n_samples,).
            target_col: list of target_ids. shape: (n_samples,).
        """

        assert len(y_hat) == len(target_col)
        check_is_fitted(self, 'pgm_')

        y_hat = self.pgm_.inference(y_hat, target_col, fold=fold)
        return y_hat
