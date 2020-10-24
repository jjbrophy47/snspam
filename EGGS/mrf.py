"""
This class handles loopy belief propagation using Libra.
"""
import os
import math
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.utils.validation import check_is_fitted


class MRF:
    """
    Class that performs loopy belief propagation using Libra.
    """

    def __init__(self, relations, relations_func, working_dir='.temp/', verbose=0,
                 epsilon=[0.05], scoring=None):
        """
        Initialization of the MRF model.

        Parameters
        ----------
        relations : list (default=None)
            Relations to use for relational modeling.
        relations_func : func (default=None)
            Domain-dependent helper method to generate pgm files.
        working_dir : str (default='.temp/')
            Temporary directory to store intermdiate files.
        epsilon : list (default=[0.1, 0.2, 0.3, 0.4])
            Epsilon values to try for each relation during training.
        scoring : str (default='aupr')
            Method of scoring to use for model selection during training.
        """
        self.relations = relations
        self.working_dir = working_dir
        self.relations_func = relations_func
        self.epsilon = epsilon
        self.scoring = scoring
        self.verbose = verbose

        if scoring is None:
            self.scoring = average_precision_score

    # public
    def fit(self, y, y_hat, target_col, fold=None):
        """
        Train an MRF model.
            y: true labels for target nodes. shape: (n_samples,).
            y_hat: priors for target nodes. shape: (n_samples,).
            target_col: list of target_ids. shape: (n_samples,).
        """

        relation_epsilons = {}

        # test each relation individually
        target_priors, relations_dict, target_name = self.relations_func(y_hat, target_col, self.relations, fold=fold)
        for relation_type, connections_list in relations_dict.items():
            relation_dict = {relation_type: connections_list}

            # test different epsilon values for this relation
            scores = []
            for epsilon in self.epsilon:
                print('epsilon: %.2f...' % epsilon)
                targets_dict, relation_dicts = self._generate_mn(target_priors, relation_dict,
                                                                 ep=epsilon, target_name=target_name)
                y_score = self._inference(targets_dict, relation_dicts)
                metric_score = self.scoring(y, y_score[:, 1])
                scores.append((metric_score, epsilon))

            relation_epsilons[relation_type] = sorted(scores, reverse=True)[0][1]
        self.relation_epsilons_ = relation_epsilons

        if self.verbose > 0:
            print('[MRF]: epsilons: %s' % self.relation_epsilons_)

        return self

    def inference(self, y_hat, target_col, fold=None):
        """
        Joint inference using PSL.
            y_hat: priors for target nodes.
            target_col: list of target_ids.
        """

        print('inference...')
        check_is_fitted(self, 'relation_epsilons_')
        target_priors, relations_dict, target_name = self.relations_func(y_hat, target_col, self.relations, fold=fold)
        targets_dict, relation_dicts = self._generate_mn(target_priors, relations_dict, target_name=target_name)
        y_score = self._inference(targets_dict, relation_dicts)
        return y_score

    def infer(self, df, mrf_f, rel_pred_f, ep=0.1, max_size=7500, max_edges=40000, dset='test'):
        fold = self.config_obj.fold
        relations = self.config_obj.relations
        epsilons = self.config_obj.epsilons

        g, subnets = self.conns_obj.find_subgraphs(df, relations, max_size,
                                                   max_edges)
        subgraphs = self.conns_obj.consolidate(subnets, max_size)

        res_dfs, rel_margs = [], {}
        for i, (ids, hubs, rels, edges) in enumerate(subgraphs):
            s = 'reasoning over sg_%d with %d msgs and %d edges...'
            t1 = self.util_obj.out(s % (i, len(ids), edges))
            sg_df = df[df['com_id'].isin(ids)]
            md, rd = self._gen_mn(sg_df, dset, mrf_f, ep, eps=epsilons)
            self._run(mrf_f, dset=dset)
            res_df, r_margs = self._process_marginals(md, rd, mrf_f, dset=dset,
                                                      pred_dir=rel_pred_f)
            res_dfs.append(res_df)
            rel_margs.update(r_margs)
            self.util_obj.time(t1)
        preds_df = pd.concat(res_dfs)
        preds_df = preds_df.groupby('com_id')['mrf_pred'].mean().reset_index()
        preds_df.to_csv(rel_pred_f + 'mrf_preds_' + fold + '.csv', index=None)

        return res_df

    # private
    def _inference(self, targets_dict, relation_dicts):

        model_name = '%smodel.mn' % self.working_dir
        marginals_name = '%smarginals.txt' % self.working_dir
        execute = 'libra bp -m %s -mo %s' % (model_name, marginals_name)
        os.system(execute)

        targets, y_score = self._marginals(targets_dict, relation_dicts)
        return y_score

    def _compute_aupr(self, preds_df, val_df):
        df = preds_df.merge(val_df, on='com_id', how='left')
        aupr = average_precision_score(df['label'], df['mrf_pred'])
        return aupr

    def _generate_mn(self, target_priors, relations_dict, ep=0.1, eps={}, target_name='user_id'):
        """
        Generates predicate files for PSL.
            target_priors: list of (target_id, prior) tuples.
            relations_dict: key=relation_type, value=list of (hub_id, target_id) tuples.
        """

        # create target nodes
        targets_dict = {}
        for i, (target_id, prior) in enumerate(target_priors):
            targets_dict[target_id] = {'ndx': i, 'prior': prior, 'card': 2}

        target_ids = [target_id for target_id, prior in target_priors]
        ndx = len(target_ids)

        # create relational hub nodes
        relation_dicts = []

        # build a dict for each relation type
        for relation_type, connection_list in relations_dict.items():
            hubs_dict = {}

            # save connections from each hub node for this relation type
            df = pd.DataFrame(connection_list, columns=[relation_type, target_name])
            for relation_id, hub_df in df.groupby(relation_type):
                connections = list(set(target_ids).intersection(set(list(hub_df[target_name]))))
                hubs_dict[relation_id] = {'ndx': ndx, relation_id: connections}
                ndx += 1

            relation_dicts.append((hubs_dict, relation_type))
        self._write_model_file(targets_dict, relation_dicts, ndx, ep=ep, eps=eps)

        return targets_dict, relation_dicts

    def _network_size(self, targets_dict, relation_dicts, dset='val'):
        ut = self.util_obj
        total_nodes, total_edges = len(targets_dict), 0
        ut.out('%s network:' % dset)

        ut.out('-> target nodes: %d' % (len(targets_dict)))
        for relation_dict, relation in relation_dicts:
            total_nodes += len(relation_dict)
            edges = 0

            for group_id, group_dict in relation_dict.items():
                edges += len(group_dict[relation])

            t = (relation, len(relation_dict), edges)
            ut.out('-> %s nodes: %d, edges: %d' % t)
            total_edges += edges

        ut.out('-> all nodes: %d, all edges: %d' % (total_nodes, total_edges))
        return total_edges

    def _marginals(self, targets_dict, relation_dicts):
        fname = '%smarginals.txt' % self.working_dir
        posteriors = []
        targets = []
        rel_margs = {}

        with open(fname, 'r') as f:
            for i, line in enumerate(f.readlines()):
                for target_id, target_dict in targets_dict.items():
                    if target_dict['ndx'] == i:  # makes sure targets are in the same order
                        pred = [float(x) for x in line.split(' ')]
                        targets.append(target_id)
                        posteriors.append(pred)

                for rels_dict, rel in relation_dicts:
                    for rel_id, rel_dict in rels_dict.items():
                        if rel_dict['ndx'] == i:
                            marg_val = round(float(line.split(' ')[1]), 2)
                            rel_margs[rel + '_' + str(rel_id)] = marg_val

        return targets, np.array(posteriors)

    def _transform_priors(self, df, col='ind_pred', transform='logit'):
        clf = self.config_obj.classifier
        df = df.copy()

        if clf != 'lr':
            if transform is not None:
                if transform == 'e':
                    scale = self._transform_e
                elif transform == 'logit':
                    scale = self._transform_logit
                elif transform == 'logistic':
                    scale = self._transform_logistic

                df['ind_pred'] = df['ind_pred'].apply(scale)
        return df

    def _transform_e(self, x):
        result = x

        if x == 0:
            result = 0
        elif x == 1:
            result == 1
        else:
            result = math.exp(x)
        return result

    def _transform_logistic(self, x, alpha=2):
        result = x

        if x == 0:
            result = 0
        elif x == 1:
            result == 1
        else:
            result = (x ** alpha) / (x + ((1 - x) ** alpha))
        return result

    def _transform_logit(self, x):
        result = x

        if x == 0:
            result = 0
        elif x == 1:
            result == 1
        else:
            result = math.log(x / (1 - x))
        return result

    def _write_model_file(self, targets_dict, relation_dicts, num_nodes, ep=0.15, eps={}):
        """
        Generates predicate files for PSL.
            targets_dict: key=target_id, value=dict(ndx: v1, prior: v2, card: v3)
            relations_dict: list of (hubs_dict, relation_type) tuples.
                            hubs_dict: key=relation_id, value=dict(ndx: v1, relation_id: list of target_ids)
        """

        dir = self.working_dir

        if not os.path.exists(dir):
            os.makedirs(dir)

        with open(dir + 'model.mn', 'w') as f:
            # first line: comma separated cardinality for each node
            line = ''
            for i in range(num_nodes):
                line += '2\n' if i == num_nodes - 1 else '2,'
            f.write(line)

            # start network
            f.write('MN {\n')

            # write single node factors
            for i, (target_id, target_dict) in enumerate(targets_dict.items()):
                assert i == target_dict['ndx']
                prior = target_dict['prior']
                ndx = target_dict['ndx']
                factor = '%.5f +v%d_1\n'
                factor += '%.5f +v%d_0\n'
                f.write(factor % (prior, ndx, 1.0 - prior, ndx))

            # write pairwise node factors
            for hubs_dict, relation_type in relation_dicts:
                for hub_id, hub_dict in hubs_dict.items():
                    epsilon = ep if eps == {} else eps[relation_type]
                    relation_ndx = hub_dict['ndx']
                    target_ids = hub_dict[hub_id]

                    for target_id in target_ids:
                        target_dict = targets_dict[target_id]
                        target_ndx = target_dict['ndx']

                        factor = '%.5f +v%d_0 +v%d_0\n'
                        factor += '%.5f +v%d_0 +v%d_1\n'
                        factor += '%.5f +v%d_1 +v%d_0\n'
                        factor += '%.5f +v%d_1 +v%d_1\n'

                        values = (1.0 - epsilon, target_ndx, relation_ndx)
                        values += (epsilon, target_ndx, relation_ndx)
                        values += (epsilon, target_ndx, relation_ndx)
                        values += (1.0 - epsilon, target_ndx, relation_ndx)

                        f.write(factor % values)
            f.write('}\n')
