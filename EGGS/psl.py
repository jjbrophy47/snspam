"""
This class handles inference using PSL.
"""
import os
import time

import numpy as np
import pandas as pd
from sklearn.utils.validation import check_is_fitted

from .connections import Connections


class PSL:
    """
    Class that performs inference using PSL.
    """

    def __init__(self, relations, relations_func,
                 working_dir='.temp/', psl_dir='psl/',
                 logger=None):
        """
        Initialization of joint inference class.

        Parameters
        ----------
        relations : list (default=None)
            Relations to use for relational modeling.
        relations_func : func (default=None)
            Domain-dependent helper method to generate pgm files.
        working_dir : str (default='.temp/')
            Temporary directory to store intermdiate files.
        psl_dir : str (default='psl/')
            Directory where psl is installed.
        logger : obj (default=None)
            Logs output.
        """
        self.relations = relations
        self.relations_func = relations_func
        self.working_dir = working_dir
        self.psl_dir = psl_dir
        self.logger = logger
        self.compiled_ = True

    # public
    def fit(self, y, y_hat, target_col, fold=None):
        """
        Train a PSL model.
            y: true labels for target nodes. shape: (n_samples,).
            y_hat: priors for target nodes. shape: (n_samples,).
            target_col: list of target_ids. shape: (n_samples,).
        """

        start = time.time()

        target_priors, relations_dict, target_name = self.relations_func(y_hat, target_col, self.relations, fold=fold)
        self._generate_files(target_priors, relations_dict, target_name=target_name, y_true=y)
        self._train(target_name, target_col, y_hat, y)
        self.fitted_ = True

        if self.logger:
            self.logger.info('time: {:.3f}s'.format(time.time() - start))

        return self

    def inference(self, y_hat, target_col, fold=None):
        """
        Joint inference using PSL.
            y_hat: priors for target nodes. shape: (n_samples,).
            target_col: list of target_ids. shape: (n_samples,).
        """

        check_is_fitted(self, 'fitted_')
        target_priors, relations_dict, target_name = self.relations_func(y_hat, target_col, self.relations, fold=fold)

        y_score = self.infer(target_priors, relations_dict)

        return y_score

    # private
    def _generate_files(self, target_priors, relations_dict, target_name='com_id', y_true=None):
        """
        Generates predicate files for PSL.
            target_priors: list of (target_id, prior) tuples.
            relations_dict: key=relation_type, value=list of (hub_id, target_id) tuples.
        """

        # generate target files
        nolabel_fname = '%s%s_nolabel.tsv' % (self.working_dir, target_name)
        prior_fname = '%s%s_prior.tsv' % (self.working_dir, target_name)
        target_df = pd.DataFrame(target_priors, columns=[target_name, 'y_hat'])
        target_df.to_csv(nolabel_fname, columns=[target_name], sep='\t', header=None, index=None)
        target_df.to_csv(prior_fname, columns=[target_name, 'y_hat'], sep='\t', header=None, index=None)

        # target file with labels
        if y_true is not None:
            assert len(y_true) == len(target_priors)
            label_fname = '%s%s_label.tsv' % (self.working_dir, target_name)
            label_df = pd.DataFrame(list(zip(target_df[target_name], y_true)), columns=[target_name, 'y_true'])
            label_df.to_csv(label_fname, sep='\t', header=None, index=None)

        # generate relational files
        for relation_type, hubs_list in relations_dict.items():
            hub_fname = '%s%s.tsv' % (self.working_dir, relation_type)
            hubconns_fname = '%s%s_connections.tsv' % (self.working_dir, relation_type)

            # id of each hub, and the id of each neighbor
            hub_df = pd.DataFrame(hubs_list, columns=[relation_type, target_name])
            hub_df.to_csv(hubconns_fname, columns=[target_name, relation_type], sep='\t', header=None, index=None)

            # ids of each hub
            hub_df = hub_df.drop_duplicates(subset=[relation_type])
            hub_df.to_csv(hub_fname, columns=[relation_type], sep='\t', header=None, index=None)

        # generate rules file
        rules = self._priors()
        for relation_id in self.relations:
            rules += self._map_relation_to_rules(relation_id)
        self._write_rules(rules)

    def infer(self, target_priors, relations_dict, max_size=40000, max_edges=40000):

        conns_obj = Connections()
        df = pd.DataFrame(target_priors, columns=['com_id', 'ind_pred'])

        g, subnets = conns_obj.find_subgraphs(df, relations_dict, max_size, max_edges, logger=self.logger)
        subgraphs = conns_obj.consolidate(subnets, max_size, logger=self.logger)

        results = []
        for i, (ids, hubs, rels, edges) in enumerate(subgraphs):
            start = time.time()

            sg_ids = [int(x) for x in ids]
            temp_df = df[df['com_id'].isin(sg_ids)]
            sg_priors = list(zip(temp_df['com_id'], temp_df['ind_pred']))

            target_col, y_hat = temp_df['com_id'].values, temp_df['ind_pred'].values

            self._generate_files(sg_priors, relations_dict, target_name='com_id')
            y_score = self._inference('com_id', target_col, y_hat)
            result_df = pd.DataFrame(list(zip(target_col, y_score[:, 1])), columns=['com_id', 'pgm_pred'])

            results.append(result_df)

            if self.logger:
                s = 'reasoning over sg_{} with {:,} msgs and {:,} edges...{:.3f}s'
                self.logger.info(s.format(i, len(ids), edges, time.time() - start))

        preds_df = pd.concat(results)
        preds_df = preds_df.groupby('com_id')['pgm_pred'].mean().reset_index()

        result_df = df.merge(preds_df, on='com_id', how='left')
        result_df['pgm_pred'] = result_df['pgm_pred'].fillna(result_df['ind_pred'])

        y_score = np.array(result_df['pgm_pred'].values)
        y_score = np.hstack([1 - y_score.reshape(-1, 1), y_score.reshape(-1, 1)])

        return y_score

    # def infer(self, df, psl_d, psl_f, rel_d, max_size=500000):
    #     fold = self.config_obj.fold
    #     relations = self.config_obj.relations

    #     g, ccs = self.conns_obj.find_subgraphs(df, relations, max_size)
    #     subgraphs = self.conns_obj.consolidate(ccs, max_size)

    #     for i, (ids, hubs, rels, edges) in enumerate(subgraphs):
    #         _id = i + int(fold)
    #         sg_df = df[df['com_id'].isin(ids)]
    #         self._gen_predicates(sg_df, 'test', psl_d, _id)
    #         self._network_size(psl_d, _id, dset='test')

    #         t1 = self.util_obj.out('reasoning over sg_%d...' % i)
    #         self._run(psl_f, _id, action='Infer')
    #         self.util_obj.time(t1)
    #     self._combine_predictions(len(subgraphs), rel_d)

    def _inference(self, target_name, target_col, y_hat):

        arg_list = ['../%s' % self.working_dir, target_name]
        execute = 'java -Xmx60g -cp ./target/classes:`cat classpath.out` '
        execute += 'spam.Infer' + ' ' + ' '.join(arg_list)

        # only need to do this once
        self._compile()

        cwd = os.getcwd()
        os.chdir(self.psl_dir)  # change to psl directory
        os.system(execute)
        os.chdir(cwd)  # change back to original directory

        score_df = pd.read_csv('%spsl_scores.tsv' % self.working_dir, sep='\t')
        target_df = pd.DataFrame(list(zip(target_col, y_hat)), columns=[target_name, 'y_hat_old'])
        target_df = target_df.merge(score_df, on=target_name, how='left')
        target_df['y_hat'] = target_df['y_hat'].fillna(target_df['y_hat_old'])

        y_hat = target_df['y_hat'].to_numpy()
        y_score = np.hstack([1 - y_hat.reshape(-1, 1), y_hat.reshape(-1, 1)])
        assert len(y_score) == len(target_col)

        return y_score

    def _train(self, target_name, target_col, y_hat, y):
        arg_list = ['../%s' % self.working_dir, target_name]
        execute = 'java -Xmx60g -cp ./target/classes:`cat classpath.out` '
        execute += 'spam.Train' + ' ' + ' '.join(arg_list)

        self._compile()

        cwd = os.getcwd()
        os.chdir(self.psl_dir)  # change to psl directory
        os.system(execute)
        os.chdir(cwd)  # change back to original directory

    def _compile(self):

        if not self.compiled_:
            mvn_compile = 'mvn compile -q'
            mvn_build = 'mvn dependency:build-classpath '
            mvn_build += '-Dmdep.outputFile=classpath.out -q'

            cwd = os.getcwd()
            os.chdir(self.psl_dir)  # change to psl directory
            os.system(mvn_compile)
            os.system(mvn_build)
            os.chdir(cwd)  # change back to original directory
            self.compiled_ = True

    def _priors(self, wgt=1.0, sq=True):
        neg_prior = str(wgt) + ': ~spam(Target)'
        pos_prior = str(wgt) + ': prior(Target) -> spam(Target)'

        if sq:
            neg_prior += ' ^2'
            pos_prior += ' ^2'

        return [neg_prior, pos_prior]

    def _map_relation_to_rules(self, relation_id, wgt=1.0, sq=True):
        hub_name = relation_id.split('_')[0]

        rule1 = str(wgt) + ': '
        rule2 = str(wgt) + ': '

        atom1 = 'has' + hub_name + '(Target, ' + hub_name.capitalize() + ')'
        atom2 = 'spmy' + hub_name + '(' + hub_name.capitalize() + ')'
        atom3 = 'spam(Target)'

        rule1 += atom1 + ' & ' + atom2 + ' -> ' + atom3
        rule2 += atom1 + ' & ' + atom3 + ' -> ' + atom2

        if sq:
            rule1 += ' ^2'
            rule2 += ' ^2'

        return [rule1, rule2]

    def _write_rules(self, rules):
        with open('%srules.txt' % self.working_dir, 'w') as w:
            for rule in rules:
                w.write(rule + '\n')

    def _network_size(self, data_f, iden=None, dset='val'):
        s_iden = self.config_obj.fold if iden is None else str(iden)
        relations = self.config_obj.relations
        all_nodes, all_edges = 0, 0

        self.util_obj.out('%s network:' % dset)
        fn_m = data_f + dset + '_' + s_iden + '.tsv'
        msg_nodes = self.util_obj.file_len(fn_m)
        self.util_obj.out('-> msg nodes: %d' % msg_nodes)
        all_nodes += msg_nodes

        for relation, group, group_id in relations:
            fn_r = data_f + dset + '_' + relation + '_' + s_iden + '.tsv'
            fn_g = data_f + dset + '_' + group + '_' + s_iden + '.tsv'
            edges = self.util_obj.file_len(fn_r)
            hubs = self.util_obj.file_len(fn_g)
            t = (relation, hubs, edges)
            self.util_obj.out('-> %s nodes: %d, edges: %d' % t)

            all_edges += edges
            all_nodes += hubs

        t = (all_nodes, all_edges)
        self.util_obj.out('-> all nodes: %d, all edges: %d' % t)
        return all_edges

    def _collect_connected_components_stats(self, ccs, df, rel_d):
        fold = self.config_obj.fold
        t1 = self.util_obj.out('collecting connected components stats...')

        df_cols = ['size', 'same', 'mean', 'median', 'std', 'max', 'min']
        df_rows = []
        ccs = [x for x in ccs if x[3] > 0]  # filter out no edge subgraphs

        for msg_nodes, hub_nodes, relations, edges in ccs:
            qf = df[df['com_id'].isin(msg_nodes)]
            ip = qf['ind_pred']

            size = len(msg_nodes)
            mean = np.mean(ip)
            median = np.median(ip)
            same = 1 if np.allclose(ip, ip[::-1], atol=1e-4) \
                and np.isclose(mean, median, atol=1e-8) else 0
            std = np.std(ip)
            mx = np.max(ip)
            mn = np.min(ip)
            row = [size, same, mean, median, std, mx, mn]

            label_col = 'label' if 'label' in list(qf) else \
                'is_attributed' if 'is_attributed' in list(qf) else None

            if label_col is not None:
                il = qf['label']
                lab_mean = np.mean(il)
                lab_diff = np.mean(np.abs(np.subtract(ip, il)))
                row.append(lab_mean)
                row.append(lab_diff)

            df_rows.append(row)
        self.util_obj.time(t1)

        if len(df_rows[0]) > 7:
            df_cols += ['lab_mean', 'lab_diff']

        sg_dir = rel_d + '../subgraphs/'
        self.util_obj.create_dirs(sg_dir)
        fname = sg_dir + 'sg_stats_%s.csv' % fold

        if os.path.exists(fname):
            old_df = pd.read_csv(fname)
            new_df = pd.DataFrame(df_rows, columns=df_cols)
            df = pd.concat([old_df, new_df])
        else:
            df = pd.DataFrame(df_rows, columns=df_cols)

        df.sort_values('size').to_csv(fname, index=None)
        return df
