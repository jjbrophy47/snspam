"""
Module to find subnetwork of a data points based on its relationships.
"""
import time
from collections import Counter

import numpy as np
import pandas as pd
import networkx as nx


class Connections:

    def __init__(self):
        self.size_threshold = 100

    # public
    def build_networkx_graph(self, prior_df, relations_dict, target_col='com_id', logger=None):
        """
        Builds relational graph to find connected components.
        """
        graph = nx.Graph()

        # create edges from target nodes to hub nodes
        for i, (relation_col, connection_list) in enumerate(relations_dict.items()):
            if logger:
                logger.info('[RELATION {}]'.format(relation_col))

            relation_df = pd.DataFrame(connection_list, columns=[relation_col, target_col])
            relation_df = relation_df[relation_df[target_col].isin(prior_df[target_col].unique())]

            for j, (hub_id, target_id) in enumerate(zip(relation_df[relation_col], relation_df[target_col])):
                graph.add_edge('{}'.format(target_id), '{}-{}'.format(relation_col, hub_id))

                if j % 100 == 0:
                    if logger:
                        logger.info('[CONNECTION {} / {}]'.format(j, len(relation_df)))

        return graph

    def build_networkx_graph_old(self, df, relations):
        g = nx.Graph()
        h = {h: i + 1 for i, h in enumerate(list(df))}

        for r in df.itertuples():
            msg_id = r[h['com_id']]

            for rel, group, g_id in relations:
                gids = r[h[g_id]]

                if type(gids) == list:
                    for gid in gids:
                        g.add_edge(msg_id, rel + '_' + str(gid))
                elif gids != -1:
                    g.add_edge(msg_id, rel + '_' + str(gids))
        return g

    def build_networkx_graph_rdfs(self, r_dfs, relations):
        g = nx.Graph()

        rdict = {g_id: rel for rel, group, g_id in relations}

        for r_df in r_dfs:
            rel = list(r_df)[1]
            for r in r_df.itertuples():
                msg_id, g_id = r[1], r[2]
                g.add_edge(msg_id, rdict[rel] + '_' + str(g_id))
        return g

    def consolidate(self, subgraphs, max_size=40000, div=2, logger=None):
        """Combine subgraphs into larger sets to reduce total number of
        subgraphs to do inference over."""
        start = time.time()

        sgs = []
        new_ids, new_hubs = set(), set()
        new_rels, new_edges = set(), 0

        for ids, hubs, rels, edges in subgraphs:
            size = int(len(new_ids) / div) + int(len(ids) / div)
            size += new_edges + edges

            if size < max_size:  # keep adding to new
                new_ids.update(ids)
                new_rels.update(rels)
                new_hubs.update(hubs)
                new_edges += edges
            else:  # new is full, start a new subgraph
                if len(new_ids) == 0:  # nothing in new
                    sgs.append((ids, hubs, rels, edges))

                else:  # new has leftovers, flush
                    sgs.append((new_ids, new_hubs, new_rels, new_edges))
                    new_ids, new_hubs = ids, hubs
                    new_rels, new_edges = rels, edges

        if len(new_ids) > 0:
            sgs.append((new_ids, new_hubs, new_rels, new_edges))

        if logger:
            logger.info('consolidating subgraphs...{:.3f}s'.format(time.time() - start))
        self._print_subgraphs_size(sgs, logger=logger)

        return sgs

    def find_subgraphs(self, df, relations, max_size=40000, max_edges=-1, logger=None):

        start = time.time()
        g = self.build_networkx_graph(df, relations)
        ccs = list(nx.connected_components(g))
        if logger:
            logger.info('building networkx graph...{:.3f}s'.format(time.time() - start))

        if max_edges > 0:
            start = time.time()
            ccs = self._partition_large_components(ccs, g, max_edges)
            if logger:
                logger.info('partitioning very large subgraphs...{:.3f}s'.format(time.time() - start))

        start = time.time()
        subgraphs = self._process_components(ccs, g)
        if logger:
            logger.info('processing connected components...{:.3f}s'.format(time.time() - start))

        start = time.time()
        subgraphs = self._filter_redundant_subgraphs(subgraphs, df)
        if logger:
            logger.info('filtering redundant subgraphs...{:.3f}s'.format(time.time() - start))

        start = time.time()
        subgraphs = self._remove_single_edge_hubs(subgraphs, g)
        if logger:
            logger.info('removing single edge hubs...{:.3f}s'.format(time.time() - start))

        # start = time.time()
        # subgraphs += self._single_node_subgraphs(subgraphs, df, max_size)
        # logger.info('compiling single node subgraphs...{:.3f}s'.format(time.time() - start))

        self._print_subgraphs_size(subgraphs, logger=logger)

        return g, subgraphs

    def find_target_subgraph(self, com_id, df, relations, debug=False):
        """Public interface to find a subnetwork given a specified comment.
        com_id: identifier of target comment.
        df: comments dataframe.
        relations: list of relations as tuples.
        debug: boolean to print extra information.
        Returns set of comment ids in subnetwork, and relations used."""
        direct, rels, edges = self._direct_connections(com_id, df, relations)

        if len(direct) < self.size_threshold:
            result = self._group(com_id, df, relations)
        else:
            result = self._iterate(com_id, df, relations)
        return result

    # private
    def _aggregate_single_node_subgraphs(self, subnets):
        no_rel_ids = [s.pop() for s, r, e in subnets if r == set()]
        no_rel_sg = (set(no_rel_ids), set(), 0)
        rel_sgs = [(s, r, e) for s, r, e in subnets if r != set()]

        subgraphs = rel_sgs.copy()
        subgraphs.append(no_rel_sg)
        return subgraphs

    def _direct_connections(self, com_id, df, possible_relations):
        com_df = df[df['com_id'] == com_id]
        subnetwork, relations, edges = set(), set(), 0

        list_filter = lambda l, v: True if v in l else False

        for relation, group, group_id in possible_relations:
            g_vals = com_df[group_id].values

            if len(g_vals) > 0:
                vals = g_vals[0]

                for val in vals:
                    rel_df = df[df[group_id].apply(list_filter, args=(val,))]
                    if len(rel_df) > 1:
                        relations.add(relation)
                        subnetwork.update(set(rel_df['com_id']))
                        edges += 1
        return subnetwork, relations, edges

    def _filter_redundant_subgraphs(self, subgraphs, df, rel_tol=1e-2):
        filtered_subgraphs = []

        for msg_nodes, hub_nodes, rels, edges in subgraphs:
            redundant = False

            qf = df[df['com_id'].isin(msg_nodes)]
            mean = qf['ind_pred'].mean()
            preds = sum(list(qf['ind_pred']))

            if (preds == 0 and mean == 0) or (preds == len(qf) and mean == 1):
                redundant = True

            if not redundant:
                filtered_subgraphs.append((msg_nodes, hub_nodes, rels, edges))
        return filtered_subgraphs

    def _single_node_subgraphs(self, subgraphs, df, max_size=69000):
        single_node_subgraphs = []

        msgs = set()
        for msg_nodes, hub_nodes, rels, edges in subgraphs:
            msgs.update(msg_nodes)

        qf = df[~df['com_id'].isin(msgs)]
        msg_nodes = set(qf['com_id'])

        if len(msg_nodes) <= max_size:
            single_node_subgraphs.append((msg_nodes, set(), set(), 0))
        else:
            num_splits = len(msg_nodes) / max_size
            rem = len(msg_nodes) % max_size
            num_splits = int(num_splits) if rem == 0 else int(num_splits) + 1
            single_node_lists = np.array_split(list(msg_nodes), num_splits)
            for msgs in single_node_lists:
                single_node_subgraphs.append((set(msgs), set(), set(), 0))

        return single_node_subgraphs

    def _group(self, target_id, df, relations, debug=False):
        subnetwork = set({target_id})
        frontier, direct_connections = set({target_id}), set()
        relations_present = set()
        edges = 0
        tier = 1

        while len(frontier) > 0:
            com_id = frontier.pop()
            connections, r, e = self._direct_connections(com_id, df, relations)
            unexplored = [c for c in connections if c not in subnetwork]

            # switch to iteration method if subnetwork is too large.
            if len(connections) >= self.size_threshold:
                return self._iterate(target_id, df, relations, debug)

            # update sets.
            subnetwork.update(unexplored)
            direct_connections.update(unexplored)
            relations_present.update(r)
            edges += e

            # tier is exhausted, move to next level.
            if len(frontier) == 0 and len(direct_connections) > 0:
                id_df = df[df['com_id'].isin(list(direct_connections))]
                id_list = id_df['com_id'].values
                frontier = direct_connections.copy()
                direct_connections.clear()

                if debug:
                    print('\nTier ' + str(tier))
                    print(id_list, len(id_list))
                tier += 1

        return subnetwork, relations_present, edges

    def _iterate(self, com_id, df, relations, debug=False):
        """Finds all comments directly and indirectly connected to com_id.
        com_id: identifier of target comment.
        df: comments dataframe.
        relations: list of relations as tuples.
        debug: boolean to print extra information.
        Returns set of comment ids in subnetwork, and relations used."""
        g_ids = [r[2] for r in relations]
        com_df = df[df['com_id'] == com_id]

        first_pass = True
        converged = False
        passes = 0

        headers = list(df)
        h = {h: i + 1 for i, h in enumerate(headers)}

        # init group values as sets in a dict.
        g_dict, g_cnt = {}, {}
        for rel, g, g_id in relations:
            g_vals = com_df[g_id].values
            vals = g_vals[0] if len(g_vals) > 0 else []
            g_dict[g_id] = set(vals)
            g_cnt[g_id] = 0

        total_cc = set()
        edges = 0

        while first_pass or not converged:
            passes += 1
            cc = set({com_id})

            if debug:
                print('pass ' + str(passes))

            for r in df.itertuples():
                connected = False

                # method
                for g_id in g_ids:
                    r_vals = r[h[g_id]]

                    if any(r_val in g_dict[g_id] for r_val in r_vals):
                        connected = True
                        cc.add(r[h['com_id']])
                        if r[h['com_id']] not in total_cc\
                                and r[h['com_id']] != com_id:
                            g_cnt[g_id] += 1
                        break

                # method
                if connected is True:
                    for g_id in g_ids:
                        r_vals = r[h[g_id]]
                        [g_dict[g_id].add(r_val) for r_val in r_vals]

            if first_pass:
                first_pass = False
                total_cc = cc
            else:
                if len(cc) != len(total_cc):
                    total_cc = cc
                else:
                    converged = True
        rels = set([r for r, g, gid in relations if g_cnt[gid] > 0])
        for g_id, ids in g_dict.items():
            edges += len(ids)
        return total_cc, rels, edges

    def _print_subgraphs_size(self, subgraphs, logger=None):
        tot_m, tot_h, tot_e = 0, 0, 0

        for ids, hubs, rels, edges in subgraphs:
            tot_m += len(ids)
            tot_h += len(hubs)
            tot_e += edges

        t = (len(subgraphs), tot_m, tot_h, tot_e)
        if logger:
            logger.info('subgraphs: %d, msgs: %d, hubs: %d, edges: %d' % t)

    def _partition_large_components(self, ccs, g, max_edges=40000):
        new_ccs = []

        for cc in ccs:
            hub_nodes = {x for x in cc if '_' in str(x)}
            num_edges = sum([g.degree(x) for x in hub_nodes])

            if num_edges >= max_edges:
                new_cc = set()
                new_edges = 0

                for hub_node in hub_nodes:
                    hub_edges = g.degree(hub_node)
                    neighbors = set(g.neighbors(hub_node))

                    if hub_edges + new_edges <= max_edges:  # keep adding
                        new_cc.add(hub_node)
                        new_cc.update(neighbors)
                        new_edges += hub_edges

                    elif hub_edges + new_edges > max_edges:  # new is full
                        if new_edges == 0:  # nothing in new
                            new_cc = {hub_node}
                            new_cc.update(neighbors)
                            new_edges = hub_edges

                        else:  # flush, then start anew
                            new_ccs.append(new_cc)
                            new_cc = {hub_node}
                            new_cc.update(neighbors)
                            new_edges = hub_edges

                if len(new_cc) > 0:  # take care of leftovers
                    new_ccs.append(new_cc)
            else:
                new_ccs.append(cc)

        return new_ccs

    def _process_components(self, ccs, g):
        subgraphs = []

        for cc in ccs:
            rels, edges = set(), 0
            msg_nodes = {x for x in cc if '_' not in str(x)}
            hub_nodes = {x for x in cc if '_' in str(x)}

            for hub_node in hub_nodes:
                rels.add(hub_node.split('_')[0])
                edges += g.degree(hub_node)

            subgraphs.append((msg_nodes, hub_nodes, rels, edges))

        return subgraphs

    def _remove_single_edge_hubs(self, subgraphs, g):
        new_subgraphs = []

        for msg_nodes, hub_nodes, rels, edges in subgraphs:
            all_nodes = msg_nodes.union(hub_nodes)
            new_hub_nodes = hub_nodes.copy()
            sg = g.subgraph(all_nodes)

            for n in sg:
                if '_' in str(n):
                    hub_deg = sg.degree(n)

                    if hub_deg == 1:
                        new_hub_nodes.remove(n)
                        edges -= 1

            new_subgraphs.append((msg_nodes, new_hub_nodes, rels, edges))
        return new_subgraphs

    def _validate_subgraphs(self, subgraphs):
        id_list = []

        for ids, hubs, rels, edges in subgraphs:
            id_list.extend(list(ids))

        for v in Counter(id_list).values():
            assert v == 1

    def _get_num_edges(self, cc, g):
        sg = g.subgraph(cc)
        hub_nodes = {x for x in cc if '_' in str(x)}
        num_edges = sum([sg.degree(x) for x in hub_nodes])
        return num_edges
