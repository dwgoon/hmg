"""
AdaBIND
- Adaptive BIND selforithm.
- This method is the same as BIND selforithm, but it is adaptive to 
  the secret message.
- It synthesizes some edges if the edges of certain types are not enough.

- We can consider the edge synthesis policy to adapt to
  the two-bit patterns of a given message.
  
   1. GW2N: Wire two nodes that make the current edge numbers close to 
            the target edge numbers based on a greedy approach.           
            'GW2N' means a greedy algorithm for wiring two nodes.
   2. C1N: Create one node and connect it to the existing node.
   3. C1E: Create a new edge whose two nodes do not exist in the graph.

  However, policy 2 and 3 can break the existing ID system 
  because they need to create new node IDs.
  
  Currently, only GW2N is implemented in AdaBIND.

"""

import gc
import random
from itertools import combinations

import numpy as np
import pandas as pd
import bitstring
from tqdm import tqdm

from hmg.algorithms.realnet.bind import BIND
from hmg.logging import write_log
from hmg.utils import get_bitwidth


class AdaBIND(BIND):
    def __init__(self, 
                 engine=None,
                 ada_policy="GW2N",
                 max_iter=None,
                 extra_target_edges=None,
                 n_samp_edges=None,
                 track_greedy_choices=False,
                 *args,
                 **kwargs):
        
        super().__init__(engine, *args, **kwargs)
        AdaBIND.initialize(self,
                           ada_policy,
                           max_iter,
                           extra_target_edges,
                           n_samp_edges,
                           track_greedy_choices)
        
        
    def initialize(self,
                   ada_policy,
                   max_iter,
                   extra_target_edges,
                   n_samp_edges,
                   track_greedy_choices,
                   *args,
                   **kwargs):
        
        super().initialize(*args, **kwargs)
        
        if ada_policy is None:
            ada_policy = "GW2N"
        self._ada_policy = ada_policy
            
        if max_iter is None:
            max_iter = 10000
        self._max_iter = max_iter        
        
        if extra_target_edges is None:
            extra_target_edges = 5            
        self._extra_target_edges = extra_target_edges
        
        if n_samp_edges is None:
            n_samp_edges = 50            
        self._n_samp_edges = n_samp_edges
        
        if track_greedy_choices is None:
            track_greedy_choices = False            
        self._track_greedy_choices = track_greedy_choices
        
        
        # i-th iteration at which the algorithm stops to stop adding edges.
        self._stop_iter = 0  
        self._greedy_choices = []

        # Internal variables.
        self._g_new = None
        self._df_edges_cover_new = None
        
        gc.collect()

    @property
    def ada_policy(self):
        return self._ada_policy

    @ada_policy.setter
    def ada_policy(self, val):
        self._ada_policy = val
    
    @property
    def max_iter(self):
        return self._max_iter

    @max_iter.setter
    def max_iter(self, val):
        self._max_iter = val

    @property
    def extra_target_edges(self):
        return self._extra_target_edges

    @extra_target_edges.setter
    def extra_target_edges(self, val):
        self._extra_target_edges = val            
        
    @property
    def n_samp_edges(self):
        return self._n_samp_edges

    @n_samp_edges.setter
    def n_samp_edges(self, val):
        self._n_samp_edges = val

    @property
    def track_greedy_choices(self):
        return self._track_greedy_choices

    @track_greedy_choices.setter
    def track_greedy_choices(self, val):
        self._track_greedy_choices = val
    
    @property
    def stop_iter(self):  # read only
        return self._stop_iter

    @property
    def greedy_choices(self): # read only
        return self._greedy_choices


    def find_bit_patterns(self, secret_bits, include_len_bits=False):
        if include_len_bits:
            msg_bits = secret_bits
            n_bitwidth = get_bitwidth(len(msg_bits))
            n_bytes_msg = int(len(msg_bits) / 8)
            
            msg_len_bits = bitstring.pack("uint:%d"%(n_bitwidth), n_bytes_msg)
            
            secret_bits = msg_len_bits + msg_bits  
            
        # Find two-bit patterns in the message.
        arr_two_bits = np.array(secret_bits, dtype=np.uint8).reshape(-1, 2)

        self._ind_bits_ee = np.where((arr_two_bits == [0, 0]).all(axis=1))[0]
        self._ind_bits_eo = np.where((arr_two_bits == [0, 1]).all(axis=1))[0]
        self._ind_bits_oe = np.where((arr_two_bits == [1, 0]).all(axis=1))[0]
        self._ind_bits_oo = np.where((arr_two_bits == [1, 1]).all(axis=1))[0]

        return (self._ind_bits_ee,
                self._ind_bits_eo,
                self._ind_bits_oe,
                self._ind_bits_oo)
    
    def _find_neighbors(self, nodes, g, df):
        successors = {}
        predecessors = {}

        # src_col = df.columns[0]
        # trg_col = df.columns[1]
        #
        # for node in nodes:
        #     successors[node] = # df.loc[df[src_col] == node, trg_col].tolist()
        #     predecessors[node] = # df.loc[df[trg_col] == node, src_col].tolist()

        # if g.is_directed():
        #     for node in nodes:
        #          successors[node] = list(g.successors(node))
        #          predecessors[node] = list(g.predecessors(node))
        # else:
        #     for node in nodes:
        #          successors[node] = list(g.neighbors(node))
        #          predecessors[node] = list(g.neighbors(node))

        if not g.is_directed():
            raise TypeError("Graph should be created as directed in AdaBIND.")

        for node in nodes:
             successors[node] = list(g.successors(node))
             predecessors[node] = list(g.predecessors(node))

        return successors, predecessors


    def _get_delta_num_edges(self,
                            g_ori,
                            g_new,
                            src,
                            trg,
                            successors,
                            predecessors):
        if g_ori.has_edge(src, trg):
            raise ValueError(f"g_ori should not contain ({src}, {trg})")

        g_new.add_edge(src, trg)

        # 0: EE, 1: EO, 2: OE, 3: OO
        delta_num_edges = np.zeros((4), dtype=np.int64)

        deg_src_ori = g_ori.degree(src)
        deg_trg_ori = g_ori.degree(trg)

        deg_src_new = g_new.degree(src)
        deg_trg_new = g_new.degree(trg)

        parity_src_ori = deg_src_ori % 2
        parity_trg_ori = deg_trg_ori % 2

        parity_src_new = deg_src_new % 2
        parity_trg_new = deg_trg_new % 2

        index_parity_edge_new = 2 * parity_src_new + parity_trg_new

        delta_num_edges[index_parity_edge_new] += 1

        edge_with_parity = [(src, parity_src_ori, parity_src_new),
                            (trg, parity_trg_ori, parity_trg_new)]

        for node, parity_ori, parity_new in edge_with_parity:

            # When node is the source of other edges.
            for neighbor in successors[node]:

                deg_neighbor_ori = g_ori.degree(neighbor)
                deg_neighbor_new = g_new.degree(neighbor)

                parity_neighbor_ori = deg_neighbor_ori % 2
                parity_neighbor_new = deg_neighbor_new % 2

                index_parity_ori = 2 * parity_ori + parity_neighbor_ori
                index_parity_new = 2 * parity_new + parity_neighbor_new

                if index_parity_ori == index_parity_new:
                    raise RuntimeError("index_parity_ori must not be equal to index_parity_new!")

                delta_num_edges[index_parity_ori] -= 1
                delta_num_edges[index_parity_new] += 1
            # end of for

            # When node is the target of other edges.
            for neighbor in predecessors[node]:

                deg_neighbor_ori = g_ori.degree(neighbor)
                deg_neighbor_new = g_new.degree(neighbor)

                parity_neighbor_ori = deg_neighbor_ori % 2
                parity_neighbor_new = deg_neighbor_new % 2

                index_parity_ori = 2 * parity_neighbor_ori + parity_ori
                index_parity_new = 2 * parity_neighbor_new + parity_new

                if index_parity_ori == index_parity_new:
                    raise RuntimeError("index_parity_ori must not be equal to index_parity_new!")

                delta_num_edges[index_parity_ori] -= 1
                delta_num_edges[index_parity_new] += 1
            # end of for

        # end of for

        g_new.del_edge(src, trg)  # To use it repeatedly

        return delta_num_edges
    
    def ensure_stego_edges(self, 
                           g, 
                           df_edges_cover, 
                           secret_bits, 
                           pw=None):

        self.find_bit_patterns(secret_bits)

        n_missing_edges_ee = self._ind_bits_ee.size - self._ind_edge_ee.size
        n_missing_edges_oe = self._ind_bits_oe.size - self._ind_edge_oe.size
        n_missing_edges_eo = self._ind_bits_eo.size - self._ind_edge_eo.size
        n_missing_edges_oo = self._ind_bits_oo.size - self._ind_edge_oo.size
        
        if (n_missing_edges_ee <= 0 and n_missing_edges_oe <= 0 
            and n_missing_edges_eo <= 0 and n_missing_edges_oo <= 0):
            return g, df_edges_cover

        # Start creating new edges.
        if self.ada_policy == "GW2N":
            return self._gw2n(g, df_edges_cover)


    def _gw2n(self, g, df_edges_cover):
        """ Wire two nodes that make the current edge numbers close to
            the target edge numbers based on a greedy approach.
            'GW2N' means a greedy algorithm for wiring two nodes.
        """
        current_num_edges = np.array([self._ind_edge_ee.size,
                                      self._ind_edge_eo.size,
                                      self._ind_edge_oe.size,
                                      self._ind_edge_oo.size],
                                     dtype=np.int64)        

        target_num_edges = np.array([max(self._ind_bits_ee.size, self._ind_edge_ee.size),
                                     max(self._ind_bits_eo.size, self._ind_edge_eo.size),
                                     max(self._ind_bits_oe.size, self._ind_edge_oe.size),
                                     max(self._ind_bits_oo.size, self._ind_edge_oo.size)],
                                    dtype=np.int64)   

        target_num_edges += self.extra_target_edges

        df_edges_t0 = df_edges_cover.copy()
        g_t0 = g.copy()
        g_t1 = g.copy()  # This copy is created to check the change in the number of each type of edge.

        n_iter = self.max_iter
        is_sol_found = False

        new_edges = set()

        # nodes = sorted(g_t0.graph.nodes, key=lambda x: g_t0.degree(x))
        nodes = list(g_t0.graph.nodes)
        successors, predecessors = self._find_neighbors(nodes, g_t0, df_edges_t0)

        n_samp_nodes = min(len(nodes), self._n_samp_edges // 2)        
        if n_samp_nodes % 2 != 0:  # Ensure the number of sampled nodes is even number.
            n_samp_nodes = n_samp_nodes - 1

        min_l1_dist = np.iinfo(np.int64).max
        min_l1_dist_edge = (np.iinfo(np.int64).max, np.iinfo(np.int64).max)
        min_changed_num_edges = current_num_edges
        min_delta_num_edges = np.array([0, 0, 0, 0])

        if self.track_greedy_choices:
            arr_l1_dist = np.zeros(self.max_iter, dtype=np.int64)
            arr_edge = np.zeros((self.max_iter, 2), dtype=np.int64)
            arr_num_edges = np.zeros((self.max_iter, 4), dtype=np.int64)
            arr_delta_num_edges = np.zeros((self.max_iter, 4), dtype=np.int64)

        disable_tqdm = True if self._verbose == 0 else False

        desc = "Search additional edges to encode the secret message"
        with tqdm(total=n_iter, desc=desc, disable=disable_tqdm) as pbar:
            self._stop_iter = 0
            for i in range(n_iter):
                self._stop_iter = i

                # Tracking the greedy chice.
                if self.track_greedy_choices:
                    arr_l1_dist[i] = min_l1_dist
                    arr_edge[i, :] = min_l1_dist_edge
                    arr_num_edges[i, :] = min_changed_num_edges
                    arr_delta_num_edges[i, :] = min_delta_num_edges

                np.random.shuffle(nodes)

                for j in range(0, n_samp_nodes, 2):
                    if j >= n_samp_nodes:
                        break

                    src = nodes[j]
                    trg = nodes[j + 1]

                    if g_t0.is_directed():
                        if g_t0.has_edge(src, trg) or (src, trg) in new_edges:
                            continue
                    else:
                        if g_t0.has_edge(src, trg) or g_t0.has_edge(trg, src):
                            continue

                        if (src, trg) in new_edges or (trg, src) in new_edges:
                            continue

                    delta_num_edges = self._get_delta_num_edges(g_t0,
                                                                g_t1,
                                                                src,
                                                                trg,
                                                                successors,
                                                                predecessors)

                    changed_num_edges = current_num_edges + delta_num_edges
                    l1_dist = np.sum(np.abs(changed_num_edges - target_num_edges))

                    if l1_dist < min_l1_dist:
                        min_l1_dist = l1_dist
                        min_l1_dist_edge = (src, trg)
                        min_changed_num_edges = changed_num_edges
                        min_delta_num_edges = delta_num_edges

                        if self.extra_target_edges != 0 and l1_dist  <= 1:
                            is_sol_found = True
                            break
                        # end of if
                    # end of if
                # end of for (src, trg) in comb_nodes

                # print(f"[Iteration #{i}] min. l1_dist: {min_l1_dist}, edge: {min_l1_dist_edge}")

                if self._verbose > 1:
                   tqdm.write(f"- [Iteration #{i}] min. l1_dist: {min_l1_dist}, edge: {min_l1_dist_edge}")

                if g_t0.has_edge(*min_l1_dist_edge):
                    continue

                # Update the graph and edge list.

                # 0. Update the current_num_edges.
                current_num_edges = min_changed_num_edges

                # 1. Add the new edge to graph data structures.
                new_edges.add(min_l1_dist_edge)
                g_t0.add_edge(*min_l1_dist_edge)
                g_t1.add_edge(*min_l1_dist_edge)

                # 2. Add a row of the new edge to the pd.DataFrame.
                other_cols = [None] * (df_edges_t0.columns.size - 2)
                df_edges_t0.loc[len(df_edges_t0), :] = [*min_l1_dist_edge] + other_cols

                # 3. Update the neighbors: (src) --> (trg).
                src, trg = min_l1_dist_edge
                successors[src].append(trg)
                predecessors[trg].append(src)

                pbar.update(1)

                if is_sol_found:
                    break
            # end of for i in range(n_iter)
        # end of with
        
        self._g_new = g_t0
        self._df_edges_cover_new = df_edges_t0
        self.estimate_max_bits(g_t0, df_edges_t0)

        if self.track_greedy_choices:
            self._greedy_choices = {
                "l1_dist": arr_l1_dist,
                "edge": arr_edge,
                "num_edges": arr_num_edges,
                "delta_num_edges": arr_delta_num_edges,
            }
        
        return self._g_new, self._df_edges_cover_new
        
    
    def encode(self, g, df_edges_cover, msg_bits, pw=None):        
        df_out, stats = super().encode(g, df_edges_cover, msg_bits, pw)
        
        if self._g_new is not None and self._df_edges_cover_new is not None:
            return df_out, stats, self._g_new, self._df_edges_cover_new
            
        return df_out, stats, g, df_edges_cover

        
        
        
