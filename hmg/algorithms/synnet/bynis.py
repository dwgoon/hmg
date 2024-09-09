"""
BYNIS
- (BY)te is encoded in the sum of (N)ode (I)Ds of a (S)ynthetic edge.
- This method synthesizes the edges of network according to the message.
- To mimic real-world networks, it uses a reference degree distribution.
"""

import random
import struct
import collections
 
import numpy as np
import networkx as nx
from networkx.generators.random_graphs import powerlaw_cluster_graph
import pandas as pd
from tqdm import tqdm

from hmg.utils import get_bitwidth, get_bytewidth
from hmg.algorithms.base import Base
from hmg.logging import write_log


# Byte-width to unsigned format in struct standard package
bw2fmt = {1: "B", 2: "H", 4: "I", 8: "Q"}

class BYNIS(Base):    
    def __init__(self, engine, *args, **kwargs):
        super().__init__(engine, *args, **kwargs)
            
    def estimate_num_nodes(self, n_bytes):
        return int(np.ceil(10**np.round(np.log10(n_bytes))))
        
    def encode(self,
               msg_bytes,
               pw=None,
               n_extra_edges=None,
               policy=None,
               g_ref=None,
               directed=False,
               max_try_rename=100):
        """Encode the message bytes into the node IDs of a synthetic edge.
        """        
        
        disable_tqdm = True if self._verbose == 0 else False
        
        stats = {}
        
        if not pw:
            pw = 1            
        np.random.seed(pw)      
        
        n_bytes = len(msg_bytes)

        # Get byte string of n_bytes    
        n_bytewidth = get_bytewidth(n_bytes)
        bs_n_bytewidth = struct.pack("B", n_bytewidth)
        bs_n_bytes = struct.pack(bw2fmt[n_bytewidth], n_bytes)
        
        if isinstance(msg_bytes, bytes):
            msg_bytes = bs_n_bytewidth + bs_n_bytes + msg_bytes
            data_origin = np.frombuffer(msg_bytes, dtype=np.uint8)
        elif isinstance(msg_bytes, np.ndarray):
            arr_n_bytewidth = np.frombuffer(bs_n_bytewidth, dtype=np.uint8)
            arr_n_bytes = np.frombuffer(bs_n_bytes, dtype=np.uint8)            
            data_origin = np.concatenate([arr_n_bytewidth,
                                          arr_n_bytes,
                                          msg_bytes])
        
        n_bytes = len(data_origin)  # Update num. bytes
        n_nodes = self.estimate_num_nodes(n_bytes)
        bias = max([int(2**np.ceil(np.log2(n_nodes))), 256])
        data_adjusted = data_origin.astype(np.uint16) + bias
        
        # if g_ref:
        #     if not isinstance(g_ref, nx.Graph):
        #         raise TypeError("g_ref should be nx.Graph object.")
        # else:
        if not g_ref:
            n_edges_per_node = int(n_bytes/n_nodes) + 1
            p_add_tri = 0.1
            g_ref = powerlaw_cluster_graph(n_nodes,
                                           n_edges_per_node,
                                           p_add_tri)
            
        degree_ref = np.array([v for k, v in g_ref.degree])
        degree_ref[::-1].sort()
        
        num_use_degree = np.zeros(degree_ref.size, dtype=np.uint32)                
        
        g = self.engine.create_graph(directed=directed)
        
        cur_num = 0
        num_try_rename = 0
        list_edges_stego = []
        node_ids = set()
        node_ids_minor = set()
        node_ids_major = set()


        desc = "Encode message bytes in edge list"
        with tqdm(total=n_bytes, desc=desc, disable=disable_tqdm) as pbar:
            for i, d in enumerate(data_adjusted):
                if degree_ref[cur_num] <= num_use_degree[cur_num]:
                    cur_num += 1
                
                node_a = cur_num
                node_b = d - cur_num
                edge = (node_a, node_b)            
                j = 1
                while g.has_edge(*edge):
                    num_try_rename += 1
                    node_b = node_b + 256*j
                    if node_b < 0:
                        err_msg = "Generating a node ID failed " \
                                  "(Negative ID: %d)"%(node_b)
                        raise ValueError(err_msg)
                    edge = (node_a, node_b)                    
                    j += 1                    
                    if j > max_try_rename:
                        raise RuntimeError("Failed to create target node...")
                        
                    
                # end of while
                
                # if j > 1:
                #     node_ids.add(node_b)
                
                g.add_edge(*edge)
                node_ids_major.add(node_a)
                node_ids_minor.add(node_b)
                node_ids.add(edge[0])
                node_ids.add(edge[1])
                list_edges_stego.append(edge)
                num_use_degree[cur_num] += 1
                pbar.update(1)         
            # end of for
            
            list_edges_ref = list(g_ref.edges)
            max_node_id = max(node_ids)
            if n_extra_edges:
                
                if policy is None:
                    policy = (0,)
                elif isinstance(policy, int):
                    policy = (policy,)
                elif not isinstance(policy, collections.abc.Sequence):
                    raise TypeError("policy should be int or sequence type.")
                    
                
                for i in policy:
                    if i < 0 or i > 6:
                        raise ValueError("policy number should be within [0, 6].")
                    
                n_nodes = g.num_nodes()
                if 0 in policy:
                    for i in range(n_extra_edges):
                        edge = np.random.randint(0, n_nodes, size=2)                
                        
                        while g.has_edge(*edge):                    
                            edge = np.random.randint(0, n_nodes, size=2)
                        # end of while
                       
                        g.add_edge(*edge)
                        list_edges_stego.append(edge)
                        pbar.update(1)
                    # end of for                    
                else:
                    for i in range(n_extra_edges):     
                        
                        while True:           
                            if 1 in policy:
                                if cur_num < num_use_degree.size:
                                    node_a = cur_num
                                    node_b = random.sample(node_ids, 1)[0]
                                    edge = (node_a, node_b)                        
                                    if not g.has_edge(*edge):
                                        num_use_degree[cur_num] += 1
                                        if degree_ref[cur_num] <= num_use_degree[cur_num]:
                                            cur_num += 1
                                        break
                                    # end of if
                                # end of if
                                
                            if 2 in policy:
                                edge = random.sample(list_edges_ref, 1)[0]
                                if not g.has_edge(*edge):
                                    break
                     
                            if 3 in policy:
                                node_a = random.sample(node_ids, 1)[0]
                                node_b = random.sample(node_ids, 1)[0]
                                edge = (node_a, node_b)
                                if not g.has_edge(*edge):
                                    break
                                
                            if 4 in policy:
                                node_a = random.sample(node_ids, 1)[0]
                                node_b = random.sample(node_ids_major, 1)[0]
                                edge = (node_a, node_b)
                                if not g.has_edge(*edge):
                                    break
                                
                            if 5 in policy:
                                node_a = random.sample(node_ids, 1)[0]
                                node_b = random.sample(node_ids_minor, 1)[0]
                                edge = (node_a, node_b)
                                if not g.has_edge(*edge):
                                    break                        
                            
                            if 6 in policy:
                                node_a, node_b = np.random.randint(0, n_nodes + 1, 2)                        
                                edge = (node_a, node_b)
                                if not g.has_edge(*edge):
                                    break                        
                        # end of while
                       
                        g.add_edge(*edge)
                        list_edges_stego.append(edge)
                        pbar.update(1)
                    # end of for
                # end of if-else
        # end of with
        
        cnet_num_nodes = g.num_nodes()
        cnet_num_edges = g.num_edges() 
         
        fstr_logging_net_nums = "Num. %s in the Synthetic Stego Network: %d"
        if self._verbose > 0:
            write_log(fstr_logging_net_nums%("Nodes", cnet_num_nodes))
            write_log(fstr_logging_net_nums%("Edges", cnet_num_edges))
                       
        
        stats["cnet_num_nodes"] = cnet_num_nodes
        stats["cnet_num_edges"] = cnet_num_edges
        stats["msg_bytewidth"] = n_bytewidth
        stats["msg_bytes"] = n_bytewidth
        stats["encoded_msg_size"] = len(msg_bytes)        
        stats["num_try_rename"] = num_try_rename
        
        df_out = pd.DataFrame(list_edges_stego)
        np.random.seed(pw)  # Seed using password.
        index_rand = np.arange(df_out.shape[0])
        np.random.shuffle(index_rand)
        df_stego = df_out.iloc[index_rand, :].reset_index(drop=True)
        
        return df_stego, g, stats
        
    def decode(self,
               df_edges_stego,
               pw=None,
               directed=False):
        """Decode the message bytes from the stego edge list.
        """      
        
        disable_tqdm = True if self._verbose == 0 else False
        
        stats = {}
        
        if not pw:
            pw = 1
            
        np.random.seed(pw)  # Seed using password.
        index_rand = np.arange(df_edges_stego.shape[0])
        np.random.shuffle(index_rand)
        index_ori = np.zeros_like(index_rand)
        index_ori[index_rand] = np.arange(df_edges_stego.shape[0])
        df_edges_stego = df_edges_stego.iloc[index_ori].reset_index(drop=True)
        
        n_bytewidth = sum(df_edges_stego.iloc[0, :]) % 256
        arr_n_bytes = np.zeros(n_bytewidth, np.uint8)
        for i in range(0, n_bytewidth):
            row = df_edges_stego.iloc[1+i, :]
            arr_n_bytes[i] = sum(row[:2]) % 256
            
        bs_n_bytes = arr_n_bytes.tobytes()
        n_bytes = struct.unpack(bw2fmt[n_bytewidth], bs_n_bytes)[0]
            
        data_rec = np.zeros(n_bytes, np.uint8)
        
        g = self.engine.create_graph(directed=directed)
        desc = "Decode Message Bytes"
        with tqdm(total=n_bytes, desc=desc, disable=disable_tqdm) as pbar:
            i_beg = 1+n_bytewidth
            i_end = i_beg + n_bytes
            df_edges_msg = df_edges_stego[i_beg:i_end].reset_index(drop=True)
            
            for i, row in df_edges_msg.iterrows():
                g.add_edge(row[0], row[1])
                v = sum(row[:2]) % 256
                data_rec[i] = v
                pbar.update(1)
        
        stats["snet_num_nodes"] = g.num_nodes()
        stats["snet_num_edges"] = g.num_edges()
        stats["decoded_msg_size"] = len(data_rec)
        return data_rec, stats