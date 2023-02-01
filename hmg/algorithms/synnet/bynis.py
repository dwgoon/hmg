"""
BYNIS
- (BY)te is encoded in the sum of (N)ode (I)Ds of a (S)ynthetic edge.
- This method synthesizes the edges of network according to the message.
- To mimic real-world networks, it uses a reference degree distribution.
"""

import struct

import numpy as np
from networkx.generators.random_graphs import powerlaw_cluster_graph
import pandas as pd
from tqdm import tqdm

from hmg.utils import get_bytewidth
from hmg.algorithms.base import Base
from hmg.logging import write_log


# Byte-width to unsigned format in struct standard package
bw2fmt = {1: "B", 2: "H", 4: "I", 8: "Q"}

class BYNIS(Base):    
    def __init__(self, engine, *args, **kwargs):
        super().__init__(engine, *args, **kwargs)
            
    def estimate_number_of_nodes(self, n_bytes):
        return int(np.ceil(10**np.round(np.log10(n_bytes))))
        
    def encode(self,
               msg_bytes,
               pw=None,
               g_ref=None,
               directed=False,
               max_try_rename=20):
        """Encode the message bytes into the node IDs of a synthetic edge.
        """        
        
        
        
        stats = {}
        
        if pw:
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
        n_nodes = self.estimate_number_of_nodes(n_bytes)
        n_adjusted = max([int(2**np.ceil(np.log2(n_nodes))), 256])
        data_adjusted = data_origin.astype(np.uint16) + n_adjusted
        
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
        desc = "Encode Message Bytes in Edge List"
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
                
                g.add_edge(*edge)
                list_edges_stego.append(edge)
                num_use_degree[cur_num] += 1
                pbar.update(1)
         
        
        cnet_num_nodes = g.num_nodes()
        cnet_num_edges = g.num_edges() 
         
        fstr_logging_net_nums = "Num. %s in the Synthetic Stego Network: %d"
        if verbose > 0:
            write_log(fstr_logging_net_nums%("Nodes", cnet_num_nodes))
            write_log(fstr_logging_net_nums%("Edges", cnet_num_edges))
                       
        
        stats["cnet_num_nodes"] = cnet_num_nodes
        stats["cnet_num_edges"] = cnet_num_edges
        stats["msg_bytewidth"] = n_bytewidth
        stats["msg_bytes"] = n_bytewidth
        stats["encoded_msg_size"] = len(msg_bytes)        
        stats["num_try_rename"] = num_try_rename

        return pd.DataFrame(list_edges_stego), g, stats
        
    def decode(self,
               df_edges_stego,
               pw=None,
               directed=False):
        """Decode the message bytes from the stego edge list.
        """      
        stats = {}
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
        with tqdm(total=n_bytes, desc=desc) as pbar:
            df_edges_msg = df_edges_stego[1+n_bytewidth:].reset_index(drop=True)
            
            for i, row in df_edges_msg.iterrows():
                g.add_edge(row[0], row[1])
                v = sum(row[:2]) % 256
                data_rec[i] = v
                pbar.update(1)
        
        stats["snet_num_nodes"] = g.num_nodes()
        stats["snet_num_edges"] = g.num_edges()
        stats["decoded_msg_size"] = len(data_rec)
        return data_rec, stats