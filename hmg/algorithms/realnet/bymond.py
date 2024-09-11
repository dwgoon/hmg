"""
BYMOND
- (BY)te is encoded in (MO)dulo of the sum of (N)ode (D)egrees of an edge.
- This method encodes a single byte into the modulo of 
  the summation of node degrees of an edge in a real-world network.
"""


import gc
import struct

import numpy as np
import pandas as pd
from tqdm import tqdm

from hmg.utils import get_bytewidth
from hmg.algorithms.base import Base
from hmg.logging import write_log


# Byte-width to unsigned format in struct standard package
bw2fmt = {1: "B", 2: "H", 4: "I", 8: "Q"}

class BYMOND(Base):
    def __init__(self, engine, *args, **kwargs):
        super().__init__(engine, *args, **kwargs)
        self.initialize()

    def initialize(self):
        self._arr_cnt_deg = np.zeros(256, dtype=np.uint64)
        self._indices_edge = []
        self._estimated_max_bytes = None
        self._is_max_bytes_estimated = False
        
        gc.collect()

    def estimate_max_bytes(self, g, df_edges_cover):
        get_degree = lambda x: g.degree(x)
        deg_a = df_edges_cover.iloc[:, 0].apply(get_degree)
        deg_b = df_edges_cover.iloc[:, 1].apply(get_degree)
        
        min_cnt_deg = df_edges_cover.shape[0]  # Total number of edges in the list 
        for i in range(256):
            cond = (deg_a + deg_b)%256 == i  # Condition
            index = df_edges_cover[cond].index
            self._indices_edge.append(index)            
            self._arr_cnt_deg[i] = index.size

            if index.size < min_cnt_deg:
                min_cnt_deg = index.size

        self._estimated_max_bytes = 256 * min_cnt_deg
        if self._verbose > 0:
            write_log("Estimated Max. Message Bytes: %d"%(self._estimated_max_bytes))
       
        self._is_max_bytes_estimated = True
       
        return self._estimated_max_bytes


    def encode(self, g, df_edges_cover, msg_bytes, pw=None):
        """Encode the message bytes in the node IDs of an edge.
        """       
        
        disable_tqdm = True if self._verbose == 0 else False
        
        stats = {}        
        
        cnet_num_nodes = g.num_nodes()
        cnet_num_edges = g.num_edges() 
                
        if self._verbose > 0:
            fstr_logging_net_nums = "Num. %s in the Cover Network: %d"       
            write_log(fstr_logging_net_nums%("Nodes", cnet_num_nodes))
            write_log(fstr_logging_net_nums%("Edges", cnet_num_edges))

        if not self._is_max_bytes_estimated:
            self.estimate_max_bytes(g, df_edges_cover)
        
        for i in range(256):
             stats["cel_num_edges_%03d"%(i)] = len(self._indices_edge[i])
             if self._verbose > 0:
                 write_log("Num. Edges (%d): %d"%(i, len(self._indices_edge[i])))
        

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
        else:
            raise TypeError("Unknown message type: %s"%(type(msg_bytes)))
        
        n_bytes = len(data_origin)  # Update the number of bytes   
        n_edges_stego = n_bytes
        ind_edge_stego = np.zeros(n_edges_stego, dtype=np.uint64)
        desc = "Encode message bytes in edge list"
        with tqdm(total=n_bytes, desc=desc, disable=disable_tqdm) as pbar:
            i_edges = np.zeros(256, dtype=np.uint64)
            i_edge_stego = 0
            for i, d in enumerate(data_origin):                
                ind_edge_stego[i_edge_stego] = self._indices_edge[d][i_edges[d]]    
                i_edges[d] += 1
                i_edge_stego += 1
                pbar.update(1)
        # end of with
        
        
        # Arrange the sub-dataframes
        list_sdf = [df_edges_cover.iloc[ind_edge_stego, :]]
        
        for i in range(256):
            #print(df_edges_cover.iloc[self._indices_edge[i][i_edges[i]:], :])
            list_sdf.append(df_edges_cover.loc[self._indices_edge[i][i_edges[i]:], :])
        
        df_out = pd.concat(list_sdf)
                        
        # Randomize the order of df_edges_cover in the list.
        if not pw:
            pw = 1            
            
        np.random.seed(pw)  # Seed using password.
        ind_rand = np.arange(df_out.shape[0])
        np.random.shuffle(ind_rand)
        df_out = df_out.iloc[ind_rand, :].reset_index(drop=True)
       
        stats["cnet_num_nodes"] = cnet_num_nodes
        stats["cnet_num_edges"] = cnet_num_edges
        stats["cel_num_edges"] = df_edges_cover.shape[0]
        stats["cel_num_edges_encoded"] = len(ind_edge_stego)
        stats["estimated_max_msg_size"] = self._estimated_max_bytes
        stats["encoded_msg_size"] = len(msg_bytes)        
        return df_out, stats
    
    def decode(self, g, df_edges_stego, pw=None):
        """Decode the message bits from the edge list.
        """
        stats = {}
        
        snet_num_nodes = g.num_nodes()
        snet_num_edges = g.num_edges()

        if not pw:
            pw = 1
        
        np.random.seed(pw)  # Seed using password.
        ind_rand = np.arange(df_edges_stego.shape[0])
        np.random.shuffle(ind_rand)
        ind_ori = np.zeros_like(ind_rand)
        ind_ori[ind_rand] = np.arange(df_edges_stego.shape[0])
        df_edges_stego = df_edges_stego.iloc[ind_ori].reset_index(drop=True)

        get_degree = lambda x: g.degree(x)
        deg_a = df_edges_stego.iloc[:, 0].apply(get_degree)
        deg_b = df_edges_stego.iloc[:, 1].apply(get_degree)
        
        mod_sum_deg = (deg_a + deg_b) % 256

        n_bytewidth = mod_sum_deg.iloc[0]
        arr_n_bytes = np.zeros(n_bytewidth, np.uint8)

        for i in range(0, n_bytewidth):
            arr_n_bytes[i] = mod_sum_deg[i + 1]
            
        bs_n_bytes = arr_n_bytes.tobytes()
        n_bytes = struct.unpack(bw2fmt[n_bytewidth], bs_n_bytes)[0]
        data_rec = np.zeros(n_bytes, np.uint8)
        
        i_beg_msg = 1 + n_bytewidth
        i_end_msg = i_beg_msg + n_bytes
        data_rec = mod_sum_deg[i_beg_msg:i_end_msg].to_numpy(np.uint8)
        
        stats["snet_num_nodes"] = snet_num_nodes
        stats["snet_num_edges"] = snet_num_edges
        stats["sel_num_edges"] = df_edges_stego.shape[0]        
        stats["decoded_msg_size"] = len(data_rec)
        
        return data_rec, stats
