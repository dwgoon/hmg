"""
BIND
- (BI)t is encoded in (N)ode (D)egree in edge list.
- This method encodes message bits according to the parity of node degree.
- In other words, the two degrees of an edge encodes two bits, respectively.
"""

import gc
 
import numpy as np
import pandas as pd
import bitstring
from tqdm import tqdm

from hmg.algorithms.base import Base
from hmg.logging import write_log
from hmg.utils import get_bitwidth


class BIND(Base):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initialize()

    def initialize(self):
        self._ind_edge_ee = None
        self._ind_edge_eo = None
        self._ind_edge_oe = None
        self._ind_edge_oo = None

        self._ind_bits_ee = None
        self._ind_bits_eo = None
        self._ind_bits_oe = None
        self._ind_bits_oo = None

        self._estimated_max_bits = None
        self._is_max_bits_estimated = False
        
        gc.collect()


    def estimate_max_bits(self, g, df_edges_cover):           
        get_degree = lambda x: g.degree(x)
        deg_a = df_edges_cover.iloc[:, 0].apply(get_degree)
        deg_b = df_edges_cover.iloc[:, 1].apply(get_degree)
        
        self._ind_edge_ee = df_edges_cover[(deg_a%2 == 0) & (deg_b%2 == 0)].index.to_numpy()
        self._ind_edge_eo = df_edges_cover[(deg_a%2 == 0) & (deg_b%2 == 1)].index.to_numpy()
        self._ind_edge_oe = df_edges_cover[(deg_a%2 == 1) & (deg_b%2 == 0)].index.to_numpy()  
        self._ind_edge_oo = df_edges_cover[(deg_a%2 == 1) & (deg_b%2 == 1)].index.to_numpy()

        self._estimated_max_bits = 8 * min((self._ind_edge_ee.size,
                                            self._ind_edge_eo.size,
                                            self._ind_edge_oe.size,
                                            self._ind_edge_oo.size))

        if self._verbose > 0:
            write_log("Estimated Max. Message Bits: %d"%(self._estimated_max_bits))
            
        self._is_max_bits_estimated = True

        return self._estimated_max_bits

    def find_bit_patterns(self, secret_bits, include_len_bits=False):
        
        if include_len_bits:
            msg_bits = secret_bits
            n_bitwidth = get_bitwidth(len(msg_bits))
            n_bytes_msg = int(len(msg_bits) / 8)
            
            msg_len_bits = bitstring.pack("uint:%d"%(n_bitwidth), n_bytes_msg)
            
            secret_bits = msg_len_bits + msg_bits  
        
        # Find two-bit patterns in the message.
        err_msg = "The number of {et}-type edges is not enough " \
                  "for encoding {et}-type bits."

        arr_two_bits = np.array(secret_bits, dtype=np.uint8).reshape(-1, 2)

        self._ind_bits_ee = np.where((arr_two_bits == [0, 0]).all(axis=1))[0]
        if self._ind_bits_ee.size > self._ind_edge_ee.size:
            raise RuntimeError(err_msg.format(et="EE"))

        self._ind_bits_eo = np.where((arr_two_bits == [0, 1]).all(axis=1))[0]
        if self._ind_bits_eo.size > self._ind_edge_eo.size:
            raise RuntimeError(err_msg.format(et="EO"))

        self._ind_bits_oe = np.where((arr_two_bits == [1, 0]).all(axis=1))[0]
        if self._ind_bits_oe.size > self._ind_edge_oe.size:
            raise RuntimeError(err_msg.format(et="OE"))

        self._ind_bits_oo = np.where((arr_two_bits == [1, 1]).all(axis=1))[0]
        if self._ind_bits_oo.size > self._ind_edge_oo.size:
            raise RuntimeError(err_msg.format(et="OO"))
            
        return (self._ind_bits_ee,
                self._ind_bits_eo,
                self._ind_bits_oe,
                self._ind_bits_oo)


    def ensure_stego_edges(self, g, df_edges_cover, secret_bits, pw=None):
        """This method is intended to be overridden by subclasses
           to ensure that the number of stego edges to encode the secret bits is guaranteed.
        """
        self.find_bit_patterns(secret_bits)

    def get_actual_bit_size(self, msg_bits):    
        # Calculate the bit-width considering the number of df_edges_cover
        # len_list_edges = len(df_edges_cover)

        n_bitwidth = get_bitwidth(len(msg_bits))
        n_bytes_msg = int(len(msg_bits) / 8)
        
        msg_len_bits = bitstring.pack("uint:%d"%(n_bitwidth), n_bytes_msg)
        
        secret_bits = msg_len_bits + msg_bits  # bits is the bitstream to be hidden.
        return len(secret_bits)

    def encode(self, g, df_edges_cover, msg_bits, pw=None):
        """Encode the message bits according to the parity of node degree.
        """
        disable_tqdm = True if self._verbose == 0 else False
        
        stats = {}
        
        len_list_edges = len(df_edges_cover)
        
        cnet_num_nodes = g.num_nodes()
        cnet_num_edges = g.num_edges()     
        
        if self._verbose > 0:
            write_log("Num. Nodes: %d"%(cnet_num_nodes))
            write_log("Num. Edges: %d"%(cnet_num_edges))

        if not self._is_max_bits_estimated:
            self.estimate_max_bits(g, df_edges_cover)
        
        if self._verbose > 0:
            write_log("Num. Edges (EE): %d"%(len(self._ind_edge_ee)))
            write_log("Num. Edges (EO): %d"%(len(self._ind_edge_eo)))
            write_log("Num. Edges (OE): %d"%(len(self._ind_edge_oe)))
            write_log("Num. Edges (OO): %d"%(len(self._ind_edge_oo)))
        
        # Calculate the bit-width considering the number of df_edges_cover
        n_bitwidth = get_bitwidth(len_list_edges)
        n_bytes_msg = int(len(msg_bits) / 8)
        
        msg_len_bits = bitstring.pack("uint:%d"%(n_bitwidth), n_bytes_msg)
        
        secret_bits = msg_len_bits + msg_bits  # bits is the bitstream to be hidden.
        n_bits = len(secret_bits)

        g, df_edges_cover = self.ensure_stego_edges(g, df_edges_cover, secret_bits, pw)

        n_edges_stego = int(n_bits / 2)
        ind_edge_stego = np.zeros(n_edges_stego, dtype=np.int64)
        desc = "Encode message bits in edge list"
        with tqdm(total=n_bits, desc=desc, disable=disable_tqdm) as pbar:

            if self._verbose > 0:
                write_log("Num. Two Bit Patterns (EE): %d"%(len(self._ind_bits_ee)))
                write_log("Num. Two Bit Patterns (EO): %d"%(len(self._ind_bits_eo)))
                write_log("Num. Two Bit Patterns (OE): %d"%(len(self._ind_bits_oe)))
                write_log("Num. Two Bit Patterns (OO): %d"%(len(self._ind_bits_oo)))

            # Randomize the order of the four-type edges.
            if pw is not None:
                np.random.seed(pw)
                np.random.shuffle(self._ind_edge_ee)
                np.random.shuffle(self._ind_edge_eo)
                np.random.shuffle(self._ind_edge_oe)
                np.random.shuffle(self._ind_edge_oo)
                                   
            # Map the two-bit patterns to the edges.
            ind_edge_stego[self._ind_bits_ee] = self._ind_edge_ee[:self._ind_bits_ee.size]
            pbar.update(2 * self._ind_bits_ee.size)

            ind_edge_stego[self._ind_bits_eo] = self._ind_edge_eo[:self._ind_bits_eo.size]
            pbar.update(2 * self._ind_bits_eo.size)
            
            ind_edge_stego[self._ind_bits_oe] = self._ind_edge_oe[:self._ind_bits_oe.size]
            pbar.update(2 * self._ind_bits_oe.size)
            
            ind_edge_stego[self._ind_bits_oo] = self._ind_edge_oo[:self._ind_bits_oo.size]
            pbar.update(2 * self._ind_bits_oo.size)
    
            df_out = pd.concat([df_edges_cover.iloc[ind_edge_stego, :],
                                df_edges_cover.iloc[self._ind_edge_ee[self._ind_bits_ee.size:], :],
                                df_edges_cover.iloc[self._ind_edge_eo[self._ind_bits_eo.size:], :],
                                df_edges_cover.iloc[self._ind_edge_oe[self._ind_bits_oe.size:], :],
                                df_edges_cover.iloc[self._ind_edge_oo[self._ind_bits_oo.size:], :]])

        # Randomize the order of df_edges_cover in the list.
        if pw is not None:            
            np.random.seed(pw)
            ind_rand = np.arange(df_out.shape[0])
            np.random.shuffle(ind_rand)
            df_out = df_out.iloc[ind_rand, :].reset_index(drop=True)
                    
        stats["cnet_num_nodes"] = cnet_num_nodes
        stats["cnet_num_edges"] = cnet_num_edges
        stats["cel_num_edges"] = df_edges_cover.shape[0]
                
        stats["cel_num_edges_ee"] = len(self._ind_edge_ee)
        stats["cel_num_edges_eo"] = len(self._ind_edge_eo)
        stats["cel_num_edges_oe"] = len(self._ind_edge_oe)
        stats["cel_num_edges_oo"] = len(self._ind_edge_oo)
        
        stats["cel_num_edges_encoded"] = len(ind_edge_stego)

        stats["encoded_msg_size"] = len(msg_bits) / 8  # in bytes
        return df_out, stats
    
    
    def get_stego_edges(self, g, df_edges_stego, pw=None):
        
        if pw is not None:
            np.random.seed(pw)  # Seed using password.
            ind_rand = np.arange(df_edges_stego.shape[0])
            np.random.shuffle(ind_rand)
            ind_ori = np.zeros_like(ind_rand)
            ind_ori[ind_rand] = np.arange(df_edges_stego.shape[0])
            df_edges_stego = df_edges_stego.iloc[ind_ori].reset_index(drop=True)
        # end of if

        n_edges = g.num_edges()
        n_bitwidth = get_bitwidth(n_edges)
        i_end_bitwidth = n_bitwidth // 2


        # Get only the number of message bytes.
        get_bits = lambda x: "%d%d"%(g.degree(x[0])%2, g.degree(x[1])%2)
        two_bits = df_edges_stego.iloc[:i_end_bitwidth].apply(get_bits, axis=1)
        str_bits = ''.join(two_bits)
        n_bytes_msg = bitstring.BitArray(bin=str_bits).uint
        
        # A single edge encodes two bits, thus n_bits_msg is divided by two.
        n_bits_msg = 8 * n_bytes_msg        
        i_end_msg = i_end_bitwidth + (n_bits_msg // 2)  
        
        return df_edges_stego.iloc[:i_end_msg, :]

    def decode(self, g, df_edges_stego, pw=None):
        """Decode the message bits from the edge list.
        """
        stats = {}
        
        snet_num_nodes = g.num_nodes()
        snet_num_edges = g.num_edges()

        n_edges = snet_num_edges
        n_bitwidth = get_bitwidth(n_edges)
        
        if pw is not None:
            np.random.seed(pw)  # Seed using password.
            ind_rand = np.arange(df_edges_stego.shape[0])
            np.random.shuffle(ind_rand)
            ind_ori = np.zeros_like(ind_rand)
            ind_ori[ind_rand] = np.arange(df_edges_stego.shape[0])
            df_edges_stego = df_edges_stego.iloc[ind_ori].reset_index(drop=True)

        n_bytes_msg = None
        n_bits_msg = None
      
        get_bits = lambda x: "%d%d"%(g.degree(x[0])%2, g.degree(x[1])%2)
        two_bits = df_edges_stego.apply(get_bits, axis=1)

        i_end_bitwidth = n_bitwidth // 2
        
        str_bits = ''.join(two_bits[:i_end_bitwidth])
        n_bytes_msg = bitstring.BitArray(bin=str_bits).uint
        n_bits_msg = 8 * n_bytes_msg
        
        i_end_msg = i_end_bitwidth + n_bits_msg // 2
        str_bits = ''.join(two_bits[i_end_bitwidth:i_end_msg])
        msg_bits = bitstring.BitArray(bin=str_bits)
        
        stats["snet_num_nodes"] = snet_num_nodes
        stats["snet_num_edges"] = snet_num_edges
        stats["sel_num_edges"] = df_edges_stego.shape[0]
        stats["estimated_max_msg_size"] = 8 * self._estimated_max_bits
        stats["decoded_msg_size"] = 8 * len(msg_bits)

        return msg_bits, stats


