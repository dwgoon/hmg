"""
WU2019
- This is an implementation of Wu et al. (2019).
- Wu et al.
  Securing graph steganography over social networks via interaction remapping.
  ICAIS 2020, pp. 303–312, Springer
  (https://link.springer.com/chapter/10.1007/978-981-15-8101-4_28)
"""


import numpy as np
from tqdm import tqdm

from hmg.algorithms.base import Base
from hmg.logging import write_log


class WU2019(Base):    
    def __init__(self, engine, *args, **kwargs):
        super().__init__(engine, *args, **kwargs)
            
    def estimate_number_of_nodes(self, n_bits):
        return (1 + np.sqrt(1 + 8 * n_bits)) / 2
    
    def synthesize_edge(self, z, n_nodes):                
        min_j = n_nodes
        for j in range(1, n_nodes):  # 1 <= j <= n-1
            s = np.sum(np.arange(n_nodes - j, n_nodes))
            if s >= z:        
                if j < min_j:
                    min_j = j
        # end of for        
        
        x = min_j    
        y = x + z - np.sum(np.arange(n_nodes - x + 1, n_nodes))
                
        return x, y
        
    def encode(self, msg_bits, pw=None, embed=True):
                
        disable_tqdm = True if self._verbose == 0 else False        
        stats = {}
        
        if not pw:
            pw = 1            
        np.random.seed(pw)      

        g_msg = self.engine.create_graph(directed=False)
        # list_edges_msg = []

        msg_bits = np.array(msg_bits, dtype=np.int8)
        n_bits = len(msg_bits)
        n_nodes = self.estimate_number_of_nodes(n_bits)
        n_nodes = int(n_nodes)
        
        # Generate the message-graph.
        desc = "Generate a Message Graph"
        with tqdm(total=n_bits, desc=desc, disable=disable_tqdm) as pbar:

            for z, bit in enumerate(msg_bits, 1):
                if bit == 0:
                    continue
                
                edge = self.synthesize_edge(z, n_nodes)                
                # list_edges_msg.append(edge)    
                g_msg.add_edge(*edge)
                pbar.update(1)     
        # end of with
        
        # Record some statistics.
        mg_num_nodes = g_msg.num_nodes()
        mg_num_edges = g_msg.num_edges() 
         
        if self._verbose > 0:
            fstr_logging_net_nums = "Num. %s in the Message Graph: %d"
            write_log(fstr_logging_net_nums%("Nodes", mg_num_nodes))
            write_log(fstr_logging_net_nums%("Edges", mg_num_edges))
                       
        
        stats["mg_num_nodes"] = mg_num_nodes
        stats["mg_num_edges"] = mg_num_edges        
        stats["encoded_msg_size"] = len(msg_bits) / 8  # in bytes
        
        # df_msg = pd.DataFrame(list_edges_msg, columns=["Source", "Target"])      
        
        
        if not embed:
            return g_msg, stats
        
        # Embed the message-graph.
        g_stego = self.engine.create_graph(directed=True)
        
        desc = "Embed the Message Graph"
        with tqdm(total=n_nodes, desc=desc, disable=disable_tqdm) as pbar:
            for i in range(1, n_nodes + 1):
                for j in range(1, i):
                    # print(j, i)
                    if g_msg.has_edge(j, i):
                        g_stego.add_edge(i, j)
                    else:
                        g_stego.add_edge(i, n_nodes + 1)                    
                # end of for
                pbar.update(1)
            # end of for
        # end of with        
        
        # Record some statistics.
        sg_num_nodes = g_stego.num_nodes()
        sg_num_edges = g_stego.num_edges() 
        
        if self._verbose > 0:
            fstr_logging_net_nums = "Num. %s in the Stego Graph: %d"
            write_log(fstr_logging_net_nums%("Nodes", sg_num_nodes))
            write_log(fstr_logging_net_nums%("Edges", sg_num_edges))
        
        stats["sg_num_nodes"] = sg_num_nodes
        stats["sg_num_edges"] = sg_num_edges        
        
        return g_msg, g_stego, stats


    def decode(self, g, n_bits, pw=None):
        
        disable_tqdm = True if self._verbose == 0 else False
        stats = {}

        ind_msg = np.arange(1, n_bits + 1)
        n_nodes = self.estimate_number_of_nodes(n_bits)
        n_nodes = int(n_nodes)

        msg_bits = np.zeros(n_bits, dtype=np.int8)

        desc = "Decode Message Bits"
        with tqdm(total=n_bits, desc=desc, disable=disable_tqdm) as pbar:            
            for i, z in enumerate(ind_msg):                
                edge = self.synthesize_edge(z, n_nodes)
                if g.has_edge(*edge):
                    msg_bits[i] = 1
                
                pbar.update(1)   
                
        stats["decoded_msg_size"] = msg_bits.size / 8
        return msg_bits, stats