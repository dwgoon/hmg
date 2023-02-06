"""
WU2019
- This is an implementation of Wu et al. (2019).
- Wu et al.
  Securing graph steganography over social networks via interaction remapping.
  ICAIS 2020, pp. 303–312, Springer
  (https://link.springer.com/chapter/10.1007/978-981-15-8101-4_28)
"""


import numpy as np
from numba import jit, njit
from tqdm import tqdm


from hmg.algorithms.base import Base
from hmg.logging import write_log


@jit('float64(int64)', nopython=True)
def _estimate_number_of_nodes(n_bits):
    return (1 + np.sqrt(1 + 8 * n_bits)) / 2
    

@jit('Tuple((int64, int64))(int64, int64)', nopython=True)
def _synthesize_edge(z, n_nodes):                
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


class WU2019(Base):    
    def __init__(self, engine, *args, **kwargs):
        super().__init__(engine, *args, **kwargs)
            

        
    def encode(self, msg_bits, pw=None, embed=True, op=1, n_extra_edges=None):
                
        disable_tqdm = True if self._verbose == 0 else False        
        stats = {}
        
        if not pw:
            pw = 1            
        np.random.seed(pw)      

        g_msg = self.engine.create_graph(directed=False)
        # list_edges_msg = []

        msg_bits = np.array(msg_bits, dtype=np.int8)
        n_bits = len(msg_bits)
        n_nodes = _estimate_number_of_nodes(n_bits)
        n_nodes = int(np.ceil(n_nodes))
        
        # Generate the message-graph.
        desc = "Generate a Message Graph"
        with tqdm(total=n_bits, desc=desc, disable=disable_tqdm) as pbar:

            for z, bit in enumerate(msg_bits, 1):
                if bit == 0:
                    pbar.update(1)
                    continue
                
                edge = _synthesize_edge(z, n_nodes)   
                # print("[MSG] Add %s"%(str(edge)))
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
        
        if not embed:
            return g_msg, stats
        
        # Embed the message-graph.
        g_stego = self.engine.create_graph(directed=True)
        
        # Modified OP 1 in the paper.
        if op == 1 and n_extra_edges:
            list_extra_edges = list()

        # Modified OP 2 in the paper.
        if op == 2 and n_extra_edges:
            # Split the index of last node (i.e., allow multiple nodes).
            ind_extra = np.arange(n_nodes + 1, n_nodes + n_extra_edges + 2)
            np.random.shuffle(ind_extra)
            i_extra = 0
        # end of if
        
                
        n_progress = n_nodes + n_extra_edges if op == 1 else n_nodes
        desc = "Embed the Message Graph"
        with tqdm(total=n_progress, desc=desc, disable=disable_tqdm) as pbar:
            for i in range(1, n_nodes + 1):
                less_ind_exists = True  # A less index exists.
                for j in range(1, i + 1):                    
                    if g_msg.has_edge(j, i):
                        g_stego.add_edge(i, j)
                        # print("[DIR] Add (%d, %d)"%(i, j))

                        # print("j=%d i=%d exists"%(j, i))
                        less_ind_exists = True                        
                # end of for        
                
                
                if not n_extra_edges:
                    # OP 1 in the paper.
                    g_stego.add_edge(i, n_nodes + 1) 
                    # print("Add i=%d, n+1=%d"%(i, n_nodes + 1))
                else:
                    # Modified OP 1 in the paper.
                    if op == 1:
                        w = np.random.randint(0, 2, size=(n_nodes - i + 1))
                        for j in range(i + 1, n_nodes + 1):                    
                            if w[j - i] == 1:                            
                                # print("[MOP1] Add (%d, %d)"%(i, j))
                                list_extra_edges.append((i, j))  # i_src < i_trg
                    
                    # Modified OP 2 in the paper.      
                    if op == 2 and not less_ind_exists:
                        if i_extra < n_extra_edges:                                          
                            g_stego.add_edge(i, ind_extra[i_extra])
                            i_extra += 1                      
                # end of if-else                                
                                
                if op == 1 and n_extra_edges:
                    # Randomly insert extra edges according to Modified OP 1.
                    np.random.shuffle(list_extra_edges)
                    for i in range(n_extra_edges):
                        edge = list_extra_edges[i]
                        g_stego.add_edge(*edge)
                        pbar.update(1)
                
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
        n_nodes = _estimate_number_of_nodes(n_bits)
        n_nodes = int(np.ceil(n_nodes))

        msg_bits = np.zeros(n_bits, dtype=np.int8)

        desc = "Decode Message Bits"
        with tqdm(total=n_bits, desc=desc, disable=disable_tqdm) as pbar:            
            for i, z in enumerate(ind_msg):                
                x, y = _synthesize_edge(z, n_nodes)                
                if g.has_edge(y, x, directed=True) and x < y:
                    msg_bits[i] = 1
                    # print("[DECODE %d] x=%d, y=%d, msg=%d"%(i, x, y, msg_bits[i]))
                # end of if                  
                pbar.update(1)   
            # end of for
        stats["decoded_msg_size"] = msg_bits.size / 8
        return msg_bits, stats