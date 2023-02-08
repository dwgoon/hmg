"""
WU2020
- This is an implementation of Wu et al. (2020).
- Wu et al.
  "Securing Graph Steganography over Social Networks via Interaction Remapping"
  ICAIS 2020, Proceedings, Part III 6 (pp. 303-312), Springer, Singapore
  (https://link.springer.com/chapter/10.1007/978-981-15-8101-4_28)
"""

from collections import defaultdict

import numpy as np
import bitstring

import numpy as np
import pandas as pd
from tqdm import tqdm

from hmg.algorithms.base import Base
from hmg.logging import write_log


class WU2020(Base):    
    def __init__(self, engine, *args, **kwargs):
        super().__init__(engine, *args, **kwargs)
            
    # def estimate_num_nodes(self, n_msg_bytes):
    #     return int(_estimate_num_nodes(8 * n_msg_bytes))
                
    def encode(self,
               msg_bits,
               df_ref=None,
               pw=None):
                
        disable_tqdm = True if self._verbose == 0 else False        
        stats = {}
        
        if not pw:
            pw = 1            
        np.random.seed(pw)      
    
        if df_ref is None:
            pass  # Create random graphs
        else:            
            g_ref = self.engine.create_graph(directed=False)
            for i in range(df_ref.shape[0]):
                x = df_ref.loc[i, "Source"]
                y = df_ref.loc[i, "Target"]
                g_ref.add_edge(x, y)
            # end of for     
        
        g_stego = g_ref.copy()
        
        n_nodes = g_ref.num_nodes()
        n_edges = g_ref.num_edges()  # n_edges corresponds to m in the paper.
        # assert(n_edges == 8)
            
        
        n_bits = len(msg_bits)
        bgs = int(np.log2(n_edges))  # the size of each bit-group
        
        # [TODO] Pad 0 bits to messages if necessary...
        assert(bgs == np.log2(n_edges))
        
        n_bg = n_edges  # the number of bit-groups
        
        # Convert to bit-groups to decimal values.
        dvals = np.zeros(n_bg, dtype=np.uint64)  # Decimal values (D)
        
        # Count each unique decimal value.
        uval_cnts = np.zeros(n_bg + 1, dtype=np.uint64)
        
        # Record the edges corresponding to each unique decimal value.
        uval_edges = defaultdict(list)
        
        n_edges_stego = 0
        for i in range(n_bg):        
            bg = msg_bits[bgs*i:bgs*(i+1)]  # bit-group        
            dval = bg.uint
            dvals[i] = dval
            # print(bg, "->", dval)
            # assert(dval == dvals_ori[i])
            
            uval_cnts[dval] += 1
            uval_edges[dval].append(i)
            
        # end of for
                
        n_edges_stego = int(n_edges + 1 + np.sum(uval_cnts[uval_cnts >= 2]))
                                    
        r = pw % (n_nodes + 1) # The random seed
        np.random.seed(r)
        
        # Data embedding: F operations in the paper.
        arr_edges_stego = np.zeros((n_edges_stego, 3), dtype=np.uint64)
        max_edge_index = df_ref.index.max()
            
        i = 0  # the index of stego edge
        for dval in range(0, n_edges + 1):    
            if dval == n_edges:
                edge = (r, 0)
                max_edge_index += 1
                arr_edges_stego[i, 0] = max_edge_index
                arr_edges_stego[i, 1:] = edge
                i += 1
                continue
            
            cnt = uval_cnts[dval]
            if cnt == 0:
                ind_node = np.random.randint(1, n_nodes + 1)
                edge = (ind_node, 0)  # New edge
                # edge = (dval, 0)  # [OPTIONAL] Remove the randomness.
                g_stego.add_edge(*edge)                

                max_edge_index += 1
                arr_edges_stego[i, 0] = max_edge_index
                arr_edges_stego[i, 1:] = edge
            elif cnt == 1:
                j = uval_edges[dval][0]
                arr_edges_stego[i, :] = df_ref.iloc[j]
                
            elif cnt >= 2:
                edge = (cnt, n_nodes + 1)  # New edge
                g_stego.add_edge(*edge)
                
                max_edge_index += 1
                arr_edges_stego[i, 0] = max_edge_index
                arr_edges_stego[i, 1:] = edge            
                inds = uval_edges[dval] 
                i += 1            
                for j in inds:
                    arr_edges_stego[i, :] = df_ref.iloc[j]
                    i += 1
                # end of for
                continue
            else:
                raise RuntimeError("Invalid count value occurs: %d"%(cnt))
            # end of if-else        
            i += 1
        # end of for
        
        # g_stego = engine.create_graph(directed=False)  # Stego graph
        # for _, x, y in arr_edges_stego:
        #     g_stego.add_edge(x, y)
        
        df_stego = pd.DataFrame(arr_edges_stego,
                                columns=["Index", "Source", "Target"])
        
        return df_stego, g_stego
        
    def decode(self, df_stego, pw=None, g_stego=None): 
        
        disable_tqdm = True if self._verbose == 0 else False
        stats = {}

    
        # Data extraction: G operations in the paper.
                   
        # Find the max value of node index.
        max_node_index = df_stego.iloc[:, 1:].max().max()        
                
        if not g_stego:            
            g_stego = self.engine.create_graph(directed=False)
            for i in range(df_stego.shape[0]):
                x = df_stego.loc[i, "Source"]
                y = df_stego.loc[i, "Target"]
                g_stego.add_edge(x, y)
            # end of for     
        # end of if
           
        g_ref = g_stego.copy()        
        g_ref.del_node(0)
        g_ref.del_node(max_node_index)
        
        n_nodes = g_ref.num_nodes()
        n_edges = g_ref.num_edges()
        
        r = pw % (n_nodes + 1) # The random seed
        np.random.seed(r)
        
        dvals_rec = np.zeros(n_edges + 1, dtype=np.uint64)  # Decimal values (D)

        j = 0
        for i in range(0, n_edges + 1):    
            ind_edge, *edge = df_stego.iloc[j, :]
            
            ind_min = min(edge)
            ind_max = max(edge)
            
            if ind_min == 0:
                pass            
            elif ind_max == max_node_index:            
                for _ in range(ind_min):
                    j += 1
                    ind_edge, *edge = df_stego.iloc[j, :]
                    dvals_rec[ind_edge] = i                
                # end of for
            elif g_ref.has_edge(*edge):
                dvals_rec[ind_edge] = i
            elif i == n_edges:
                pass
            
            j += 1
        # end of for
       
       
        dvals_rec = dvals_rec[1:]
        
        bgs = int(np.log2(n_edges))  # the size of each bit-group    
        msg_bits = np.zeros(bgs* n_edges, dtype=np.int8)  # m * log2(m)
        
        groups = [bitstring.pack('uint%d'%(bgs), dval) for dval in dvals_rec]
        msg_bits = sum(groups)
 
        return msg_bits, stats
        