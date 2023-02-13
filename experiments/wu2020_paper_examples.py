"""
This is a reproduction of the examples in the following paper:

  Wu et al.
  "Securing Graph Steganography over Social Networks via Interaction Remapping"
  ICAIS 2020, Proceedings, Part III 6 (pp. 303-312), Springer, Singapore
  (https://link.springer.com/chapter/10.1007/978-981-15-8101-4_28)
"""        

import numpy as np
import pandas as pd

from hmg.algorithms.synnet import WU2020
from hmg.engine import GraphEngine
from hmg.msg import to_bitarr
        

if __name__ == "__main__":

    # The example in the paper.
    pw = 4  # Password
        
    msg_bits = to_bitarr(np.array([0, 1, 0, 
                                   0, 1, 1, 
                                   1, 1, 0, 
                                   0, 1, 1, 
                                   0, 0, 0, 
                                   1, 0, 0,
                                   0, 0, 1, 
                                   1, 1, 1]))
        
    # The original decimal values in the paper.
    # dvals_ori = np.array([2, 3, 6, 3, 0, 4, 1, 7])
           
    df_ref = pd.DataFrame(
        [[1, 1, 2],
         [2, 1, 3],
         [3, 1, 4],
         [4, 1, 5],
         [5, 2, 3],
         [6, 2, 5],
         [7, 3, 5],
         [8, 4, 5]],
        columns=["Index", "Source", "Target"]
    )
    
    ge = GraphEngine("networkx")

    g_ref = ge.create_graph(directed=False)  # Reference graph    
    for _, (index, src, trg) in df_ref.iterrows():
        g_ref.add_edge(src, trg)
        
        
    alg = WU2020(ge)
    
    df_stego, g_stego, stats_encode = alg.encode(msg_bits, df_ref, pw=pw)    
    msg_bits_rec, stats_decode = alg.decode(df_stego, pw=pw)
    
    
    # Calculate BER.    
    len_msg = np.min([len(msg_bits), len(msg_bits_rec)])
    arr_msg = np.array(msg_bits[:len_msg])
    arr_msg_rec = np.array(msg_bits_rec[:len_msg])
    ber = ((arr_msg != arr_msg_rec).sum()/len_msg) 
    print("\nBER:", ber)
        
    
    # import networkx as nx
    # import matplotlib.pyplot as plt
    # plt.figure()
    # nx.draw_networkx(g_stego.graph)
    
 