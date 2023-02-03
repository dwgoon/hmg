"""
This is a reproduction of the examples in the following paper:

  Wu et al.
  "Securing graph steganography over social networks 
   via interaction remapping."
  ICAIS 2020, pp. 303–312, Springer
  (https://link.springer.com/chapter/10.1007/978-981-15-8101-4_28)
  
"""        

import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
import bitstring

from hmg.algorithms.synnet import WU2019
from hmg.engine import GraphEngine
from hmg.msg import generate_bits


if __name__ == "__main__":

    # Create a message. 
    msg_bits = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 1])
    
    ge = GraphEngine("networkx")    
    alg = WU2019(ge)
    
    g_msg, _ = alg.encode(msg_bits, embed=False)
    msg_bits_rec, _ = alg.decode(g_msg, msg_bits.size)
    
    # Calculate BER.    
    len_msg = np.min([len(msg_bits), len(msg_bits_rec)])
    arr_msg = np.array(msg_bits[:len_msg])
    arr_msg_rec = np.array(msg_bits_rec[:len_msg])
    ber = ((arr_msg != arr_msg_rec).sum()/len_msg) 
    print("\nBER:", ber)
    
    
    # Create another message. 
    msg_bits = np.array([1, 0, 1, 0, 0, 1, 0, 1, 0, 0,
                         0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0])
        
    g_msg, g_stego, _ = alg.encode(msg_bits, embed=True)
    msg_bits_rec, _ = alg.decode(g_msg, msg_bits.size)
    
    # Calculate BER.    
    len_msg = np.min([len(msg_bits), len(msg_bits_rec)])
    arr_msg = np.array(msg_bits[:len_msg])
    arr_msg_rec = np.array(msg_bits_rec[:len_msg])
    ber = ((arr_msg != arr_msg_rec).sum()/len_msg) 
    print("\nBER:", ber)
    
    # Visualize the stego graph.
    nx.draw_networkx(g_stego.graph)