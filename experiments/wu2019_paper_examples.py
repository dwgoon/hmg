"""
This is a reproduction of the examples in the following paper:

  Wu et al.
  "New graph-theoretic approach to social steganography."
  Electronic Imaging, 2019.5 (2019): 539-1.
  (https://library.imaging.org/ei/articles/31/5/art00013)
"""        

import numpy as np

from hmg.algorithms.synnet import WU2019
from hmg.engine import GraphEngine
from hmg.msg import generate_bits
        

if __name__ == "__main__":

    # [The example of Fig. 1]
    # Create the message of Fig. 1 in the paper. 
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
    
    
    # [The example of Fig. 2]
    # Create the message of Fig. 2 in the paper. 
    msg_bits = np.array([1, 0, 1, 0, 0, 1, 0, 1, 0, 0,
                          0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0])
        
    g_msg, g_stego, _ = alg.encode(msg_bits, embed=True, op=2, extra_target_edges=0)
    msg_bits_rec, _ = alg.decode(g_stego, msg_bits.size)
        
    # Calculate BER.    
    len_msg = np.min([len(msg_bits), len(msg_bits_rec)])
    arr_msg = np.array(msg_bits[:len_msg])
    arr_msg_rec = np.array(msg_bits_rec[:len_msg])
    ber = ((arr_msg != arr_msg_rec).sum()/len_msg) 
    print("\nBER:", ber)
    
    
    # [An example of Modified OP 1]
    # Create a random message. 
    msg_bits = generate_bits(20)
    g_msg, g_stego, _ = alg.encode(msg_bits, embed=True, op=1, extra_target_edges=10)
    msg_bits_rec, _ = alg.decode(g_stego, len(msg_bits))
    
    # Calculate BER.    
    len_msg = np.min([len(msg_bits), len(msg_bits_rec)])
    arr_msg = np.array(msg_bits[:len_msg])
    arr_msg_rec = np.array(msg_bits_rec[:len_msg])
    ber = ((arr_msg != arr_msg_rec).sum()/len_msg) 
    print("\nBER:", ber)
        
    # [An example of Modified OP 2]
    g_msg, g_stego, _ = alg.encode(msg_bits, embed=True, op=2, extra_target_edges=10)
    msg_bits_rec, _ = alg.decode(g_stego, len(msg_bits))
    
    # Calculate BER.    
    len_msg = np.min([len(msg_bits), len(msg_bits_rec)])
    arr_msg = np.array(msg_bits[:len_msg])
    arr_msg_rec = np.array(msg_bits_rec[:len_msg])
    ber = ((arr_msg != arr_msg_rec).sum()/len_msg) 
    print("\nBER:", ber)
        
    
    # Visualize the stego graph.
    # import networkx as nx
    # import matplotlib.pyplot as plt
    # plt.figure()
    # nx.draw_networkx(g_stego.graph)
    