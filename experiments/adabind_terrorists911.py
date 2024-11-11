import os
from os.path import join as pjoin
import time
import pickle

import numpy as np
import matplotlib.pyplot as plt

from hmg.engine import GraphEngine
from hmg.algorithms.hybrid import AdaBIND
from hmg.msg import generate_bits
from hmg.logging import use_logging, write_log, finish_logging


if __name__ == "__main__":  
       
    use_logging("adabind-terrorists_911")
    
    ge = GraphEngine("networkx")    
    droot = os.path.abspath(".")
    
    fpath_cover = pjoin(droot, "../data/test/terrorists-911", "edges.csv")
    fpath_stego = pjoin(droot, "../data/test/terrorists-911", "edges_sg.csv")
    
    is_directed = True
    
    # Create the data structures from the network data
    fileio = ge.create_fileio()
    g_cover, df_cover = fileio.read_edgelist(fpath_cover, directed=is_directed)
    
    # Create a random message based on the number of edges.
    n_edges = g_cover.num_edges()
    n_bits_msg = 2 * n_edges
    msg_bits = generate_bits(n_bits_msg // 8)    

    
    # Create an algorithm object
    alg = AdaBIND(engine=ge,
                  max_iter=100,
                  extra_target_edges=5,
                  n_samp_edges=20,
                  track_greedy_choices=True,
                  verbose=2)
    
    pw = 1234

    df_stego, stats_encode, g_cover_new, df_cover_new = alg.encode(g_cover, 
                                                                   df_cover, 
                                                                   msg_bits, 
                                                                   pw)
        
    # Write the stego in a network file
    fileio.write_edgelist(fpath_stego, df_stego)
    
    # Recover the message
    g_stego, df_stego = fileio.read_edgelist(fpath_stego, directed=is_directed)
    
    write_log("Num. Nodes in Stego Network: %d"%(g_stego.num_nodes()))
    write_log("Num. Edges in Stego Network: %d"%(g_stego.num_edges()))
    
    msg_bits_recovered, stats_decode = alg.decode(g_stego, df_stego, pw)
    assert(msg_bits == msg_bits_recovered)
    assert(g_cover_new.num_nodes() == g_stego.num_nodes())  
    assert(g_cover_new.num_edges() == g_stego.num_edges())  
    
    df1 = df_cover_new.sort_values(by=["Source", "Target"], ignore_index=True)
    df2 = df_stego.sort_values(by=["Source", "Target"], ignore_index=True)    
    assert((df1 == df2).all().all())
    
    # Show statistics.
    fstat = os.stat(fpath_stego)
    payload_bpe = n_bits_msg / df_stego.shape[0]  # BPE (Bits Per Edge)
    payload_bytes = (n_bits_msg/8) / fstat.st_size
    write_log("BPE (Bits Per Edge): %.3f"%(payload_bpe))
    write_log("Payload in Bytes: %.3f %%"%(payload_bytes))
    
    stats = {}
    
    stats["payload_bpe"] = payload_bpe
    stats["payload_bytes"] = payload_bytes
    
    stats.update(stats_encode)
    stats.update(stats_decode)   
    
    # Get the stego edges.
    df_stego_edges = alg.get_stego_edges(g_stego, df_stego, pw)
    
    
    # Output the results.
    dict_out = {}
    
    dict_out["stego_edges"] = np.array(df_stego_edges.values, dtype=np.int64)
    dict_out["track_added_edges"] = alg.greedy_choices["edge"][1:alg.stop_iter]
    dict_out["track_delta_num_edges"] = alg.greedy_choices["delta_num_edges"][1:alg.stop_iter]
    dict_out["track_num_edges"] = alg.greedy_choices["num_edges"][1:alg.stop_iter]
    dict_out["track_l1_dist"] = alg.greedy_choices["l1_dist"][1:alg.stop_iter]
    
    with open("results_terrorist-911.pkl","wb") as fout:
        pickle.dump(dict_out, fout)
    
    # Plot the history of greedy choices.
    for i, edge in enumerate(alg.greedy_choices["edge"][1:alg.stop_iter]):
        print(f"[Edge selected at #{i}] {edge}")
    
    # plt.plot(alg.greedy_choices["delta_num_edges"][1:alg.stop_iter])
    # plt.plot(alg.greedy_choices["num_edges"][1:alg.stop_iter])
    plt.plot(alg.greedy_choices["l1_dist"][1:alg.stop_iter])
        
