import os
import time

import numpy as np
import networkx as nx
import pandas as pd
import bitstring

from hmg.engine import GraphEngine
from hmg.msg import generate_bits
from hmg.logging import use_logging, write_log, finish_logging
from hmg.utils import get_bitwidth

def _estimate_max_bits(g, df_edges_cover):

    get_degree = lambda x: g.degree(x)
    deg_a = df_edges_cover.iloc[:, 0].apply(get_degree)
    deg_b = df_edges_cover.iloc[:, 1].apply(get_degree)
    
    index_edge_ee = df_edges_cover[(deg_a%2 == 0) & (deg_b%2 == 0)].index
    index_edge_eo = df_edges_cover[(deg_a%2 == 0) & (deg_b%2 == 1)].index
    index_edge_oe = df_edges_cover[(deg_a%2 == 1) & (deg_b%2 == 0)].index  
    index_edge_oo = df_edges_cover[(deg_a%2 == 1) & (deg_b%2 == 1)].index

    estimated_max_bits = 8 * min((index_edge_ee.size,
                                  index_edge_eo.size,
                                  index_edge_oe.size,
                                  index_edge_oo.size))

    write_log("Estimated Max. Message Bits: %d"%(estimated_max_bits))

    return estimated_max_bits, [index_edge_ee, index_edge_eo, index_edge_oe, index_edge_oo]

def _encode(g, df_edges_cover, msg_bits, index_edge):
    """Encode the message bits according to the parity of node degree.
       This is a part of the actual implementation.
    """
    
    len_list_edges = len(df_edges_cover)
    
    cnet_num_nodes = g.num_nodes()
    cnet_num_edges = g.num_edges()     
    
    write_log("Num. Nodes: %d"%(cnet_num_nodes))
    write_log("Num. Edges: %d"%(cnet_num_edges))

    index_edge_ee = index_edge[0]
    index_edge_eo = index_edge[1]
    index_edge_oe = index_edge[2]
    index_edge_oo = index_edge[3]
    
    # write_log("Num. Edges (EE): %d"%(len(index_edge_ee)))
    # write_log("Num. Edges (EO): %d"%(len(index_edge_eo)))
    # write_log("Num. Edges (OE): %d"%(len(index_edge_oe)))
    # write_log("Num. Edges (OO): %d"%(len(index_edge_oo)))
    
    # Calculate the bit-width considering the number of df_edges_cover
    n_bitwidth = get_bitwidth(len_list_edges)
    n_bytes_msg = int(len(msg_bits) / 8)
    
    msg_len_bits = bitstring.pack("uint:%d"%(n_bitwidth), n_bytes_msg)
    bits = msg_len_bits + msg_bits
    arr_two_bits = np.array(bits, dtype=np.uint8).reshape(-1, 2)
    
    err_msg = "The number of {et}-type edges is not enough "\
              "to encode {et}-type bytes."
    index_bits_ee = np.where((arr_two_bits == [0, 0]).all(axis=1))[0]
    if index_bits_ee.size > index_edge_ee.size:
        print(index_bits_ee.size, index_edge_ee.size)
        raise RuntimeError(err_msg.format(et="EE"))
    
    index_bits_eo = np.where((arr_two_bits == [0, 1]).all(axis=1))[0]
    if index_bits_eo.size > index_edge_eo.size:
        raise RuntimeError(err_msg.format(et="EO"))
    
    index_bits_oe = np.where((arr_two_bits == [1, 0]).all(axis=1))[0]
    if index_bits_oe.size > index_edge_oe.size:
        raise RuntimeError(err_msg.format(et="OE"))
    
    index_bits_oo = np.where((arr_two_bits == [1, 1]).all(axis=1))[0]
    if index_bits_oo.size > index_edge_oo.size:
        raise RuntimeError(err_msg.format(et="OO"))


if __name__ == "__main__":
    use_logging("bind-rand-ba")

    ge = GraphEngine("networkx")
        
    pw = 1  # Password is used for seeding
    
    # Create the algorithm object
    # arr_ratio_ae = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
    
    
    arr_msg_bytes = 2 ** np.arange(7, 11)
    arr_bpe = np.array([1.0, 1.25, 1.5, 1.75, 2.0])
    
    num_trials = 2  
    list_stats = []
    
    
    for n_msg_bytes in arr_msg_bytes:
        t_beg_total = time.perf_counter()
        stats = {}
        
        msg_bits = generate_bits(n_msg_bytes)
        write_log("Message Size (Bits): %d"%(len(msg_bits)))
        
        for bpe in arr_bpe:
            num_success = 0
            
            for j in range(num_trials):
                
                n_edges = int(n_msg_bytes / bpe)
                n_nodes = n_edges // 2 + 1
                g_cover = nx.barabasi_albert_graph(n_nodes, 1)
                df_cover = nx.to_pandas_edgelist(g_cover)
                max_bits, index_edge = _estimate_max_bits(g_cover, df_cover)     
                
                # Hide the message.
                try:
                    t_beg = time.perf_counter()
                    _encode(ge.create_graph(g_cover),
                            df_cover,
                            msg_bits,
                            index_edge)
                    t_end = time.perf_counter()
                    stats["duration_encode"] = t_end - t_beg
                    write_log("Duration for Encoding: %f sec."%(stats["duration_encode"]))
                    num_success += 1
                except Exception as err:
                    write_log(err)
                    write_log("Failed to encode...")
                
                write_log(f"[Current success rate] {num_success / num_trials}")
            # end of for
            
            
            # Record statistics.
            stats["bpe"] = bpe
            stats["num_trials"] = num_trials
            stats["num_success"] = num_success
            stats["ratio_success"] = num_success / num_trials
            list_stats.append(stats.copy())
        # end of for
    # end of for

    finish_logging()
    df_stats = pd.DataFrame(list_stats)
    df_stats.to_csv("results_bind_rand-ba_increasing_payload.csv", index=False)
