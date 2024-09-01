import os
import time

import numpy as np
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
    
    write_log("Num. Edges (EE): %d"%(len(index_edge_ee)))
    write_log("Num. Edges (EO): %d"%(len(index_edge_eo)))
    write_log("Num. Edges (OE): %d"%(len(index_edge_oe)))
    write_log("Num. Edges (OO): %d"%(len(index_edge_oo)))
    
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
    use_logging("ex-bind-ogb", mode='w')

    ge = GraphEngine("networkx")
    fileio = ge.create_fileio()

    droot = "../data/ogb/"
    list_datasets = [
        "ogbl_ddi",
        "ogbn_arxiv",        
        "ogbl_collab",
        "ogbl_wikikg2",
        "ogbl_ppa",
        "ogbl_citation2",
        "ogbn_proteins",
        "ogbn_products"
    ]
    
    for i, dataset in enumerate(list_datasets):
        fpath_cover = f"../data/ogb/{dataset}/raw/edge.csv"
        if not os.path.isfile(fpath_cover):
            raise FileNotFoundError("No dataset file: %s"%(fpath_cover))
        else:
            write_log(f"[{i+1}] {dataset} dataset file exists...")
    # end of for
        
    pw = 1  # Password is used for seeding
    
    # Create the algorithm object
    # arr_ratio_ae = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
    arr_bpe = np.array([1.0, 1.25, 1.5, 1.75, 2.0])
    
    num_trials = 100    
    list_stats = []
    for dataset in list_datasets:
        t_beg_total = time.perf_counter()
        stats = {}
        stats["dataset"] = dataset.replace('_', '-')
        write_log(f"[Dataset] {dataset}")
        fpath_cover = f"../data/ogb/{dataset}/raw/edge.csv"
        fpath_stego = f"../data/ogb/{dataset}/raw/edge_sg.csv"
    
        # Read the dataset.
        t_beg = time.perf_counter()        
        g_cover, df_cover = fileio.read_ogb(fpath_cover, directed=True)        
        t_end = time.perf_counter()
        stats["duration_read"] = t_end - t_beg
        
        write_log("Duration for Reading: %f sec."%(stats["duration_read"]))
    
        # Create a random message based on the size of edge list.
        estimated_max_bits, index_edge = _estimate_max_bits(g_cover, df_cover)        
        for bpe in arr_bpe:
            num_success = 0            
            n_edges = g_cover.number_of_edges()
            n_msg_bits = int(bpe * n_edges)
            n_msg_bytes = 8 * n_msg_bits
            
            for j in range(num_trials):
                write_log(f"[{stats['dataset']}][bpe={bpe}][trial#{j+1}]")

                msg_bits = generate_bits(n_msg_bytes)              
               
                # n_bits_msg = int(ratio_ae * max_bits)
                # msg_bits = generate_bits(n_bits_msg // 8)
                write_log("Message Size (Bits): %d"%(len(msg_bits)))
                # Hide the message.
                try:
                    t_beg = time.perf_counter()
                    _encode(g_cover, df_cover, msg_bits, index_edge)
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
            stats["ratio_ae"] = n_msg_bits / estimated_max_bits
            stats["num_trials"] = num_trials
            stats["num_success"] = num_success
            stats["ratio_success"] = num_success / num_trials
            list_stats.append(stats.copy())
        # end of for
    # end of for

    finish_logging()
    df_stats = pd.DataFrame(list_stats)
    df_stats.to_csv("results_bind_ogb_increasing_payload.csv", index=False)