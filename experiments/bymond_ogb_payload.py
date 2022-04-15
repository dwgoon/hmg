import os
import time
import struct
from collections import Counter

import numpy as np
import pandas as pd

from hmg.engine import GraphEngine
from hmg.msg import generate_bytes
from hmg.logging import use_logging, write_log, finish_logging
from hmg.utils import get_bytewidth

# Byte-width to unsigned format in struct standard package
bw2fmt = {1: "B", 2: "H", 4: "I", 8: "Q"}

def _estimate_max_bytes(g, df_edges_cover):
    arr_cnt_deg = np.zeros(256, dtype=np.uint64)
    estimated_max_bytes = None
        
    get_degree = lambda x: g.degree(x)
    deg_a = df_edges_cover.iloc[:, 0].apply(get_degree)
    deg_b = df_edges_cover.iloc[:, 1].apply(get_degree)
    
    min_cnt_deg = df_edges_cover.shape[0]  # Total number of edges in the list 
    for i in range(256):
        cond = (deg_a + deg_b)%256 == i  # Condition
        index = df_edges_cover[cond].index
        arr_cnt_deg[i] = index.size

        if index.size < min_cnt_deg:
            min_cnt_deg = index.size

    estimated_max_bytes = 256 * min_cnt_deg
    write_log("Estimated Max. Message Bytes: %d"%(estimated_max_bytes))
       
    return estimated_max_bytes, arr_cnt_deg

def _encode(g, msg_bytes, arr_cnt_deg):
    """Encode the message bytes in the node IDs of an edge.
    """       
    n_bytes = len(msg_bytes)
    
    # Get byte string of n_bytes    
    n_bytewidth = get_bytewidth(n_bytes)
    bs_n_bytewidth = struct.pack("B", n_bytewidth)
    bs_n_bytes = struct.pack(bw2fmt[n_bytewidth], n_bytes)
    
    if isinstance(msg_bytes, bytes):
        msg_bytes = bs_n_bytewidth + bs_n_bytes + msg_bytes
        data_origin = np.frombuffer(msg_bytes, dtype=np.uint8)
    elif isinstance(msg_bytes, np.ndarray):
        arr_n_bytewidth = np.frombuffer(bs_n_bytewidth, dtype=np.uint8)
        arr_n_bytes = np.frombuffer(bs_n_bytes, dtype=np.uint8)            
        data_origin = np.concatenate([arr_n_bytewidth,
                                      arr_n_bytes,
                                      msg_bytes])
    else:
        raise TypeError("Unknown message type: %s"%(type(msg_bytes)))
    
    
    cnts_bytes = Counter(data_origin)

    for i in range(256):
        if cnts_bytes[i] > arr_cnt_deg[i]:
            err_msg = "[%d] cnt_byte: %d vs cnt_deg: %d"%(i+1,
                                                          cnts_bytes[i],
                                                          arr_cnt_deg[i])
            write_log(err_msg)
            return i

    return -1

if __name__ == "__main__":
    use_logging("ex-bymond-ogb", mode='w')
    
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
            
    arr_ratio_ae = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
    num_trials = 1000
    list_stats = []   
    for dataset in list_datasets:
        np.random.seed(1)
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

        for i in range(256):
            stats["cnt_fail_edge_%d"%(i)] = 0
  
        # Create a random message based on the size of edge list.        
        max_bytes, arr_cnt_deg = _estimate_max_bytes(g_cover, df_cover) 
        for ratio_ae in arr_ratio_ae:
            num_success = 0
            for j in range(num_trials):
                write_log(f"[{stats['dataset']}][ratio_ae={ratio_ae}][trial#{j+1}]")
                n_bytes_msg = int(ratio_ae * max_bytes)
                msg_bytes = generate_bytes(n_bytes_msg)
                write_log("Message Size (Bytes): %d"%(len(msg_bytes)))
                # Hide the message.
                try:
                    t_beg = time.perf_counter()
                    res = _encode(g_cover, msg_bytes, arr_cnt_deg)
                    t_end = time.perf_counter()
                    stats["duration_encode"] = t_end - t_beg
                    write_log("Duration for Encoding: %f sec."%(stats["duration_encode"]))
                    
                    if res >= 0:
                        stats["cnt_fail_edge_%d"%(res)] += 1
                        err_msg = "The number of {et}-type edges is not enough "\
                                  "to encode {et}-type bytes."
                        write_log("Failed to encode...")
                        write_log(err_msg.format(et=res))
                    else:                    
                        num_success += 1

                except Exception as err:
                    write_log(err)
                    write_log("Failed to encode...")
                    continue
                
            # end of for
            stats["ratio_ae"] = ratio_ae
            stats["num_trials"] = num_trials
            stats["num_success"] = num_success
            stats["ratio_success"] = num_success / num_trials
            list_stats.append(stats.copy())
    # end of for

        
    finish_logging()
    df_stats = pd.DataFrame(list_stats)
    df_stats.to_csv("results_bymond_ogb_increasing_payload.csv", index=False)
