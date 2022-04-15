import os
import time

from hmg.engine import GraphEngine
from hmg.algorithms.realnet import BYMOND
from hmg.msg import generate_bytes
from hmg.logging import use_logging, write_log, finish_logging


if __name__ == "__main__":
    t_beg = time.perf_counter()

    use_logging("ex-bymond-omnipath", mode='w')
    ge = GraphEngine("networkx")
    
    pw = 1  # Password is used for seeding.

    # A small subset of string database
    fpath_cover = "../data/test/ddi/edge.csv"
    fpath_stego = "bymond_ddi.csv"
    
    # Create an algorithm object
    alg = BYMOND()
    
    # Create the data structures from the network data.
    fileio = ge.create_fileio()
    g_cover, df_cover = fileio.read_ogb(fpath_cover, directed=False)   
    
    # Create a random message based on the number of edges.    
    n_bytes_max = alg.estimate_max_bytes(g_cover, df_cover)
    n_bytes_msg = int(0.95*n_bytes_max)  # Use 95% of edges
    msg_bytes = generate_bytes(n_bytes_msg)
    
    # Hide the message.
    df_stego, stats_encode = alg.encode(g_cover, df_cover, msg_bytes, pw)
    
    # Write the stego in a network file.
    fileio.write_ogb(fpath_stego, df_stego)
    
    # Recover the message.
    g_stego, df_stego = fileio.read_ogb(fpath_stego, directed=False)
    
    write_log("Num. Nodes in Stego Network: %d"%(g_stego.num_nodes()))
    write_log("Num. Edges in Stego Network: %d"%(g_stego.num_edges()))
    
    msg_bytes_recovered, stats_decode = alg.decode(g_stego, df_stego, pw)
    
    assert((msg_bytes == msg_bytes_recovered).all())
    assert(g_cover.num_nodes() == g_stego.num_nodes())  
    assert(g_cover.num_edges() == g_stego.num_edges())  
    
    df1 = df_cover.sort_values(by=["Source", "Target"], ignore_index=True)
    df2 = df_stego.sort_values(by=["Source", "Target"], ignore_index=True)    
    assert((df1 == df2).all().all())
        
    # Show statistics.
    fstat_stego = os.stat(fpath_stego)
    payload_bpe = (8 * n_bytes_msg) / df_stego.shape[0]  # BPE (Bits Per Edge)
    payload_bytes = n_bytes_msg / fstat_stego.st_size
    write_log("BPE (Bits Per Edge): %.3f"%(payload_bpe))
    write_log("Payload in Bytes: %.3f"%(payload_bytes))
    
    stats = {}
    stats["stego_file_size"] = fstat_stego.st_size
    stats["payload_bpe"] = payload_bpe
    stats["payload_bytes"] = payload_bytes

    stats.update(stats_encode)
    stats.update(stats_decode)

    t_end = time.perf_counter()    
    write_log("Elapsed Time: %f sec."%(t_end - t_beg))
    finish_logging()

    