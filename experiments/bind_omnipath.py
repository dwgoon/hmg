import os
import time

from hmg.engine import GraphEngine
from hmg.algorithms.realnet import BIND
from hmg.msg import generate_bits
from hmg.logging import use_logging, write_log, finish_logging


if __name__ == "__main__":
    t_beg = time.perf_counter()

    use_logging("ex-bind-omnipath", mode='w')
    ge = GraphEngine("networkx")    
    
    fpath_cover = "../data/test/omnipath/omnipath.sif"
    fpath_stego = "../data/test/omnipath/omnipath_sg.sif"
    pw = 1234  # Password is used for seeding.    

    # Create an algorithm object
    alg = BIND(ge)
    
    # Create the data structures from the network data
    fileio = ge.create_fileio()
    g_cover, df_cover = fileio.read_sif(fpath_cover, directed=True)
    
    # Create a random message based on the number of edges.
    n_edges = g_cover.num_edges()
    n_bits_msg = int(0.95*n_edges)  # Use 95% of edges
    msg_bits = generate_bits(n_bits_msg // 8)    
    
    # Hide the message
    df_stego, stats_encode = alg.encode(g_cover, df_cover, msg_bits, pw)
    
    # Write the stego in a network file
    fileio.write_sif(fpath_stego, df_stego)
    
    # Recover the message
    g_stego, df_stego = fileio.read_sif(fpath_stego, directed=True)
    
    write_log("Num. Nodes in Stego Network: %d"%(g_stego.num_nodes()))
    write_log("Num. Edges in Stego Network: %d"%(g_stego.num_edges()))
    
    msg_bits_recovered, stats_decode = alg.decode(g_stego, df_stego, pw)
    assert(msg_bits == msg_bits_recovered)
    assert(g_cover.num_nodes() == g_stego.num_nodes())  
    assert(g_cover.num_edges() == g_stego.num_edges())  
    
    df1 = df_cover.sort_values(by=["Source", "Target"], ignore_index=True)
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

    finish_logging()
    