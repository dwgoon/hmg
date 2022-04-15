import os
import time

from hmg.engine import GraphEngine
from hmg.algorithms.synnet import BYNIS
from hmg.msg import generate_bytes
from hmg.logging import use_logging, write_log, finish_logging

if __name__ == "__main__":
    t_beg = time.perf_counter()

    use_logging("ex-bynis-powerlaw", mode='w')
    ge = GraphEngine("networkx")

    pw = 1

    fpath_stego = "bynis_powerlaw_sg.csv" # Reference degrees

    # Create an algorithm object
    alg = BYNIS(ge)
    
    # Create a random message based on the number of edges.
    n_bytes_msg = 10000
    msg_bytes = generate_bytes(n_bytes_msg)
        
    # Hide the message.
    df_stego, g_stego, stats_encode = alg.encode(msg_bytes,
                                                 pw=pw,
                                                 directed=False)    
    
    # Write the stego network in an edge list.
    fileio = ge.create_fileio()
    fileio.write_edgelist(fpath_stego, df_stego)
    
    # Recover the message.
    g_rec, df_rec = fileio.read_edgelist(fpath_stego, directed=False)
    
    write_log("Num. Nodes in Recovered Network: %d"%(g_rec.num_nodes()))
    write_log("Num. Edges in Recovered Network: %d"%(g_rec.num_edges()))
    
    msg_bytes_recovered, stats_decode = alg.decode(df_stego, pw=pw)
    assert((msg_bytes == msg_bytes_recovered).all())  

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
