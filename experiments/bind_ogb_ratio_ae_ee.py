import os

import numpy as np
import pandas as pd

from hmg.engine import GraphEngine
from hmg.logging import use_logging, write_log, finish_logging

def get_min_num_edges(g, df_edges_cover):

    get_degree = lambda x: g.degree(x)
    deg_a = df_edges_cover.iloc[:, 0].apply(get_degree)
    deg_b = df_edges_cover.iloc[:, 1].apply(get_degree)
    
    index_edge_ee = df_edges_cover[(deg_a%2 == 0) & (deg_b%2 == 0)].index
    index_edge_eo = df_edges_cover[(deg_a%2 == 0) & (deg_b%2 == 1)].index
    index_edge_oe = df_edges_cover[(deg_a%2 == 1) & (deg_b%2 == 0)].index  
    index_edge_oo = df_edges_cover[(deg_a%2 == 1) & (deg_b%2 == 1)].index

    min_num_edges =  min((index_edge_ee.size,
                          index_edge_eo.size,
                          index_edge_oe.size,
                          index_edge_oo.size))

    write_log("Min. num. edges: %d"%(min_num_edges))

    return min_num_edges, [index_edge_ee, index_edge_eo, index_edge_oe, index_edge_oo]



if __name__ == "__main__":
    use_logging("ex-bind-ogb-min-num-edges", mode='w')

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

    list_stats = []
    for dataset in list_datasets:
        stats = {}
        stats["dataset"] = dataset.replace('_', '-')
        write_log(f"[Dataset] {dataset}")
        fpath_cover = f"../data/ogb/{dataset}/raw/edge.csv"
        fpath_stego = f"../data/ogb/{dataset}/raw/edge_sg.csv"
    
        # Read the dataset.
        g_cover, df_cover = fileio.read_ogb(fpath_cover, directed=True)       
           
        # Create a random message based on the size of edge list.
        min_num_edges, indices_edges = get_min_num_edges(g_cover, df_cover)        
        n_edges = g_cover.num_edges()
                          
        # Record statistics.
        stats["ratio_ae_ee"] = n_edges / (4 * min_num_edges)
        stats["min_num_edges"] = min_num_edges
        stats["num_edges_ee"] = indices_edges[0].size
        stats["num_edges_eo"] = indices_edges[1].size
        stats["num_edges_oe"] = indices_edges[2].size
        stats["num_edges_oo"] = indices_edges[3].size          
        list_stats.append(stats.copy())
    # end of for

    finish_logging()
    df_stats = pd.DataFrame(list_stats)
    df_stats.to_csv("results_bind_ogb_ratio_ae_ee.csv", index=False)