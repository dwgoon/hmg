from ogb.nodeproppred import NodePropPredDataset
from ogb.linkproppred import LinkPropPredDataset

list_nodeproppred_datasets = [
    "ogbn-arxiv",
    "ogbn-proteins",
    "ogbn-products"
]

list_linkproppred_datasets = [
    "ogbl-ddi",
    "ogbl-collab",
    "ogbl-wikikg2",
    "ogbl-ppa",
    "ogbl-citation2",
]

droot = "../data/ogb/"

for dataset_name in list_nodeproppred_datasets:
    dataset = NodePropPredDataset(name=dataset_name, root=droot)

for dataset_name in list_linkproppred_datasets:
    dataset = LinkPropPredDataset(name=dataset_name, root=droot)
