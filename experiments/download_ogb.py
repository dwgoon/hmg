from ogb.nodeproppred import PygNodePropPredDataset
from ogb.linkproppred import PygLinkPropPredDataset

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
    dataset = PygNodePropPredDataset(name=dataset_name, root=droot)

for dataset_name in list_linkproppred_datasets:
    dataset = PygLinkPropPredDataset(name=dataset_name, root=droot)
