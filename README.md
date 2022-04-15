# Hiding Secret Messages in Graph Datasets (HMG)

**HMG**(**H**iding Secret **M**essages in **G**raph Datasets) is a Python package that presents a collection of steganographic algorithms for graph datasets.


## Algorithms

- Real-world Networks
    - BIND
    - BYMOND

- Synthetic Networks
    - BYNIS


## Installation

```
python setup.py install
```

- Dependencies

  - [numpy](https://www.numpy.org)
  - [scipy](https://www.scipy.org)
  - [pandas](https://pandas.pydata.org)
  - [networkx](https://networkx.org)
  - [bitstring](https://github.com/scott-griffiths/bitstring)
  - [tqdm](https://github.com/tqdm/tqdm)


## Graph Engine

The default graph engine is based on the functionality of [`networkx`](https://networkx.org).
However, we can also use [`python-igraph`](https://igraph.org/python) instead of `networkx`.

```
from hmg.engine import GraphEngine

ge = GraphEngine('networkx')  # Use networkx for creating GraphEngine object.
ge = GraphEngine('igraph')  # Use python-igraph for creating GraphEngine object.
```


## Experiments

### 1. Basic Experiments

This repository provides some basic experiments for each algorithm in `experiments` directory.

- BIND: `bind_omnipath.py`
- BYMOND: `bymond_ddi.py`
- BYNIS: `bynis_powerlaw.py`


### 2. Experiments for OGB datasets

#### 2.1. Download OGB datasets

To perform the experiments for [OGB](https://ogb.stanford.edu/) datasets,
we need to install the following packages.

 - [PyTorch](https://pytorch.org/)
 - [PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric)

The reason for installing the PyTorch packages is that `ogb` package depends on these packages.
After installing the above packages, install `ogb` package.

```
pip install ogb
```

Now, we can download the datasets using `experiments/download_ogb.py`.
The default download directory is `data/ogb`.

```
cd experiments
python download_ogb.py
```

#### 2.2. Perform Experiments

In `experiments` directory, execute ```python (algorithm)_ogb_payload.py```.
These scripts perform the encoding simulation experiments for all datasets of OGB.

- BIND: `bind_ogb_pyaload.py`
- BYMOND: `bymond_ogb_pyaload.py`


## Citation

