# License: BSD-3-Clause

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.utils import check_symmetric
from snf.compute import _find_dominate_set

try:
    import torch
    deepmodule_installed = True
except ImportError:
    deepmodule_installed = False
    deepmodule_error = "Module 'deep' needs to be installed. See https://imml.readthedocs.io/stable/main/installation.html#optional-dependencies"

TorchDatasetBase = torch.utils.data.Dataset if deepmodule_installed else object


class IntegrAODataset(TorchDatasetBase):
    r"""
    This class provides a `torch.utils.data.Dataset` implementation for handling multi-modal datasets with `IntegrAO`.

    Parameters
    ----------
    Xs : list of array-likes objects
        - Xs length: n_mods
        - Xs[i] shape: (n_samples, n_features_i)

        A list of different modalities.
    neighbor_size : int, default=None
        Number of neighbors to use. If None, it will use N/6).
    networks : list of array-like of shape (n_samples_i, n_samples_i)
        Modal-specific graphs.

    Returns
    -------
    Xs_idx : list of array-likes objects
        - Xs length: n_mods
        - Xs[i] shape: (1, n_features_i)

        A list of different modalities for one sample.
    edge_index : list of array-likes objects
        List with edges.
    indexes :
        All indexes.
    idx : int
        Index.

    Example
    --------
    >>> import numpy as np
    >>> import torch
    >>> from imml.cluster import IntegrAO
    >>> from lightning import Trainer
    >>> from torch.utils.data import DataLoader
    >>> from imml.load import IntegrAODataset
    >>> Xs = [torch.from_numpy(np.random.default_rng(42).random((20, 10))) for i in range(3)]
    >>> estimator = IntegrAO(Xs=Xs, random_state=42)
    >>> train_data = IntegrAODataset(Xs=Xs, neighbor_size=estimator.neighbor_size, networks=estimator.fused_networks_)
    """

    def __init__(self, Xs, networks : list, neighbor_size : int = None):
        if not deepmodule_installed:
            raise ImportError(deepmodule_error)

        if not isinstance(Xs, list):
            raise ValueError(f"Invalid Xs. It must be a list. A {type(Xs)} was passed.")
        if len(Xs) < 2:
            raise ValueError(f"Invalid Xs. It must have at least two modalities. Got {len(Xs)} modalities.")
        if any(len(X) == 0 for X in Xs):
            raise ValueError("Invalid Xs. All elements must have at least one sample.")
        if len(set(len(X) for X in Xs)) > 1:
            raise ValueError("Invalid Xs. All elements must have the same number of samples.")
        if not isinstance(neighbor_size, int):
            raise ValueError(f"Invalid neighbor_size. It must be an int. A {type(neighbor_size)} was passed.")
        if neighbor_size < 1:
            raise ValueError(f"Invalid neighbor_size. It must be a positive number. {neighbor_size} was passed.")
        if not isinstance(networks, list):
            raise ValueError(f"Invalid networks. It must be a list of array-likes objects objects. A {type(networks)} was passed.")

        if not isinstance(Xs[0], pd.DataFrame):
            Xs = [pd.DataFrame(X) for X in Xs]

        self.Xs = []
        self.edge_index = []
        self.indexes = []
        for X, network in zip(Xs, networks):
            idxs = X.dropna(axis=0).index
            idxs = torch.from_numpy(idxs.values).long()
            X = torch.from_numpy(X.values).type(torch.float32)
            self.Xs.append(X)
            self.indexes.append(idxs)
            k = min(int(neighbor_size), network.shape[0])

            network = _find_dominate_set(network, K=k)
            network = check_symmetric(network, raise_warning=False)
            network[network > 0.0] = 1.0
            G = nx.from_numpy_array(network)

            adj = nx.to_scipy_sparse_array(G).tocoo()
            row = torch.from_numpy(adj.row.astype(np.int64)).to(torch.long)
            col = torch.from_numpy(adj.col.astype(np.int64)).to(torch.long)
            edge_index = torch.stack([row, col], dim=0)
            self.edge_index.append(edge_index)


    def __len__(self):
        return len(self.Xs[0])


    def __getitem__(self, idx):
        Xs = [X[idx] for X in self.Xs]
        indexes = [idx for idx in self.indexes]
        edge_index = self.edge_index
        return Xs, edge_index, indexes, idx
