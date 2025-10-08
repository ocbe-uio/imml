# License: BSD-3-Clause

from .utils import check_matrix_symmetry
import numpy as np


class MultiplexNet:
    """
    Multiplex network representation
    """

    def __init__(self, adj_matrices: list, node_labels: np.ndarray):
        """ Create MultiplexNet object from list of adjacency matrices

        Args:
            adj_matrices (list): list of adjacency matrices (Numpy.ndarray objects)
            node_labels (Numpy.ndarray): one dimensional array of node/sample labels
        """

        if not all([isinstance(layer, np.ndarray) for layer in adj_matrices]):
            raise TypeError("Incorrect type of objects in 'layers' list")
        if not all([check_matrix_symmetry(layer) for layer in adj_matrices]):
            raise ValueError("Non symmetrical / non square adjacency matrix found!")
        if len(adj_matrices) == 0:
            raise ValueError("Empty list of adjacency matrices")
        if not all([adj_matrices[0].shape == layer.shape for layer in adj_matrices]):
            raise ValueError("Different number of nodes in layers")
        if not all(
                [layer[~np.isnan(layer)].min() >= 0 and layer[~np.isnan(layer)].max() <= 1 for layer in adj_matrices]):
            raise ValueError(
                "Data in layers outside of normalized [0,1] range, use 'prepare' to create corrected input files")

        # adjacency matrices
        self.adj_matrices = adj_matrices

        # number of layers
        self.layers = len(self.adj_matrices)

        # number of nodes
        self.nodes = self.adj_matrices[0].shape[0]

        # node labels
        self.sample_names = node_labels

        # sample indices for available (not missing) samples in every layer
        self.samples = [np.array([i for i in range(a.shape[1]) if not np.all(np.isnan(a[:, i]))])
                        for a in self.adj_matrices]

        # check connections between samples in layers
        self.w = []
        # list of matrices with binary values, each describing if pairs of samples are connected in specific layer
        for i in range(self.layers):
            a = self.adj_matrices[i].copy()
            a[~np.isnan(a)] = 1
            a[np.isnan(a)] = 0
            self.w.append(a)

        self.connections = np.sum(self.w, axis=0)
        # connections[i,j] describes in how many layers nodes i and j are connected

    def get_clustering_quality(self, labels: np.ndarray):
        assert labels.shape[0] == self.nodes
        sim = 0
        for i in range(self.layers):
            adj = self.adj_matrices[i]
            sim_layer = 0
            for cluster in np.unique(labels):
                # sum of similarities for all pairs of samples in cluster
                from itertools import combinations
                sim_layer += np.sum(
                    [adj[x, y] for (x, y) in combinations(list(np.argwhere(labels == cluster).T[0]), 2)])
            # normalize using number of samples available in layer
            sim += (sim_layer / (self.samples[i].shape[0] ** 2))
        return sim
