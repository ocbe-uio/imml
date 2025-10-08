# License: BSD-3-Clause

from abc import ABC, abstractmethod
import numpy as np

from ...._sumo.constants import CLUSTER_METHODS, RUN_DEFAULTS
from ...._sumo.modes.run.utils import svdEM
from ...._sumo.network import MultiplexNet
from ...._sumo.utils import extract_max_value, extract_spectral


class SumoNMFResults:
    """
    Wrapper class for SumoNMF factorization results
    """

    def __init__(self, graph: MultiplexNet, h: np.ndarray, s: np.ndarray, objval: np.ndarray, steps: int,
                 sample_ids: np.ndarray, **kwargs):
        self.graph = graph
        self.h = h
        self.s = s
        self.delta_cost = objval
        self.RE = np.sum(self.delta_cost[-1, :self.graph.layers])  # residual error
        self.sample_ids = sample_ids
        self.steps = steps

        self.labels = None  # cluster labels for every node
        self.connectivity = None  # samples x samples with 1 if pair of samples is in the same cluster, 0 otherwise

        for key, value in kwargs.items():
            setattr(self, key, value)

    def extract_clusters(self, method: str, random_state: int = None, verbose = False):
        """ Extract cluster labels using selected method

        Args:
            method (str): either "max_value" for extraction based on maximum value in each row of h matrix \
                or "spectral" for spectral clustering on h matrix values

        """
        if method not in CLUSTER_METHODS:
            raise ValueError(
                'Incorrect method of cluster extraction - supported methods: {}'.format(CLUSTER_METHODS))
        elif method == "max_value":
            # normalize H column-wise
            means = np.mean(self.h, axis=0)
            sds = np.std(self.h, axis=0)
            h = (self.h - means) / sds
            self.labels = extract_max_value(h)

            if np.unique(self.labels).size != self.h.shape[1]:
                if verbose:
                    print('Number of clusters extracted from H matrix is different then expected (k)!')
        else:
            self.labels = extract_spectral(self.h, random_state=random_state)

        labels = np.zeros(self.h.shape)
        labels[np.arange(self.h.shape[0])[:, None], np.array([self.labels]).T] = 1
        self.connectivity = np.zeros((self.graph.nodes, self.graph.nodes))
        self.connectivity[self.sample_ids, self.sample_ids[:, None]] = labels @ labels.T


class SumoSolver(ABC):
    """
    Defines solver of sumo

    Args:
        | graph (MultiplexNet): network object, containing data about connections between nodes in each layer \
            in form of adjacency matrices
        | nbins (int): number of bins, to distribute samples into
        | bin_size (int): size of bin, if None set to number of samples
    """

    def __init__(self, graph: MultiplexNet, nbins: int, bin_size: int = None, verbose = False):

        #todo fix random seed

        if not isinstance(graph, MultiplexNet):
            raise ValueError("Unrecognized graph object")

        if bin_size is None:
            bin_size = round(graph.sample_names.size * (1 - RUN_DEFAULTS['subsample']))

        if nbins <= 0 or bin_size > graph.nodes:
            # This should never happen due to creation of solver objects in sumo 'run'
            raise ValueError("Incorrect number of bins or bin size")

        self.graph = graph
        self.bin_size = bin_size
        self.nbins = nbins
        self.verbose = verbose

    @abstractmethod
    def factorize(self, sparsity_penalty: float, k: int, max_iter: int, tol: float, calc_cost: int,
                  bin_id: int) -> SumoNMFResults:
        """Run solver specific factorization"""
        raise NotImplementedError("Not implemented")

    def calculate_avg_adjacency(self) -> np.ndarray:
        """ Creates average adjacency matrix """
        avg_adj = np.zeros((self.graph.nodes, self.graph.nodes))
        connections = self.graph.connections
        connections[connections == 0] = np.nan

        for a in self.graph.adj_matrices:
            avg_adj = np.nansum(np.dstack((a, avg_adj)), 2)
        avg_adj = avg_adj / self.graph.connections

        # impute NaN values in average adjacency with SVD-EM algorithm
        if np.sum(np.isnan(avg_adj)) > 0:
            avg_adj = svdEM(avg_adj, verbose=self.verbose)

        return avg_adj

    def create_sample_bins(self, random_state) -> list:
        """  Separate samples randomly into bins of set size, while making sure that each sample is allocated
        in at least one bin.

        Returns: list of arrays containing indices of samples allocated to the bin

        """
        if any([x is None for x in [self.graph, self.nbins, self.bin_size]]):
            raise ValueError("Solver parameters not set!")
        sample_ids = list(range(self.graph.nodes))
        sample_ids = np.random.default_rng(random_state).permutation(sample_ids).tolist()
        bins = [sample_ids[i::self.nbins] for i in range(self.nbins)]
        for i in range(len(bins)):
            ms = self.bin_size - len(bins[i])
            bins[i] = np.array(sorted(bins[i] + list(
                np.random.default_rng(random_state + i).choice(list(set(sample_ids) - set(bins[i])),
                                                           size=ms, replace=False))))
        return bins
