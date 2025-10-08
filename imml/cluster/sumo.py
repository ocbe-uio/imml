# License: BSD-3-Clause

from typing import Union
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.cluster.hierarchy import cophenet, linkage
from scipy.spatial.distance import pdist
import numpy as np

from ..utils import check_Xs
from ._sumo.utils import extract_ncut
from ._sumo.network import MultiplexNet
from ._sumo.modes.run.solvers.unsupervised_sumo import UnsupervisedSumoNMF
from ._sumo.modes.prepare.similarity import feature_to_adjacency


class SUMO(BaseEstimator, ClassifierMixin):
    r"""
    Subtyping Tool for Multi-Omic Data (SUMO). [#sumopaper1]_ [#sumocode]_

    SUMO, originally designed for molecular subtyping in multi-omics datasets, utilizes a state-of-the-art
    nonnegative matrix factorization (NMF) algorithm to identify clusters of samples with similar characteristics.

    The authors strongly suggest removing features and samples with a large fraction of missing values (>10%); log
    transform or a variant stabilizing transform when using count data as input; and standardize each input feature. For
    more information, read the sumo's documentation: https://python-sumo.readthedocs.io/en/latest/index.html.

    Parameters
    ----------
    n_clusters : int, default=8
        The number of clusters to generate. If it is not provided, it will use the default one from the algorithm.
    method : str or list of str, default='euclidean'
        either one method of sample-sample similarity calculation, or list of methods for every modality (available
        methods: ['euclidean', 'cosine', 'pearson', 'spearman']).
    missing : float or list of float, default=[0.1]
        acceptable fraction of available values for assessment of distance/similarity between pairs of samples - either
        one value or list for every modality.
    neighbours : float, default=0.1
        fraction of nearest neighbours to use for sample similarity calculation using Euclidean distance
        similarity.
    alpha : float, default=0.5
        hypherparameter of RBF similarity kernel, for Euclidean distance similarity.
    sparsity : float or list of float, default=[0.1]
        either one value or list of sparsity penalty values for H matrix (sumo will try different values and select
        the best results).
    repetitions : int, default=60
        Number of repetitions.
    cluster_method : str, default="max_value"
        Method of cluster extraction. Options are 'max_value' or'spectral'.
    max_iter : int, default=500
        Maximum number of iterations for factorization.
    tol : float, default=1e-5
        If objective cost function value fluctuation is smaller than this value, stop iterations before
        reaching max_iter.
    subsample : float, default=0.05
        Fraction of samples randomly removed from each run, cannot be greater than 0.5.
    calc_cost : int, default=20
        Number of steps between every calculation of objective cost function.
    h_init : int, default=None
        index of adjacency matrix to use for H matrix initialization (by default using average adjacency).
    rep : int, default=5
        number of times consensus matrix is created for the purpose of assessing clustering quality.
    random_state : int, default=None
        Determines the randomness. Use an int to make the randomness deterministic.
    verbose : bool, default=False
        Verbosity mode.
    n_jobs : int, default=1
        Number of threads to run in parallel.

    Attributes
    ----------
    labels_ : array-like of shape (n_samples,)
        Labels of each point in training data.
    embedding_ : array-like of shape (n_samples, n_clusters)
        The final spectral representation of the data to be used as input for the KMeans clustering step.
    graph_ : MultiplexNet
        Multi-modal graph.
    nmf_ : UnsupervisedSumoNMF
        The nonnegative matrix factorization (NMF) object.
    similarity_ : dict of length n_mods, with mods as keys and an array-like of shape (n_samples,n_samples) as values.
        List of adjacency matrix.
    cophenet_list_ : ndarray of shape (rep,).
        Object created by SUMO
    pac_list_ : ndarray of shape (rep,).
        Object created by SUMO

    References
    ----------
    .. [#sumopaper1] Sienkiewicz, K., Chen, J., Chatrath, A., Lawson, J. T., Sheffield, N. C., Zhang, L., & Ratan, A.
                     (2022). Detecting molecular subtypes from multi-omics datasets using SUMO. In Cell Reports Methods
                     (Vol. 2, Issue 1, p. 100152). Elsevier BV. https://doi.org/10.1016/j.crmeth.2021.100152
    .. [#sumocode] https://github.com/ratan-lab/sumo/tree/master

    Example
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from imml.cluster import SUMO
    >>> Xs = [pd.DataFrame(np.random.default_rng(42).random((20, 10))) for i in range(3)]
    >>> estimator = SUMO(n_clusters = 2)
    >>> labels = estimator.fit_predict(Xs)
    """

    def __init__(self, n_clusters: int = 8, method: Union[str, list] = None, missing: list = None,
                 neighbours: float = 0.1, alpha: float = 0.5, sparsity: list = None, repetitions: int = 60,
                 cluster_method: str = "max_value", max_iter: int = 500, tol: float = 1e-5, subsample: float = 0.05,
                 calc_cost: int = 20, h_init: int = None, rep: int = 5, random_state: int = None,
                 verbose: bool = False, n_jobs: int = 1):

        if not isinstance(n_clusters, int):
            raise ValueError(f"Invalid n_clusters. It must be an int. A {type(n_clusters)} was passed.")
        if n_clusters < 2:
            raise ValueError(f"Invalid n_clusters. It must be an greater than 1. {n_clusters} was passed.")
        if random_state is None:
            random_state = int(np.random.default_rng().integers(10000))
        if method is None:
            method = ['euclidean']
        method_option = ['euclidean', 'cosine', 'pearson', 'spearman']
        if isinstance(method, str):
            if method not in method_option:
                msg = f"Invalid method. Expected one of: {method_option}."
                raise ValueError(msg)
        elif isinstance(method, list):
            for method_i in method:
                if method_i not in method_option:
                    msg = f"Invalid method. Expected one of: {method_option}."
                    raise ValueError(msg)

        if missing is None:
            missing = [0.1]
        if sparsity is None:
            sparsity = [0.1]

        if repetitions < 1:
            msg = "Incorrect repetitions. It must be repetitions > 0."
            raise ValueError(msg)
        if subsample > 0.5 or subsample < 0:
            msg = "Incorrect subsample. It must be 0 < subsample < 0.5."
            raise ValueError(msg)
        if rep < 1:
            msg = "Incorrect rep. It must be rep > 1."
            raise ValueError(msg)
        if n_jobs < 1:
            msg = "Incorrect n_jobs. It must be n_jobs > 0."
            raise ValueError(msg)


        self.method = method
        self.missing = missing
        self.neighbours = neighbours
        self.alpha = alpha
        self.n_clusters = n_clusters
        self.sparsity = sparsity
        self.repetitions = repetitions
        self.cluster_method = cluster_method
        self.max_iter = max_iter
        self.tol = tol
        self.subsample = subsample
        self.calc_cost = calc_cost
        self.h_init = h_init
        self.rep = rep
        self.random_state = random_state
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.runs_per_con = max(round(repetitions * 0.8), 1)  # number of runs per consensus matrix creation


    def fit(self, Xs, y=None):
        r"""
        Fit the transformer to the input data.

        Parameters
        ----------
        Xs : list of array-likes objects
            - Xs length: n_mods
            - Xs[i] shape: (n_samples, n_features_i)

            A list of different modalities.
        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self :  returns an instance of self.
        """
        Xs = check_Xs(Xs, ensure_all_finite='allow-nan')

        if not isinstance(Xs[0], pd.DataFrame):
            Xs = [pd.DataFrame(X) for X in Xs]

        if len(self.missing) == 1:
            if self.verbose:
                print(f"#Setting all 'missing' parameters to {self.missing[0]}")
            self.missing = [self.missing[0]] * len(Xs)
        if len(self.method) == 1:
            self.method = [self.method[0]] * len(Xs)
        elif len(Xs) != len(self.method):
            raise ValueError("len(Xs) and number of similarity methods does not correspond.")
        self.graph_ = None
        self.nmf_ = None

        all_samples = Xs[0].index
        if self.verbose:
            print(f"Total number of unique samples: {len(all_samples)}")
        self.similarity_ = {}
        adj_matrices = []
        # create adjacency matrices
        for X_idx, X in enumerate(Xs):
            if self.verbose:
                print(f"#Modality: {X_idx}")
                print(f"Feature matrix: ({X.shape[0]} samples x {X.shape[1]} features)")
            # create adjacency matrix
            a = feature_to_adjacency(X.values, missing=self.missing[X_idx], method=self.method[X_idx],
                                     n=self.neighbours, alpha=self.alpha)
            if self.verbose:
                print(f"Adjacency matrix: ({a.shape} created [similarity method: {self.method[X_idx]}")
            # add matrices to output arrays
            adj_matrices.append(a)
            self.similarity_[str(X_idx)] = a

        ##################################################################
        if self.h_init is not None:
            if self.h_init >= len(adj_matrices) or self.h_init < 0:
                raise ValueError("Incorrect h_init.")

        # create multilayer graph
        self.graph_ = MultiplexNet(adj_matrices=adj_matrices, node_labels=all_samples)
        n_sub_samples = round(all_samples.size * self.subsample)
        if self.verbose:
            print(f"#Number of samples randomly removed in each run: {n_sub_samples} out of {all_samples.size}")
        # create solver
        self.nmf_ = UnsupervisedSumoNMF(graph=self.graph_, nbins=self.repetitions,
                                        bin_size=self.graph_.nodes - n_sub_samples,
                                        random_state=self.random_state)
        global _sumo_run
        _sumo_run = self  # this solves multiprocessing issue with pickling

        results = [SUMO._run_factorization(sparsity=sparsity, k=self.n_clusters,
                                           sumo_run=_sumo_run, verbose=self.verbose,
                                           random_state=self.random_state)
                   for sparsity in self.sparsity]
        sparsity_order = self.sparsity

        # select best result
        best_result = sorted(results, reverse=True)[0]
        best_eta = None

        quality_output = []
        for (result, sparsity) in zip(results, sparsity_order):
            if self.verbose:
                print(f"#Clustering quality (eta={sparsity}): {result[0]}")
            quality_output.append(np.array([sparsity, result[0]]))
            if result[1] == best_result[1]:
                best_eta = sparsity

        # summarize results
        assert best_eta is not None
        out_arrays = best_result[1]

        self.cophenet_list_ = out_arrays["cophenet"]
        self.pac_list_ = out_arrays["pac"]
        self.embedding_ = out_arrays["embedding"]
        self.labels_ = out_arrays["clusters"][:,1].astype(int)
        return self


    def _predict(self, Xs):
        r"""
        Return clustering results for samples.

        Parameters
        ----------
        Xs : list of array-likes objects
            - Xs length: n_mods
            - Xs[i] shape: (n_samples, n_features_i)

            A list of different modalities.

        Returns
        -------
        labels : list of array-likes objects, shape (n_samples,)
            The predicted data.
        """
        labels = self.labels_
        return labels


    def fit_predict(self, Xs, y=None):
        r"""
        Fit the model and return clustering results.
        Convenience method; equivalent to calling fit(X) followed by predict(X).

        Parameters
        ----------
        Xs : list of array-likes objects
            - Xs length: n_mods
            - Xs[i] shape: (n_samples, n_features_i)

            A list of different modalities.
        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        labels : ndarray, shape (n_samples,)
            The predicted data.
        """

        labels = self.fit(Xs)._predict(Xs)
        return labels


    @staticmethod
    def _run_factorization(sparsity: float, k: int, sumo_run, verbose: bool, random_state):
        """ Run factorization for set sparsity and number of clusters
        Args:
            sparsity (float): value of sparsity penalty
            k (int): number of clusters
            sumo_run: SumoRun object
        Returns:
            quality (float): assessed quality of cluster structure
            outfile (str): path to .npz output file with results of factorization
        """
        # run factorization N times
        results = []
        for repeat in range(sumo_run.repetitions):
            if verbose:
                print(f"#Runing NMF algorithm with sparsity {sparsity} (N={repeat + 1})")
            opt_args = {
                "sparsity_penalty": sparsity,
                "k": k,
                "max_iter": sumo_run.max_iter,
                "tol": sumo_run.tol,
                "calc_cost": sumo_run.calc_cost,
                "bin_id": repeat,
                "h_init": sumo_run.h_init
            }
            result = sumo_run.nmf_.factorize(**opt_args)
            # extract computed clusters
            if verbose:
                print(f"#Using {sumo_run.cluster_method} for cluster labels extraction)")
            result.extract_clusters(method=sumo_run.cluster_method, random_state=sumo_run.random_state, verbose=verbose)
            results.append(result)

        # consensus graph
        assert len(results) > 0

        all_REs = []  # residual errors
        for run_idx in range(sumo_run.repetitions):
            all_REs.append(results[run_idx].RE)

        out_arrays = {'pac': np.array([]), 'cophenet': np.array([])}
        minRE, maxRE = min(all_REs), max(all_REs)

        for rep in range(sumo_run.rep):
            run_indices = list(np.random.default_rng(sumo_run.random_state).choice(range(len(results)),
                                                                                   sumo_run.runs_per_con,
                                                                                   replace=False))

            consensus = np.zeros((sumo_run.graph_.nodes, sumo_run.graph_.nodes))
            weights = np.empty((sumo_run.graph_.nodes, sumo_run.graph_.nodes))
            weights[:] = np.nan

            all_equal = np.allclose(minRE, maxRE)

            for run_idx in run_indices:
                weight = np.empty((sumo_run.graph_.nodes, sumo_run.graph_.nodes))
                weight[:] = np.nan
                sample_ids = results[run_idx].sample_ids
                if all_equal:
                    weight[sample_ids, sample_ids[:, None]] = 1.
                else:
                    weight[sample_ids, sample_ids[:, None]] = (maxRE - results[run_idx].RE) / (maxRE - minRE)

                weights = np.nansum(np.stack((weights, weight)), axis=0)
                consensus_run = np.nanprod(np.stack((results[run_idx].connectivity, weight)), axis=0)
                consensus = np.nansum(np.stack((consensus, consensus_run)), axis=0)

            if verbose:
                print(f"#Creating consensus graphs [{rep + 1} out of {sumo_run.rep}]")
            assert not np.any(np.isnan(consensus))
            consensus = consensus / weights

            org_con = consensus.copy()
            consensus[consensus < 0.5] = 0

            # calculate cophenetic correlation coefficient
            dist = pdist(org_con, metric="correlation")
            if np.any(np.isnan(dist)):
                ccc = np.nan
                if verbose:
                    print(
                        "Cannot calculate cophenetic correlation coefficient! Please inspect values in your consensus matrix")
            else:
                ccc = cophenet(linkage(dist, method="complete", metric="correlation"), dist)[0]

            # calculate proportion of ambiguous clustering
            den = (sumo_run.graph_.nodes ** 2) - sumo_run.graph_.nodes
            num = org_con[(org_con > 0.1) & (org_con < 0.9)].size
            pac = num * (1. / den)

            out_arrays.update({'pac': np.append(out_arrays['pac'], pac),
                               'cophenet': np.append(out_arrays['cophenet'], ccc)})

        if verbose:
            print("#Extracting final clustering result, using normalized cut")
        consensus_labels, embeddings = extract_ncut(consensus, k=k, random_state=random_state)

        cluster_array = np.empty((sumo_run.graph_.sample_names.shape[0], 2), dtype=object)
        cluster_array[:, 0] = sumo_run.graph_.sample_names
        cluster_array[:, 1] = consensus_labels

        clusters_dict = {num: sumo_run.graph_.sample_names[list(np.where(consensus_labels == num)[0])] for num in
                         np.unique(consensus_labels)}
        for cluster_idx in sorted(clusters_dict.keys()):
            if verbose:
                print(
                    f"Cluster {cluster_idx} ({len(clusters_dict[cluster_idx])} samples): \n{clusters_dict[cluster_idx]}")

        # calculate quality of clustering for given sparsity
        quality = sumo_run.graph_.get_clustering_quality(labels=cluster_array[:, 1])
        # create output file
        conf_array = np.empty((9, 2), dtype=object)
        conf_array[:, 0] = ['method', 'n', 'max_iter', 'tol', 'subsample', 'calc_cost', 'h_init', 'seed', 'sparsity']
        conf_array[:, 1] = [sumo_run.cluster_method, sumo_run.repetitions, sumo_run.max_iter, sumo_run.tol,
                            sumo_run.subsample,
                            sumo_run.calc_cost, np.nan if sumo_run.h_init is None else sumo_run.h_init,
                            np.nan if sumo_run.random_state is None else sumo_run.random_state, sparsity]
        out_arrays.update({
            "clusters": cluster_array,
            "consensus": consensus,
            "unfiltered_consensus": org_con,
            "quality": np.array(quality),
            "samples": sumo_run.graph_.sample_names,
            "embedding": embeddings,
            "config": conf_array
        })

        steps_reached = [results[i].steps for i in range(len(results))]
        maxiter_proc = round((sum([step == sumo_run.max_iter for step in steps_reached]) / len(steps_reached)) * 100, 3)
        if verbose:
            print(f"#Reached maximum number of iterations in {maxiter_proc}% of runs")
        if maxiter_proc >= 90:
            if verbose:
                print(f"Consider increasing -max_iter and decreasing -tol to achieve better accuracy")
        out_arrays['steps'] = np.array([steps_reached])
        return quality, out_arrays


