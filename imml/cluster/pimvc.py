# License: BSD-3-Clause

import os
from os.path import dirname
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import KMeans

from ..impute import get_observed_mod_indicator
from ..utils import check_Xs

matlabmodule_installed = False
oct2py_module_error = "Module 'matlab' needs to be installed. See https://imml.readthedocs.io/stable/main/installation.html#optional-dependencies"
try:
    import oct2py
    matlabmodule_installed = True
except ImportError:
    pass


class PIMVC(BaseEstimator, ClassifierMixin):
    r"""
    Projective Incomplete Multi-View Clustering (PIMVC). [#pimvcpaper]_ [#pimvccode]_

    The objective of PIMVC is to simultaneously discover the projection matrix for each modality and establish a unified
    feature representation shared across incomplete multiple views, facilitating clustering. Essentially, PIMVC
    transforms the traditional multi-modality matrix factorization model into a multi-modality projection learning model. By
    consolidating various modality-specific objective losses into a cohesive subspace of equal dimensions, it adeptly
    handles the challenge where a single modality might overly influence consensus representation learning due to
    imbalanced information across views stemming from diverse dimensions. Furthermore, to capture the data geometric
    structure, PIMVC incorporates a penalty term for graph regularization.

    Parameters
    ----------
    n_clusters : int, default=8
        The number of clusters to generate.
    dele : float, default=0.1
        nonnegative.
    lamb : float, default=100000
        Penalty parameters. Should be greather than 0.
    beta : float, default=1
        Trade-off parameter.
    k : int, default=3
        Parameter k of KNN graph.
    neighbor_mode : str, default='KNN'
        Indicates how to construct the graph. Options are 'KNN' (default), and 'Supervised'.
    weight_mode : str, default='Binary'
        Indicates how to assign weights for each edge in the graph. Options are 'Binary' (default), 'Cosine' and 'HeatKernel'.
    max_iter : int, default=100
        Maximum number of iterations.
    random_state : int, default=None
        Determines the randomness. Use an int to make the randomness deterministic.
    engine : str, default=matlab
        Engine to use for computing the model. Currently only 'matlab' is supported.
    verbose : bool, default=False
        Verbosity mode.
    clean_space : bool, default=True
        If engine is 'matlab' and clean_space is True, the session will be closed after fitting the model.

    Attributes
    ----------
    labels_ : array-like of shape (n_samples,)
        Labels of each point in training data.
    embedding_ : array-like of shape (n_samples, n_clusters)
        Consensus clustering matrix to be used as input for the KMeans clustering step.
    loss_ : array-like of shape (n_iter\_,)
        Values of the loss function.
    n_iter_ : int
        Number of iterations.

    References
    ----------
    .. [#pimvcpaper] S. Deng, J. Wen, C. Liu, K. Yan, G. Xu and Y. Xu, "Projective Incomplete Multi-View Clustering,"
                     in IEEE Transactions on Neural Networks and Learning Systems, doi: 10.1109/TNNLS.2023.3242473.
    .. [#pimvccode] https://github.com/Dshijie/PIMVC

    Example
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from imml.cluster import PIMVC
    >>> Xs = [pd.DataFrame(np.random.default_rng(42).random((20, 10))) for i in range(3)]
    >>> estimator = PIMVC(n_clusters = 2)
    >>> labels = estimator.fit_predict(Xs)
    """

    def __init__(self, n_clusters: int = 8, dele: float = 0.1, lamb: int = 100000, beta: int = 1, k: int = 3,
                 neighbor_mode: str = 'KNN', weight_mode: str = 'Binary', max_iter: int = 100,
                 random_state: int = None, engine: str = "matlab", verbose = False, clean_space: bool = True):
        if not isinstance(n_clusters, int):
            raise ValueError(f"Invalid n_clusters. It must be an int. A {type(n_clusters)} was passed.")
        if n_clusters < 2:
            raise ValueError(f"Invalid n_clusters. It must be an greater than 1. {n_clusters} was passed.")
        engines_options = ["matlab"]
        if engine not in engines_options:
            raise ValueError(f"Invalid engine. Expected one of {engines_options}. {engine} was passed.")
        if (engine == "matlab") and (not matlabmodule_installed):
            raise ImportError(oct2py_module_error)
        if lamb <= 0:
            raise ValueError(f"Invalid lamb. It must be a positive value. {lamb} was passed.")
        if k <= 0:
            raise ValueError(f"Invalid k. It must be a positive value. {k} was passed.")

        self.n_clusters = n_clusters
        self.dele = dele
        self.lamb = lamb
        self.beta = beta
        self.k = k
        self.neighbor_mode = neighbor_mode
        self.weight_mode = weight_mode
        self.max_iter = max_iter
        self.random_state = random_state
        self.engine = engine
        self.verbose = verbose
        self.clean_space = clean_space

        if self.engine == "matlab":
            matlab_folder = dirname(__file__)
            matlab_folder = os.path.join(matlab_folder, "_" + (os.path.basename(__file__).split(".")[0]))
            self._matlab_folder = matlab_folder
            matlab_files = [x for x in os.listdir(matlab_folder) if x.endswith(".m")]
            self._oc = oct2py.Oct2Py(temp_dir= matlab_folder)
            for matlab_file in matlab_files:
                with open(os.path.join(matlab_folder, matlab_file)) as f:
                    self._oc.eval(f.read())


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
        self :  Fitted estimator.
        """
        Xs = check_Xs(Xs, ensure_all_finite='allow-nan')

        try:
            assert self.n_clusters <= min([X.shape[1] for X in Xs])
        except AssertionError:
            raise ValueError(f"n_clusters ({self.n_clusters}) should be smaller or equal to " +
                             f"the smallest n_features_i ({min([X.shape[1] for X in Xs])}).")

        if self.engine=="matlab":

            observed_mod_indicator = get_observed_mod_indicator(Xs)
            if isinstance(observed_mod_indicator, pd.DataFrame):
                observed_mod_indicator = observed_mod_indicator.values
            transformed_Xs = tuple([X.T for X in Xs])

            if self.random_state is not None:
                self._oc.rand('seed', self.random_state)
            v, loss = self._oc.PIMVC(transformed_Xs, self.n_clusters, observed_mod_indicator, self.lamb, self.beta,
                               self.max_iter, {"NeighborMode": self.neighbor_mode,
                                               "WeightMode": self.weight_mode,
                                               "k": self.k}, nout=2)

            if self.clean_space:
                self._clean_space()

        model = KMeans(n_clusters= self.n_clusters, n_init= "auto", random_state= self.random_state)
        v = v.T
        self.labels_ = model.fit_predict(X= v)
        self.embedding_ = v
        self.loss_ = loss[:, 0]
        self.n_iter_ = len(self.loss_)

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
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        return self.labels_


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

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """

        labels = self.fit(Xs)._predict(Xs)
        return labels


    def _clean_space(self):
        [os.remove(os.path.join(self._matlab_folder, x)) for x in ["reader.mat", "writer.mat"]]
        self._oc.exit()
        del self._oc
        return None

