# License: BSD-3-Clause

import os
from os.path import dirname
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import KMeans

from ..impute import get_observed_mod_indicator
from ..utils import check_Xs
from ..preprocessing import remove_missing_samples_by_mod

matlabmodule_installed = False
oct2py_module_error = "Module 'matlab' needs to be installed. See https://imml.readthedocs.io/stable/main/installation.html#optional-dependencies"
try:
    import oct2py
    matlabmodule_installed = True
except ImportError:
    pass


class IMSCAGL(BaseEstimator, ClassifierMixin):
    r"""
    Incomplete Multiview Spectral Clustering With Adaptive Graph Learning (IMSCAGL). [#imscaglpaper1]_ [#imscaglpaper2]_ [#imscaglcode1]_ [#imscaglcode2]_

    IMSCAGL utilizes graph learning and spectral clustering techniques to derive a unified representation for
    incomplete multiview clustering.

    Parameters
    ----------
    n_clusters : int, default=8
        The number of clusters to generate.
    lambda1 : float, default=0.1
        Penalty parameter for learning model of the multi-modal subspace clustering.
    lambda2 : float, default=1000
        Penalty parameter for learning model of the multi-modal subspace clustering.
    lambda3 : float, default=100
        Penalty parameter for learning the consensus representation from those cluster indicator matrices of all views.
    k : int, default=5
        Parameter k of KNN graph.
    neighbor_mode : str, default='KNN'
        Indicates how to construct the graph. Options are 'KNN' (default), and 'Supervised'.
    weight_mode : str, default='Binary'
        Indicates how to assign weights for each edge in the graph. Options
        are 'Binary' (default), 'Cosine' and 'HeatKernel'.
    max_iter : int, default=100
        Maximum number of iterations.
    miu : float, default=0.01
        Constant for updating variables during the learning process.
    rho : float, default=100
        Constant for updating variables during the learning process.
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
        Consensus representation matrix to be used as input for the KMeans clustering step.

    References
    ----------
    .. [#imscaglpaper1] J. Wen, Y. Xu and H. Liu, "Incomplete Multiview Spectral Clustering With Adaptive Graph
                         Learning," in IEEE Transactions on Cybernetics, vol. 50, no. 4, pp. 1418-1429, April 2020,
                         doi: 10.1109/TCYB.2018.2884715.
    .. [#imscaglpaper2] Jie Wen, Zheng Zhang, Lunke Fei, Bob Zhang, Yong Xu, Zhao Zhang, Jinxing Li, A Survey on
                         Incomplete Multi-view Clustering, IEEE TRANSACTIONS ON SYSTEMS, MAN, AND CYBERNETICS:
                         SYSTEMS, 2022.
    .. [#imscaglcode1] https://github.com/DarrenZZhang/Survey_IMC
    .. [#imscaglcode2] https://github.com/ckghostwj/Incomplete-Multiview-Spectral-Clustering-with-Adaptive-Graph-Learning

    Example
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from imml.cluster import IMSCAGL
    >>> Xs = [pd.DataFrame(np.random.default_rng(42).random((20, 10))) for i in range(3)]
    >>> estimator = IMSCAGL(n_clusters = 2)
    >>> labels = estimator.fit_predict(Xs)
    """

    def __init__(self, n_clusters: int = 8, lambda1: float = 0.1, lambda2: float = 1000, lambda3: float = 100, k: int = 5,
                 neighbor_mode: str = 'KNN', weight_mode: str = 'Binary', max_iter: int = 100, miu: float = 0.01,
                 rho: float = 1.1, random_state: int = None, engine: str = "matlab", verbose = False,
                 clean_space: bool = True):
        if not isinstance(n_clusters, int):
            raise ValueError(f"Invalid n_clusters. It must be an int. A {type(n_clusters)} was passed.")
        if n_clusters < 2:
            raise ValueError(f"Invalid n_clusters. It must be an greater than 1. {n_clusters} was passed.")
        engines_options = ["matlab"]
        if engine not in engines_options:
            raise ValueError(f"Invalid engine. Expected one of {engines_options}. {engine} was passed.")
        if (engine == "matlab") and (not matlabmodule_installed):
            raise ImportError(oct2py_module_error)

        self.n_clusters = n_clusters
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.miu = miu
        self.rho = rho
        self.beta = rho
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

        if self.engine=="matlab":
            if not isinstance(Xs[0], pd.DataFrame):
                Xs = [pd.DataFrame(X) for X in Xs]
            observed_mod_indicator = get_observed_mod_indicator(Xs=Xs)
            transformed_Xs = remove_missing_samples_by_mod(Xs=Xs)
            w = [pd.DataFrame(np.eye(len(X)), index=X.index, columns=X.index) for X in Xs]
            w = [eye.loc[samples,:].values for eye, (_, samples) in zip(w, observed_mod_indicator.items())]
            w = tuple(w)

            transformed_Xs = tuple([X.T for X in transformed_Xs])

            if self.random_state is not None:
                self._oc.rand('seed', self.random_state)
            F = self._oc.IMSAGL(transformed_Xs, w, self.n_clusters, self.lambda1, self.lambda2, self.lambda3,
                          self.miu, self.rho, self.max_iter,
                          {"NeighborMode": self.neighbor_mode, "WeightMode": self.weight_mode, "k": self.k})

            if self.clean_space:
                self._clean_space()

        model = KMeans(n_clusters= self.n_clusters, n_init="auto", random_state= self.random_state)
        self.labels_ = model.fit_predict(X= F)
        self.embedding_ = F

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
