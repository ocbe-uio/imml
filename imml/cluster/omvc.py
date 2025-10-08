# License: BSD-3-Clause

import os
from os.path import dirname
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import KMeans

from ..utils import check_Xs
from ..explore import get_missing_samples_by_mod

matlabmodule_installed = False
oct2py_module_error = "Module 'matlab' needs to be installed. See https://imml.readthedocs.io/stable/main/installation.html#optional-dependencies"
try:
    import oct2py
    matlabmodule_installed = True
except ImportError:
    pass


class OMVC(BaseEstimator, ClassifierMixin):
    r"""
    Online Multi-View Clustering (OMVC). [#omvcpaper]_ [#omvccode]_

    OMVC aims to learn latent feature matrices for all views while driving them towards a consensus. To enhance the
    robustness of these learned matrices, it incorporates lasso regularization. Additionally, to mitigate the impact of
    incomplete data, it introduces dynamic weight adjustment.

    Parameters
    ----------
    n_clusters : int, default=8
        The number of clusters to generate.
    max_iter : int, default=3
        Maximum number of iterations.
    tol : float, default=1e-4
        Tolerance of the stopping condition.
    block_size : int, default=50
        Size of the chunk.
    n_pass : int, default=1
        Number of passes.
    random_state : int, default=None
        Determines the randomness. Use an int to make the randomness deterministic.
    engine : str, default=matlab
        Engine to use for computing the model. Current options are 'matlab'.
    verbose : bool, default=False
        Verbosity mode.
    clean_space : bool, default=True
        If engine is 'matlab' and clean_space is True, the session will be closed after fitting the model.

    Attributes
    ----------
    labels_ : array-like of shape (n_samples,)
        Labels of each point in training data.
    embedding_ : array-like of shape (n_samples, n_clusters)
        Common consensus, latent feature matrix across all the views to be used as input for the KMeans clustering step.
    U_ : list of n_mods array-like of shape (n_samples, n_clusters)
        Basis matrix.
    V_ : list of n_mods array-like of shape (n_features_i, n_clusters)
        Latent feature matrix.
    loss_ : array-like of shape (n_iter\_,)
        Values of the loss function.
    n_iter_ : int
        Number of iterations.

    References
    ----------
    .. [#omvcpaper] W. Shao, L. He, C. -t. Lu and P. S. Yu, "Online multi-view clustering with incomplete views,"
                    2016 IEEE International Conference on Big Data (Big Data), Washington, DC, USA, 2016, pp.
                    1012-1017, doi: 10.1109/BigData.2016.7840701.
    .. [#omvccode] https://github.com/software-shao/online-multiview-clustering-with-incomplete-view

    Example
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from imml.cluster import OMVC
    >>> Xs = [pd.DataFrame(np.random.default_rng(42).random((20, 10))) for i in range(3)]
    >>> estimator = OMVC(n_clusters = 2)
    >>> labels = estimator.fit_predict(Xs)

    """

    def __init__(self, n_clusters: int = 8, max_iter: int = 200, tol: float = 1e-4, decay: float = 1,
                 block_size: int = 50, n_pass: int = 1, random_state:int = None,
                 engine: str ="matlab", verbose = False, clean_space: bool = True):
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
        self.max_iter = max_iter
        self.tol = tol
        self.decay = decay
        self.block_size = block_size
        self.n_pass = n_pass
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
            n_mods = len(Xs)
            ones = np.ones((n_mods, 1))
            option = {"k": self.n_clusters, "maxiter": self.max_iter, "tol": self.tol, "num_cluster": self.n_clusters,
                      "decay": self.decay, "alpha": 1e-2*ones, "beta": 1e-7*ones,
                      "pass": self.n_pass}

            if isinstance(Xs[0], pd.DataFrame):
                transformed_Xs = [X.values for X in Xs]
            elif isinstance(Xs[0], np.ndarray):
                transformed_Xs = Xs
            missing_samples_by_mod = get_missing_samples_by_mod(Xs=transformed_Xs, return_as_list=True)
            missing_samples_by_mod = tuple([np.array(missing_samples)+1 for missing_samples in missing_samples_by_mod])
            transformed_Xs = [np.nan_to_num(np.clip(X, a_min=0, a_max=None), nan=0.0) for X in transformed_Xs]
            transformed_Xs = [X/(X.sum().sum()) for X in transformed_Xs]

            if self.random_state is not None:
                self._oc.rand('seed', self.random_state)
            u, v, u_star_loss, loss = self._oc.ONMF_Multi_PGD_search(transformed_Xs, option, len(Xs[0]),
                                                               missing_samples_by_mod, self.block_size, nout=4)
            u_star_loss = u_star_loss[self.n_pass-1]
            v = [np.array(arr) for arr in v[0]]
            u = [np.array(arr[0]) for arr in u]

            if self.clean_space:
                self._clean_space()

        model = KMeans(n_clusters= self.n_clusters, n_init= "auto", random_state= self.random_state)
        self.labels_ = model.fit_predict(X= u_star_loss)
        self.U_ = u
        self.V_ = v
        self.embedding_ = u_star_loss
        if isinstance(loss, float):
            loss = np.array([[loss]])
        self.loss_ = loss[0]
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

