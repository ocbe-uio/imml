# License: BSD-3-Clause

import os
from os.path import dirname
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

from ..impute import get_observed_mod_indicator, simple_mod_imputer
from ..utils import check_Xs

matlabmodule_installed = False
oct2py_module_error = "Module 'matlab' needs to be installed. See https://imml.readthedocs.io/stable/main/installation.html#optional-dependencies"
try:
    import oct2py
    matlabmodule_installed = True
except ImportError:
    pass


class OPIMC(BaseEstimator, ClassifierMixin):
    r"""
    One-Pass Incomplete Multi-View Clustering (OPIMC). [#opimcpaper1]_ [#opimcpaper2]_ [#opimccode]_

    OPIMC deals with large scale incomplete multi-view clustering problem by considering the instance missing
    information with the help of regularized matrix factorization and weighted matrix factorization.

    Parameters
    ----------
    n_clusters : int, default=8
        The number of clusters to generate.
    alpha : float, default=10
        Nonnegative parameter.
    max_iter : int, default=30
        Maximum number of iterations.
    tol : float, default=1e-6
        Tolerance of the stopping condition.
    block_size : int, default=50
        Size of the chunk.
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
        Consensus clustering matrix to be used as input for the KMeans clustering step.

    References
    ----------
    .. [#opimcpaper1] Hu, M., & Chen, S. (2019). One-Pass Incomplete Multi-View Clustering. Proceedings of the AAAI
                     Conference on Artificial Intelligence, 33(01), 3838-3845.
                     https://doi.org/10.1609/aaai.v33i01.33013838.
    .. [#opimcpaper2] Jie Wen, Zheng Zhang, Lunke Fei, Bob Zhang, Yong Xu, Zhao Zhang, Jinxing Li, A Survey on
                      Incomplete Multi-view Clustering, IEEE TRANSACTIONS ON SYSTEMS, MAN, AND CYBERNETICS:
                      SYSTEMS, 2022.
    .. [#opimccode] https://github.com/software-shao/online-multiview-clustering-with-incomplete-view

    Example
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from imml.cluster import OPIMC
    >>> Xs = [pd.DataFrame(np.random.default_rng(42).random((20, 10))) for i in range(3)]
    >>> estimator = OPIMC(n_clusters = 2)
    >>> labels = estimator.fit_predict(Xs)
    """

    def __init__(self, n_clusters: int = 8, alpha: float = 10, num_passes: int = 1, max_iter: int = 30,
                 tol: float = 1e-6, block_size: int = 250, random_state:int = None, engine: str ="matlab",
                 verbose = False, clean_space: bool = True):
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
        self.alpha = alpha
        self.num_passes = num_passes
        self.max_iter = max_iter
        self.tol = tol
        self.block_size = block_size
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
            if isinstance(Xs[0], pd.DataFrame):
                transformed_Xs = [X.values for X in Xs]
            elif isinstance(Xs[0], np.ndarray):
                transformed_Xs = Xs
            observed_mod_indicator = get_observed_mod_indicator(transformed_Xs)
            transformed_Xs = simple_mod_imputer(transformed_Xs, value="zeros")
            transformed_Xs = [X.T for X in transformed_Xs]
            transformed_Xs = tuple(transformed_Xs)

            w = tuple([self._oc.diag(missing_mod) for missing_mod in observed_mod_indicator])
            options = {"block_size": self.block_size, "k": self.n_clusters, "maxiter": self.max_iter,
                       "tol": self.tol, "pass": self.num_passes, "loss": 0, "alpha": self.alpha}
            if self.random_state is not None:
                self._oc.rand('seed', self.random_state)
            labels, V = self._oc.OPIMC(transformed_Xs, w, options, observed_mod_indicator, nout= 2)

            if self.clean_space:
                self._clean_space()

        self.labels_ = pd.factorize(labels[:,0])[0]
        self.embedding_ = V

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

