# License: BSD-3-Clause

import os
from os.path import dirname
import numpy as np
import pandas as pd
import random
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import KMeans
from scipy.sparse.linalg import eigs

from ..impute import get_observed_mod_indicator
from ..utils import check_Xs

matlabmodule_installed = False
oct2py_module_error = "Module 'matlab' needs to be installed. See https://imml.readthedocs.io/stable/main/installation.html#optional-dependencies"
try:
    import oct2py
    matlabmodule_installed = True
except ImportError:
    pass


class IMSR(BaseEstimator, ClassifierMixin):
    r"""
    Self-representation Subspace Clustering for Incomplete Multi-view Data (IMSR). [#imsrpaper]_ [#imscaglcode]_

    IMSR performs feature extraction, imputation and self-representation learning to obtain a low-rank regularized
    consensus coefficient matrix.

    Parameters
    ----------
    n_clusters : int, default=8
        The number of clusters to generate.
    lbd : float, default=1
        Positive trade-off parameter used for the optimization function. It is recommended to set from 0 to 1.
    gamma : float, default=1
        Positive trade-off parameter used for the optimization function. It is recommended to set from 0 to 1.
    random_state : int, default=None
        Determines the randomness. Use an int to make the randomness deterministic.
    engine : str, default=python
        Engine to use for computing the model. Current options are 'matlab' or 'python'.
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
    .. [#imsrpaper] Jiyuan Liu, Xinwang Liu, Yi Zhang, Pei Zhang, Wenxuan Tu, Siwei Wang, Sihang Zhou, Weixuan Liang,
                     Siqi Wang, and Yuexiang Yang. 2021. Self-Representation Subspace Clustering for Incomplete
                     Multi-view Data. In Proceedings of the 29th ACM International Conference on Multimedia (MM '21).
                     Association for Computing Machinery, New York, NY, USA, 2726–2734.
                     https://doi.org/10.1145/3474085.3475379.
    .. [#imscaglcode] https://github.com/liujiyuan13/IMSR-code_release

    Example
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from imml.cluster import IMSR
    >>> Xs = [pd.DataFrame(np.random.default_rng(42).random((20, 10))) for i in range(3)]
    >>> estimator = IMSR(n_clusters = 2)
    >>> labels = estimator.fit_predict(Xs)
    """

    def __init__(self, n_clusters: int = 8, lbd : float = 1, gamma: float = 1, random_state:int = None,
                 engine: str ="python", verbose = False, clean_space: bool = True):
        if not isinstance(n_clusters, int):
            raise ValueError(f"Invalid n_clusters. It must be an int. A {type(n_clusters)} was passed.")
        if n_clusters < 2:
            raise ValueError(f"Invalid n_clusters. It must be an greater than 1. {n_clusters} was passed.")
        engines_options = ["matlab", "python"]
        if engine not in engines_options:
            raise ValueError(f"Invalid engine. Expected one of {engines_options}. {engine} was passed.")
        if (engine == "matlab") and (not matlabmodule_installed):
            raise ImportError(oct2py_module_error)
        if lbd <= 0:
            raise ValueError(f"Invalid lbd. It must be a positive value. {lbd} was passed.")
        if gamma <= 0:
            raise ValueError(f"Invalid gamma. It must be a positive value. {gamma} was passed.")

        self.n_clusters = n_clusters
        self.lbd = lbd
        self.gamma = gamma
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

        if not isinstance(Xs[0], pd.DataFrame):
            Xs = [pd.DataFrame(X) for X in Xs]
        observed_mod_indicator = get_observed_mod_indicator(Xs)
        if isinstance(observed_mod_indicator, pd.DataFrame):
            observed_mod_indicator = observed_mod_indicator.reset_index(drop=True)
        observed_mod_indicator = [(1 + missing_mod[missing_mod == 0].index).to_list() for _, missing_mod in observed_mod_indicator.items()]
        transformed_Xs = [X.T.values for X in Xs]

        if self.engine=="matlab":
            if self.random_state is not None:
                self._oc.rand('seed', self.random_state)
            Z, obj = self._oc.IMSC(transformed_Xs, tuple(observed_mod_indicator), self.n_clusters, self.lbd, self.gamma, nout=2)

            if self.clean_space:
                self._clean_space()

        elif self.engine == "python":
            Z, obj = self._imsc(transformed_Xs, tuple(observed_mod_indicator), self.n_clusters, self.lbd, self.gamma)

        model = KMeans(n_clusters= self.n_clusters, n_init="auto", random_state= self.random_state)
        self.labels_ = model.fit_predict(X= Z)
        self.embedding_, self.loss_ = Z, obj
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



    def _imsc(self, X, Im, n_cluters, lbd, gamma):
        r"""
        Runs the IMSR clustering algorithm.

        Parameters
        ----------
        X : list of array-likes objects
            - Xs length: n_mods
            - Xs[i] shape: (n_samples, n_features_i)
        Im : array of shape (n_mods, columns_with_missing_values)
        n_cluters : int
            The number of clusters.
        lbd : float, default=1
            Positive trade-off parameter used for the optimization function. It is recommended to set from 0 to 1.
        gamma : float, default=1
            Positive trade-off parameter used for the optimization function. It is recommended to set from 0 to 1.

        Returns
        -------
        Utmp : list of array-likes objects of shape (n_samples, n_clusters)
        obj : float
        """
        V = len(X)
        max_iter = 100

        # Initialization
        X = [np.nan_to_num(mat, nan=0.0) for mat in X]
        beta = np.ones(shape=(V, )) / V

        Z = self._init_z(X, beta, lbd)

        t = 0
        flag = 1
        obj = []
        while flag:
            F = self._update_f(Z, n_cluters)
            Z = self._update_z(X, F, beta, lbd, gamma)
            X = self._update_x(X, Im, Z)

            append_obj, _, _, _ = self._cal_obj(X, Z, F, beta, lbd, gamma)
            obj.append([append_obj])

            if (t >= 2) and ((np.abs(np.subtract(obj[t - 1], obj[t]) / (obj[t])) < 1e-3) or (t > max_iter)):
                flag = 0
            t += 1

        Ztmp = (np.abs(Z) + np.abs(Z).T) / 2
        Utmp = np.real(self._baseline_spectral_onkernel(Ztmp, n_cluters))

        return Utmp, obj

    @staticmethod
    def _init_z(X, beta, lbd):
        r"""
        Initializes Z variable.

        Parameters
        ----------
        X : list of array-likes objects
            - X length: n_mods
            - X[i] shape: (n_samples, n_features_i)
        beta : list of n_mods values
        lbd : float, default=1
            Positive trade-off parameter used for the optimization function. It is recommended to set from 0 to 1.

        Returns
        -------
        Z : list of array-likes objects of shape (n_samples, n_samples)
        """
        V = len(X)
        n = X[0].shape[1]

        D = np.zeros(shape=(n, n))
        D = D + lbd * np.diag(np.ones(shape=(n,)))
        for v in range(V):
            D = D + beta[v] * (np.matmul(X[v].T, X[v]))

        D = np.linalg.inv(D)
        Z = -D / np.diag(D).T[: np.newaxis]
        Z = Z - np.diag(np.diag(Z))
        Z = (Z + Z.T) / 2

        return Z

    @staticmethod
    def _update_f(Z, n_cluters):
        r"""
        Updates the F variables.

        Parameters
        ----------
        Z : list of array-likes objects of shape (n_samples, n_samples)
        n_cluters : int
            The number of clusters.

        Returns
        -------
        F : list of array-likes objects of shape (n_clusters, n_samples)
        """
        _, Ftmp = eigs(A=(Z + Z.T), k=n_cluters, which='LR')
        F = Ftmp.T

        return np.real(F)

    @staticmethod
    def _update_z(X, F, beta, lbd, gamma):
        r"""
        Updates the Z variables.

        Parameters
        ----------
        X : list of array-likes objects
            - Xs length: n_mods
            - Xs[i] shape: (n_samples, n_features_i)
        F : list of array-likes objects of shape (n_clusters, n_samples)
        beta : list of n_mods values
        lbd : float, default=1
            Positive trade-off parameter used for the optimization function. It is recommended to set from 0 to 1.
        gamma : float, default=1
            Positive trade-off parameter used for the optimization function. It is recommended to set from 0 to 1.

        Returns
        -------
        Z : list of array-likes objects of shape (n_samples, n_samples)
        """
        V = len(X)
        n = X[0].shape[1]

        K = np.zeros(shape=(n, n))
        for v in range(V):
            K += beta[v] * np.matmul(X[v].T, X[v])

        C = np.matmul(F.T, F)
        base = K + (lbd + gamma) * np.eye(n)

        D = np.linalg.inv(base)
        beta = np.diag(D)
        Z1 = -D / np.diag(D).T[: np.newaxis]
        Z1 -= np.diag(np.diag(Z1))

        C = C - np.diag(np.diag(C))
        a = np.matmul(D, C)
        a -= np.diag(np.diag(a))
        b = np.diag(np.matmul(Z1.T, C))
        c = beta * b
        d = np.matmul(Z1, np.diag(c))
        d -= np.diag(np.diag(d))
        Z2 = gamma * (a - d)

        Z = Z1 + Z2
        return Z


    @staticmethod
    def _update_x(X, Im, Z):
        r"""
        Updates the X variables.

        Parameters
        ----------
        X : list of array-likes objects
            - Xs length: n_mods
            - Xs[i] shape: (n_samples, n_features_i)
        Im : array of shape (n_mods, columns_with_missing_values)
        Z : list of array-likes objects of shape (n_samples, n_samples)

        Returns
        -------
        X : list of array-likes objects
            - Xs length: n_mods
            - Xs[i] shape: (n_samples, n_features_i)
        """
        V = len(X)
        n = X[0].shape[1]

        B = np.eye(n) - Z - Z.T + (np.matmul(Z, Z.T))
        Io = [None] * V
        for v in range(V):
            Im_temp = [i-1 for i in Im[v]]
            Io[v] = np.setdiff1d(ar1=np.array([i for i in range(0, n)]), ar2=Im_temp)
            X[v][:, Im_temp] = np.linalg.solve(B[np.ix_(Im_temp, Im_temp)].conj().T,
                                               (np.matmul(-X[v][:, Io[v]], B[np.ix_(Io[v], Im_temp)]).conj().T)).conj().T

        return X

    @staticmethod
    def _cal_obj(X, Z, F, beta, lbd, gamma):
        r"""
        Returns pbj values with the individual terms that contribute to it.

        Parameters
        ----------
        X : list of array-likes objects
            - Xs length: n_mods
            - Xs[i] shape: (n_samples, n_features_i)
        Z : list of array-likes objects of shape (n_samples, n_samples)
        F
        beta : list of n_views values
        lbd : float, default=1
            Positive trade-off parameter used for the optimization function. It is recommended to set from 0 to 1.
        gamma : float, default=1
            Positive trade-off parameter used for the optimization function. It is recommended to set from 0 to 1.

        Returns
        -------
        obj : float
        term1 : float
        term2 : float
        term3 : float
        """
        V = len(X)
        term1 = 0

        for v in range(V):
            tmp = X[v] - np.matmul(X[v], Z)
            term1 = term1 + beta[v] * np.sum(np.sum(tmp ** 2))

        term2 = np.sum(np.sum(Z ** 2))
        tmp = Z - np.matmul(F.T, F)
        term3 = np.sum(np.sum(tmp ** 2))
        obj = term1 + lbd * term2 + gamma * term3

        return obj, term1, term2, term3

    @staticmethod
    def _baseline_spectral_onkernel(K, n_clusters):
        r"""
        This function returns the top eigenvectors of the given matrix.

        Parameters
        ----------
        K : list of array-likes objects of shape (n_samples, n_samples)
        n_clusters: int
            The number of clusters.

        Returns
        -------
        U : list of array-likes objects of shape (n_samples, n_clusters)
        """
        D = np.diag(np.sum(K, axis=1) + np.finfo(float).eps)
        inv_sqrt_D = np.sqrt(np.linalg.inv(np.abs(D)))
        L = np.matmul(np.matmul(inv_sqrt_D, K), inv_sqrt_D)
        L = (L + L.T) / 2
        V, U = eigs(L, k=n_clusters, which='LR', tol=0)

        return U
