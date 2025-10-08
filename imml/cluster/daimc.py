# License: BSD-3-Clause

import os
from os.path import dirname
import numpy as np
import pandas as pd
from control.matlab import lyap
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import KMeans

from ..impute import get_observed_mod_indicator, simple_mod_imputer
from ..utils import check_Xs

matlabmodule_installed = False
oct2py_module_error = "Module 'matlab' needs to be installed. See https://imml.readthedocs.io/stable/main/installation.html#optional-dependencies"
try:
    import oct2py
    matlabmodule_installed = True
except ImportError:
    pass


class DAIMC(BaseEstimator, ClassifierMixin):
    r"""
    Doubly Aligned Incomplete Multi-view Clustering (DAIMC). [#daimcpaper1]_ [#daimcpaper2]_ [#daimccode]_

    The DAIMC algorithm integrates weighted semi-nonnegative matrix factorization (semi-NMF) to address incomplete
    multi-view clustering challenges. It leverages instance alignment information to learn a unified latent feature
    matrix across views and employs L2,1-Norm regularized regression to establish a consensus basis matrix, minimizing
    the impact of missing instances.

    Parameters
    ----------
    n_clusters : int, default=8
        The number of clusters to generate.
    alpha : float, default=1
        Nonnegative value.
    beta : float, default=1
        Define the trade-off between sparsity and accuracy of regression for the i-th modality.
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
        Commont latent feature matrix to be used as input for the KMeans clustering step.
    U_ : list of n_mods array-like of shape (n_features_i, n_clusters)
        Basis matrices.
    B_ : list of n_mods array-like of shape (n_features_i, n_clusters)
        Regression coefficient matrices.

    References
    ----------
    .. [#daimcpaper1] Menglei Hu and Songcan Chen. 2018. Doubly aligned incomplete multi-view clustering. In
                      Proceedings of the 27th International Joint Conference on Artificial Intelligence (IJCAI'18).
                      AAAI Press, 2262–2268.
    .. [#daimcpaper2] Jie Wen, Zheng Zhang, Lunke Fei, Bob Zhang, Yong Xu, Zhao Zhang, Jinxing Li, A Survey on
                      Incomplete Multi-view Clustering, IEEE TRANSACTIONS ON SYSTEMS, MAN, AND CYBERNETICS:
                      SYSTEMS, 2022.
    .. [#daimccode] https://github.com/DarrenZZhang/Survey_IMC

    Example
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from imml.cluster import DAIMC
    >>> Xs = [pd.DataFrame(np.random.default_rng(42).random((20, 10))) for i in range(3)]
    >>> estimator = DAIMC(n_clusters = 2)
    >>> labels = estimator.fit_predict(Xs)
    """

    def __init__(self, n_clusters: int = 8, alpha: float = 1, beta: float = 1, random_state: int = None,
                 engine: str = "python", verbose = False, clean_space: bool = True):
        if not isinstance(n_clusters, int):
            raise ValueError(f"Invalid n_clusters. It must be an int. A {type(n_clusters)} was passed.")
        if n_clusters < 2:
            raise ValueError(f"Invalid n_clusters. It must be an greater than 1. {n_clusters} was passed.")
        engines_options = ["matlab", "python"]
        if engine not in engines_options:
            raise ValueError(f"Invalid engine. Expected one of {engines_options}. {engine} was passed.")
        if (engine == "matlab") and (not matlabmodule_installed):
            raise ImportError(oct2py_module_error)

        self.n_clusters = n_clusters
        self.alpha = alpha
        self.beta = beta
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
            self._oc.eval("pkg load statistics")
            self._oc.eval("pkg load control")
            self._oc.warning("off", "Octave:possible-matlab-short-circuit-operator")


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
            transformed_Xs, observed_mod_indicator = self._processing_xs(Xs)

            w = tuple([self._oc.diag(missing_mod) for missing_mod in observed_mod_indicator.T])
            if self.random_state is not None:
                self._oc.rand('seed', self.random_state)
            u_0, v_0, b_0 = self._oc.newinit(transformed_Xs, w, self.n_clusters, len(transformed_Xs),
                                             self.random_state, nout=3)
            u, v, b, f, p, n = self._oc.DAIMC(transformed_Xs, w, u_0, v_0, b_0, None, self.n_clusters,
                                        len(transformed_Xs), {"afa": self.alpha, "beta": self.beta}, nout=6)
            b = [np.array(arr[0]) for arr in b]
            u = [np.array(arr[0]) for arr in u]

            if self.clean_space:
                self._clean_space()

        elif self.engine == "python":
            transformed_Xs, observed_mod_indicator = self._processing_xs(Xs)
            w = tuple([np.diag(missing_mod) for missing_mod in observed_mod_indicator.T])
            u_0, v_0, b_0 = self._new_init(transformed_Xs, w, self.n_clusters, len(transformed_Xs))
            u, v, b, f = self._daimc(transformed_Xs, w, u_0, v_0, b_0, self.n_clusters,
                                     len(transformed_Xs), {"afa": self.alpha, "beta": self.beta})

        model = KMeans(n_clusters= self.n_clusters, n_init="auto", random_state= self.random_state)
        self.labels_ = model.fit_predict(X= v)
        self.U_ = u
        self.embedding_ = v
        self.B_ = b

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


    @staticmethod
    def _new_init(X, W, n_clusters, viewNum, random_state=42):
        r"""
        It is the fist step of the DAIMC algorithm. The goal is to initiate the
        first variables.

        Parameters
        ----------
        X : list of array-likes objects
            - X length: viewNum
            - X[i] shape: (n_samples, n_features_i)
            A list of different mods.
        W : tuple of array
            - W length : viewNum
            - W[i] shape : (n_samples, n_samples)
        n_clusters : int, default=8
            The number of clusters to generate.
        viewNum : numbers of views (X length)
        random_state : int, default=None
            Determines the randomness. Use an int to make the randomness deterministic.

        Returns
        -------
        U : list of array :
            - U length: viewNum
            - U[i] shape: (n_features_i, n_clusters)
        V : array of shape (n_samples, n_clusters)
        B : list of array :
            - B length: n_clusters
            - B[i] shape: (n_features_i, n_clusters)
        """

        B = []
        U = []
        H = []
        XX = []

        for i in range(viewNum):
            item = np.diag(W[i])
            temp = np.where(item == 0)
            XX.append(X[i].copy())
            XX[i] = np.delete(XX[i], temp, axis=1)
            Mx = np.mean(XX[i], axis=1, keepdims=True)
            X[i][:, temp[0]] = np.tile(Mx, (1, len(temp)))

        sumH = 0
        for i in range(viewNum):
            d, n = X[i].shape
            kmeans = KMeans(n_clusters=n_clusters, n_init="auto", random_state=random_state)
            ilabels = kmeans.fit_predict(X[i].T)
            C = kmeans.cluster_centers_
            U.append(C.T + (0.1 * np.ones((d, n_clusters))))
            G = np.zeros((n, n_clusters))
            for j in range(1, n_clusters + 1):
                G[:, j - 1] = (ilabels == j * np.ones(shape=(n,)))
            H.append(G + 0.1 * np.ones((n, n_clusters)))
            sumH += H[i]

        V = sumH / viewNum
        Q = np.diag(np.matmul(np.ones((V.shape[0],)), V))
        V = np.matmul(V, np.linalg.inv(Q))

        U = [np.matmul(U[i], Q) for i in range(viewNum)]

        lamda = 1e-5
        for i in range(viewNum):
            d = U[i].shape[0]
            invI = np.diag(1.0 / np.diag(lamda * np.eye(d)))
            B.append(np.matmul((invI - np.matmul(np.matmul(np.matmul(np.matmul(invI, U[i]), np.linalg.inv(np.matmul(
                np.matmul(U[i].T, invI), U[i]) + np.eye(n_clusters))), U[i].T), invI)), U[i]))

        return U, V, B

    def _daimc(self, X, W, U, V, B, n_clusters, viewNum, options):
        r"""

        Parameters
        ----------
        X : list of array-likes objects
            - X length: n_mods
            - X[i] shape: (n_samples, n_features_i)
            A list of different views.
        W : tuple of array
            - W length : viewNum
            - W[i] shape : (n_samples, n_samples)
        U : list of array :
            - U length: viewNum
            - U[i] shape: (n_features_i, n_clusters)
        V : array of shape (n_samples, n_clusters)
        B : list of array :
            - B length: n_clusters
            - B[i] shape: (n_features_i, n_clusters)
        n_clusters : int, default=8
            The number of clusters to generate.
        viewNum : numbers of views (X length)
        options : options parameters

        Returns
        -------
        U : list of array :
            - U length: viewNum
            - U[i] shape: (n_features_i, n_clusters)
        V : array of shape (n_samples, n_clusters)
        B : list of array :
            - B length: n_clusters
            - B[i] shape: (n_features_i, n_clusters)
        F : float
            calculated from “ff” which is a loop stop condition
        """
        eta = 1e-10
        F = 0
        P = 0
        N = 0
        D = [np.zeros((B[i].shape[0], B[i].shape[0])) for i in range(viewNum)]

        for i in range(viewNum):
            for k in range(B[i].shape[0]):
                D[i][k, k] = 1 / np.sqrt(np.linalg.norm(B[i][k, :], 2) ** 2 + eta)

        time = 0
        f = 0
        while True:
            time = time + 1
            for i in range(viewNum):
                tmp1 = options['afa'] * np.matmul(B[i], B[i].T)
                tmp2 = np.matmul(np.matmul(V.T, W[i]), V)
                tmp3 = np.matmul(np.matmul(X[i], W[i]), V) + options['afa'] * B[i]
                U[i] = lyap(tmp1, tmp2, -tmp3)

            V = self._update_v_daimc(X, W, U, V, viewNum)
            Q = np.diag(np.matmul(np.ones(V.shape[0], ), V))
            V = np.matmul(V, np.linalg.inv(Q))
            for i in range(viewNum):
                U[i] = np.matmul(U[i], Q)
                invD = np.diag(1. / np.diag(0.5 * options['beta'] * D[i]))
                B[i] = np.matmul((invD - np.matmul(np.matmul(np.matmul(np.matmul(invD, U[i]), np.linalg.inv(np.matmul(
                    np.matmul(U[i].T, invD), U[i]) + np.eye(n_clusters))), U[i].T), invD)), U[i])
                for k in range(B[i].shape[0]):
                    D[i][k, k] = 1 / np.sqrt(np.linalg.norm(B[i][k, :], 2) ** 2 + eta)

            ff = 0
            for i in range(viewNum):
                tmp1 = np.matmul((X[i] - np.matmul(U[i], V.T)), W[i])
                tmp2 = np.matmul(B[i].T, U[i]) - np.eye(n_clusters)
                tmp3 = np.sum(1. / np.diag(D[i]))
                ff = ff + np.sum(np.sum(tmp1 ** 2)) + options['afa'] * np.sum(np.sum(tmp2 ** 2)) + options[
                    'beta'] * tmp3

            F += ff
            if (np.abs(ff - f) / f < 1e-4) or (np.abs(ff - f) > 1e100) | (time == 30):
                break
            f = ff

        return U, V, B, F

    @staticmethod
    def _update_v_daimc(X, W, U, V, viewNum):
        r"""
        Udpate V parameters.

        Parameters
        ----------
        X : list of array-likes objects
            - X length: n_mods
            - X[i] shape: (n_samples, n_features_i)
            A list of different views.
        W : tuple of array
            - W length : viewNum
            - W[i] shape : (n_samples, n_samples)
        U : list of array :
            - U length: viewNum
            - U[i] shape: (n_features_i, n_clusters)
        V : array of shape (n_samples, n_clusters)
        viewNum : numbers of views (X length)

        Returns
        -------
        V : array of shape (n_samples, n_clusters)
        """
        time = 0
        f = 0
        while True:
            time = time + 1
            sumVUUminus = 0
            sumVUUplus = 0
            sumXUminus = 0
            sumXUplus = 0

            for i in range(viewNum):
                XU = np.matmul(X[i].T, U[i])
                absXU = np.abs(XU)
                XUplus = (absXU + XU) / 2
                XUminus = (absXU - XU) / 2

                UU = np.matmul(U[i].T, U[i])
                absUU = np.abs(UU)
                UUplus = (absUU + UU) / 2
                UUminus = (absUU - UU) / 2

                sumXUminus = sumXUminus + np.matmul(W[i], XUminus)
                sumXUplus = sumXUplus + np.matmul(W[i], XUplus)

                sumVUUplus = sumVUUplus + np.matmul(np.matmul(W[i], V), UUplus)
                sumVUUminus = sumVUUminus + np.matmul(np.matmul(W[i], V), UUminus)

            V = V * np.sqrt((sumXUplus + sumVUUminus) / (np.maximum(sumXUminus + sumVUUplus, 1e-10)))
            ff = 0

            for i in range(viewNum):
                tmp = np.matmul((X[i] - np.matmul(U[i], V.T)), W[i])
                ff = ff + np.sum(np.sum(tmp ** 2))

            if (np.abs((ff - f) / f) < 1e-4) | (np.abs(ff - f) > 1e100) | (time == 30):
                break

            f = ff
        return V

    @staticmethod
    def _processing_xs(Xs):
        if isinstance(Xs[0], pd.DataFrame):
            transformed_Xs = [X.values for X in Xs]
        elif isinstance(Xs[0], np.ndarray):
            transformed_Xs = Xs
        observed_mod_indicator = get_observed_mod_indicator(transformed_Xs)
        transformed_Xs = simple_mod_imputer(transformed_Xs, value="zeros")
        transformed_Xs = [X.T for X in transformed_Xs]
        transformed_Xs = tuple(transformed_Xs)
        return transformed_Xs, observed_mod_indicator
