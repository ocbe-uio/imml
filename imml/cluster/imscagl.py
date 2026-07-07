# License: BSD-3-Clause

import os
from os.path import dirname
import numpy as np
import pandas as pd
from scipy.linalg import eigh
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

from ..impute import get_observed_mod_indicator
from ..utils import check_Xs_y
from ..preprocessing import remove_missing_samples_by_mod
from .. import octavemodule_installed, oct2py_module_error, Tensor

if octavemodule_installed:
    import oct2py


class IMSCAGL(BaseEstimator, ClusterMixin):
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
    engine : str, default='python'
        Engine to use for computing the model. Options are 'octave' or 'python'.
    verbose : bool, default=False
        Verbosity mode.
    clean_space : bool, default=True
        If engine is 'octave' and clean_space is True, the session will be closed after fitting the model.

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
                 rho: float = 1.1, random_state: int = None, engine: str = "python", verbose = False,
                 clean_space: bool = True):
        if not isinstance(n_clusters, int):
            raise ValueError(f"Invalid n_clusters. It must be an int. A {type(n_clusters)} was passed.")
        if n_clusters < 2:
            raise ValueError(f"Invalid n_clusters. It must be an greater than 1. {n_clusters} was passed.")
        engines_options = ["octave", "python"]
        if engine not in engines_options:
            raise ValueError(f"Invalid engine. Expected one of {engines_options}. {engine} was passed.")
        if (engine == "octave") and (not octavemodule_installed):
            raise ImportError(oct2py_module_error)
        if lambda1 <= 0:
            raise ValueError(f"Invalid lambda1. It must be a positive value. {lambda1} was passed.")
        if lambda2 <= 0:
            raise ValueError(f"Invalid lambda2. It must be a positive value. {lambda2} was passed.")
        if lambda3 <= 0:
            raise ValueError(f"Invalid lambda3. It must be a positive value. {lambda3} was passed.")
        if k <= 0:
            raise ValueError(f"Invalid k. It must be a positive value. {k} was passed.")
        if neighbor_mode not in ["KNN", "Supervised"]:
            raise ValueError(f"Invalid neighbor_mode. Expected one of ['KNN', 'Supervised']. {neighbor_mode} was passed.")
        if weight_mode not in ["Binary", "Cosine", "HeatKernel"]:
            raise ValueError(f"Invalid weight_mode. Expected one of ['Binary', 'Cosine', 'HeatKernel']. {weight_mode} was passed.")
        if max_iter <= 0:
            raise ValueError(f"Invalid max_iter. It must be a positive value. {max_iter} was passed.")
        if miu <= 0:
            raise ValueError(f"Invalid miu. It must be a positive value. {miu} was passed.")
        if rho <= 0:
            raise ValueError(f"Invalid rho. It must be a positive value. {rho} was passed.")
        if not isinstance(clean_space, bool):
            raise ValueError(f"Invalid clean_space. It must be a boolean. {clean_space} was passed.")

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

        if self.engine == "octave":
            octave_folder = dirname(__file__)
            octave_folder = os.path.join(octave_folder, "_" + (os.path.basename(__file__).split(".")[0]))
            self._octave_folder = octave_folder
            octave_files = [x for x in os.listdir(octave_folder) if x.endswith(".m")]
            self._oc = oct2py.Oct2Py(temp_dir= octave_folder)
            for octave_file in octave_files:
                with open(os.path.join(octave_folder, octave_file)) as f:
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
        Xs = check_Xs_y(Xs, ensure_all_finite='allow-nan')

        transformed_Xs, selection_mats, observed_indices = self._process_xs(
            Xs, return_selection_mats=self.engine == "octave"
        )

        if self.engine=="octave":
            if self.random_state is not None:
                self._oc.rand('seed', self.random_state)
            F = self._oc.IMSAGL(tuple(transformed_Xs), tuple(selection_mats), self.n_clusters, self.lambda1, self.lambda2, self.lambda3,
                          self.miu, self.rho, self.max_iter,
                          {"NeighborMode": self.neighbor_mode, "WeightMode": self.weight_mode, "k": self.k})

            if self.clean_space:
                self._clean_space()

        elif self.engine == "python":
            F = self._imsagl(transformed_Xs, observed_indices, n_samples=len(Xs[0]))

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
        [os.remove(os.path.join(self._octave_folder, x)) for x in ["reader.mat", "writer.mat"]
         if os.path.exists(os.path.join(self._octave_folder, x))]
        self._oc.exit()
        del self._oc
        return None

    def _process_xs(self, Xs, return_selection_mats):
        if not isinstance(Xs[0], pd.DataFrame):
            if isinstance(Xs[0], Tensor):
                Xs = [pd.DataFrame(X.detach().cpu().numpy()) for X in Xs]
            else:
                Xs = [pd.DataFrame(X) for X in Xs]
        observed_mod_indicator = get_observed_mod_indicator(Xs=Xs)
        transformed_Xs = remove_missing_samples_by_mod(Xs=Xs)
        observed_indices = [
            np.flatnonzero(np.asarray(samples, dtype=bool))
            for _, samples in observed_mod_indicator.items()
        ]
        selection_mats = None
        if return_selection_mats:
            selection_mats = [pd.DataFrame(np.eye(len(X)), index=X.index, columns=X.index) for X in Xs]
            selection_mats = [eye.loc[samples, :].values.astype(float)
                              for eye, (_, samples) in zip(selection_mats, observed_mod_indicator.items())]
        transformed_Xs = [X.T.values.astype(float) for X in transformed_Xs]
        transformed_Xs = [np.nan_to_num(X, nan=0.0, posinf=1e5, neginf=-1e5) for X in transformed_Xs]
        return transformed_Xs, selection_mats, observed_indices

    def _imsagl(self, Xs_T, observed_indices, n_samples):
        n_mods = len(Xs_T)
        z_ini = [self._construct_w(X.T) for X in Xs_T]
        inv_xx = [
            np.linalg.inv(X.T @ X + 2 * np.eye(X.shape[1]))
            for X in Xs_T
        ]

        F = self._solve_f(z_ini, observed_indices, n_samples)
        Z = [Z_iv.copy() for Z_iv in z_ini]
        S = [Z_iv.copy() for Z_iv in z_ini]
        W = [Z_iv.copy() for Z_iv in z_ini]
        E = [np.zeros_like(X) for X in Xs_T]
        C1 = [np.zeros_like(X) for X in Xs_T]
        C2 = [np.zeros_like(Z_iv) for Z_iv in Z]
        C3 = [np.zeros_like(Z_iv) for Z_iv in Z]

        miu = self.miu
        obj = []
        U = None
        for _ in range(self.max_iter):
            f_sum = sum(F_iv @ F_iv.T for F_iv in F)
            f_sum = np.nan_to_num(f_sum, nan=0.0, posinf=1e5, neginf=-1e5)
            U = self._top_eigenvectors(f_sum, self.n_clusters, largest=True)

            f_sum = np.zeros_like(f_sum)
            obj_l1 = 0.0
            obj_constraints = 0.0
            norm_x = 0.0
            for i in range(n_mods):
                G1 = Xs_T[i] - E[i] + C1[i] / miu
                G2 = S[i] - C2[i] / miu
                G3 = W[i] - C3[i] / miu
                Z[i] = inv_xx[i] @ (Xs_T[i].T @ G1 + G2 + G3)

                obs_idx = observed_indices[i]
                projected = F[i][obs_idx]
                Q = self._squared_l2_distance(projected.T, projected.T)
                M = Z[i] + C3[i] / miu
                W_candidate = M - 0.5 * self.lambda1 * Q / miu
                W[i] = self._project_rows_simplex_without_diagonal(W_candidate)

                temp = Z[i] + C2[i] / miu
                U_svd, singular_values, Vt_svd = np.linalg.svd(temp, full_matrices=False)
                singular_values = np.nan_to_num(singular_values, nan=0.0, posinf=0.0, neginf=0.0)
                keep = np.flatnonzero(singular_values > 1 / miu)
                if keep.size >= 1:
                    rank = keep.size
                    shrunk = singular_values[:rank] - 1 / miu
                else:
                    rank = 1
                    shrunk = np.zeros(1)
                S[i] = U_svd[:, :rank] @ np.diag(shrunk) @ Vt_svd[:rank, :]

                WW = (np.abs(W[i]) + np.abs(W[i].T)) * 0.5
                LL = np.diag(WW.sum(axis=1)) - WW
                M = self.lambda3 * (U @ U.T)
                M[np.ix_(obs_idx, obs_idx)] -= self.lambda1 * LL
                M = np.nan_to_num(M, nan=0.0, posinf=1e5, neginf=-1e5)
                F[i] = self._top_eigenvectors(M, self.n_clusters, largest=True)
                f_sum += F[i] @ F[i].T

                temp = Xs_T[i] - Xs_T[i] @ Z[i] + C1[i] / miu
                threshold = self.lambda2 / miu
                E[i] = np.maximum(0.0, temp - threshold) + np.minimum(0.0, temp + threshold)

                leq1 = Xs_T[i] - Xs_T[i] @ Z[i] - E[i]
                leq2 = Z[i] - S[i]
                leq3 = Z[i] - W[i]
                C1[i] += miu * leq1
                C2[i] += miu * leq2
                C3[i] += miu * leq3

                obj_l1 += (
                    np.sum(np.abs(shrunk))
                    + self.lambda1 * np.trace(F[i][obs_idx].T @ LL @ F[i][obs_idx])
                    + self.lambda2 * np.sum(np.abs(E[i]))
                )
                obj_constraints += (
                    np.linalg.norm(leq1, "fro") ** 2
                    + np.linalg.norm(leq2, "fro") ** 2
                    + np.linalg.norm(leq3, "fro") ** 2
                )
                norm_x += np.linalg.norm(Xs_T[i], "fro") ** 2

            value = (obj_l1 + obj_constraints + self.lambda3 * (n_mods * self.n_clusters - np.trace(U.T @ f_sum @ U))) / norm_x
            obj.append(value)
            miu = min(self.rho * miu, 1e8)
            if len(obj) > 2 and abs(obj[-1] - obj[-2]) < 1e-7:
                break

        if U is None:
            U = self._top_eigenvectors(sum(F_iv @ F_iv.T for F_iv in F), self.n_clusters, largest=True)
        return self._normalize_rows(U)

    def _solve_f(self, Z, observed_indices, n_samples):
        F = []
        for Z_iv, obs_idx in zip(Z, observed_indices):
            W_iv = (np.abs(Z_iv) + np.abs(Z_iv.T)) / 2
            laplacian = np.diag(W_iv.sum(axis=1)) - W_iv
            M = np.zeros((n_samples, n_samples), dtype=laplacian.dtype)
            M[np.ix_(obs_idx, obs_idx)] = laplacian
            M = np.nan_to_num(M, nan=0.0, posinf=1e5, neginf=-1e5)
            eigvals, eigvecs = eigh((M + M.T) / 2)
            positive = np.flatnonzero(eigvals > 1e-6)
            if positive.size > self.n_clusters:
                F.append(eigvecs[:, positive[:self.n_clusters]])
            else:
                F.append(eigvecs[:, :self.n_clusters])
        return F

    def _construct_w(self, fea):
        n_samples = fea.shape[0]
        n_neighbors = min(self.k + 1, n_samples)
        W = np.zeros((n_samples, n_samples))

        if n_samples == 0:
            return W

        mode = self.weight_mode.lower()
        if self.neighbor_mode.lower() != "knn":
            raise ValueError("Supervised neighbor_mode requires labels and is not supported by IMSCAGL.fit.")

        nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm="brute", metric="euclidean").fit(fea)
        distances, indices = nbrs.kneighbors(fea)
        rows = np.repeat(np.arange(n_samples), n_neighbors)
        cols = indices.ravel()

        if mode == "binary":
            weights = np.ones(rows.size)
        elif mode == "heatkernel":
            D = self._euclidean_distances(fea, squared=False)
            t = np.mean(D)
            if t <= 0:
                t = 1.0
            weights = np.exp(-(distances.ravel() ** 2) / (2 * t ** 2))
        elif mode == "cosine":
            norms = np.linalg.norm(fea, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            fea_norm = fea / norms
            weights = np.einsum("ij,ij->i", fea_norm[rows], fea_norm[cols])
        else:
            raise ValueError(f"Invalid weight_mode. Expected one of ['Binary', 'Cosine', 'HeatKernel']. {self.weight_mode} was passed.")

        W[rows, cols] = weights
        np.fill_diagonal(W, 0.0)
        return np.maximum(W, W.T)

    @staticmethod
    def _top_eigenvectors(matrix, n_components, largest):
        matrix = (matrix + matrix.T) / 2
        if n_components < matrix.shape[0]:
            if largest:
                subset = [matrix.shape[0] - n_components, matrix.shape[0] - 1]
            else:
                subset = [0, n_components - 1]
            eigvals, eigvecs = eigh(matrix, subset_by_index=subset)
        else:
            eigvals, eigvecs = eigh(matrix)
        order = np.argsort(eigvals)
        if largest:
            order = order[::-1]
        return np.real(eigvecs[:, order[:n_components]])

    @staticmethod
    def _project_rows_simplex_without_diagonal(matrix):
        n_rows, n_cols = matrix.shape
        if n_rows != n_cols:
            raise ValueError("Simplex projection expects a square matrix.")
        if n_rows <= 1:
            return np.zeros_like(matrix)

        off_diag = ~np.eye(n_rows, dtype=bool)
        values = matrix[off_diag].reshape(n_rows, n_cols - 1)
        sorted_values = np.sort(values, axis=1)[:, ::-1]
        cssv = np.cumsum(sorted_values, axis=1) - 1.0
        ind = np.arange(1, n_cols)
        cond = sorted_values - cssv / ind > 0
        n_positive = cond.sum(axis=1)
        if np.any(n_positive == 0):
            projected = np.full_like(values, 1.0 / (n_cols - 1))
            result = np.zeros_like(matrix)
            result[off_diag] = projected.ravel()
            return result
        rho = n_positive - 1
        theta = cssv[np.arange(n_rows), rho] / (rho + 1)
        projected = np.maximum(values - theta[:, None], 0.0)

        result = np.zeros_like(matrix)
        result[off_diag] = projected.ravel()
        return result

    @staticmethod
    def _project_simplex(v, k=1.0):
        v = np.asarray(v, dtype=float)
        if v.size == 0:
            return v.copy()
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u) - k
        ind = np.arange(1, v.size + 1)
        cond = u - cssv / ind > 0
        if not np.any(cond):
            return np.full_like(v, k / v.size)
        rho = ind[cond][-1]
        theta = cssv[cond][-1] / rho
        return np.maximum(v - theta, 0.0)

    @staticmethod
    def _squared_l2_distance(a, b):
        if a.shape[0] == 1:
            a = np.vstack([a, np.zeros((1, a.shape[1]))])
            b = np.vstack([b, np.zeros((1, b.shape[1]))])
        d = np.sum(a * a, axis=0)[:, None] + np.sum(b * b, axis=0)[None, :] - 2 * (a.T @ b)
        return np.maximum(np.real(d), 0.0)

    @staticmethod
    def _euclidean_distances(fea, squared=False):
        d = np.sum(fea * fea, axis=1)[:, None] + np.sum(fea * fea, axis=1)[None, :] - 2 * (fea @ fea.T)
        d = np.maximum(d, 0.0)
        if not squared:
            d = np.sqrt(d)
        return np.maximum(d, d.T)

    @staticmethod
    def _normalize_rows(matrix):
        norms = np.sqrt(np.sum(matrix * matrix, axis=1, keepdims=True))
        norms = np.where(norms == 0, 1.0, norms)
        return matrix / norms
