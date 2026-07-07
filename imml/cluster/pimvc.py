# License: BSD-3-Clause

import os
from os.path import dirname
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

from ..impute import get_observed_mod_indicator
from ..utils import check_Xs_y
from .. import octavemodule_installed, oct2py_module_error, Tensor

if octavemodule_installed:
    import oct2py


class PIMVC(BaseEstimator, ClusterMixin):
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
    neighbor_mode : str, default='knn'
        Indicates how to construct the graph. Options are 'knn' (default), and 'supervised'.
    weight_mode : str, default='binary'
        Indicates how to assign weights for each edge in the graph. Options are 'binary' (default), 'cosine' and 'heatkernel'.
    max_iter : int, default=100
        Maximum number of iterations.
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
                 neighbor_mode: str = 'knn', weight_mode: str = 'binary', max_iter: int = 100,
                 random_state: int = None, engine: str = "python", verbose = False, clean_space: bool = True):
        if not isinstance(n_clusters, int):
            raise ValueError(f"Invalid n_clusters. It must be an int. A {type(n_clusters)} was passed.")
        if n_clusters < 2:
            raise ValueError(f"Invalid n_clusters. It must be an greater than 1. {n_clusters} was passed.")
        engines_options = ["octave", "python"]
        if engine not in engines_options:
            raise ValueError(f"Invalid engine. Expected one of {engines_options}. {engine} was passed.")
        if (engine == "octave") and (not octavemodule_installed):
            raise ImportError(oct2py_module_error)
        if lamb <= 0:
            raise ValueError(f"Invalid lamb. It must be a positive value. {lamb} was passed.")
        if k <= 0:
            raise ValueError(f"Invalid k. It must be a positive value. {k} was passed.")
        if neighbor_mode not in ["knn", "supervised"]:
            raise ValueError(f"Invalid neighbor_mode. Expected one of ['knn', 'supervised']. {neighbor_mode} was passed.")
        if weight_mode not in ["binary", "cosine", "heatkernel"]:
            raise ValueError(f"Invalid weight_mode. Expected one of ['binary', 'cosine', 'heatkernel']. {weight_mode} was passed.")
        if max_iter <= 0:
            raise ValueError(f"Invalid max_iter. It must be a positive value. {max_iter} was passed.")
        if not isinstance(clean_space, bool):
            raise ValueError(f"Invalid clean_space. It must be a boolean. {clean_space} was passed.")

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

        try:
            assert self.n_clusters <= min([X.shape[1] for X in Xs])
        except AssertionError:
            raise ValueError(f"n_clusters ({self.n_clusters}) should be smaller or equal to " +
                             f"the smallest n_features_i ({min([X.shape[1] for X in Xs])}).")

        transformed_Xs, observed_mod_indicator = self._process_xs(Xs)

        if self.engine == "octave":
            if self.random_state is not None:
                self._oc.rand('seed', self.random_state)
            v, loss = self._oc.PIMVC(tuple(transformed_Xs), self.n_clusters, observed_mod_indicator, self.lamb,
                                      self.beta, self.max_iter, {"NeighborMode": self.neighbor_mode,
                                                                  "WeightMode": self.weight_mode,
                                                                  "k": self.k}, nout=2)
            if self.clean_space:
                self._clean_space()

        elif self.engine == "python":
            v, loss = self._pimvc(transformed_Xs, observed_mod_indicator.astype(float))

        model = KMeans(n_clusters=self.n_clusters, n_init="auto", random_state=self.random_state)
        v = v.T
        self.labels_ = model.fit_predict(X=v)
        self.embedding_ = v
        self.loss_ = np.ravel(loss)
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
        [os.remove(os.path.join(self._octave_folder, x)) for x in ["reader.mat", "writer.mat"]]
        self._oc.exit()
        del self._oc
        return None


    def _process_xs(self, Xs):
        if isinstance(Xs[0], pd.DataFrame):
            transformed_Xs = [X.T.values for X in Xs]
            observed_mod_indicator = get_observed_mod_indicator(Xs).values
        elif isinstance(Xs[0], np.ndarray):
            transformed_Xs = [X.T for X in Xs]
            observed_mod_indicator = get_observed_mod_indicator(Xs)
        elif isinstance(Xs[0], Tensor):
            transformed_Xs = [X.T.numpy() for X in Xs]
            observed_mod_indicator = get_observed_mod_indicator(Xs).numpy()
        return transformed_Xs, observed_mod_indicator


    def _pimvc(self, Xs_T, ind_folds):
        n_mods = len(Xs_T)
        n_samples = ind_folds.shape[0]

        norm_x = 0.0
        sum_obs_counts = np.zeros(n_samples)          # linshi_GG
        sum_laplacians = np.zeros((n_samples, n_samples))  # linshi_LS

        Piv = []        # per-view projection matrices, (dim, n_features_v)
        XG = []         # X_obs @ G.T,  (n_features_v, n_samples)
        St2_list = []   # whitening matrices, (n_features_v, n_features_v)
        St3_list = []   # St2 @ X_obs @ G.T, (n_features_v, n_samples)
        G_list = []     # selection matrices, (n_samples, n_observed_v)
        X_obs_list = [] # observed data, (n_features_v, n_observed_v)

        for iv in range(n_mods):
            obs_idx = np.where(ind_folds[:, iv] == 1)[0]
            n_obs = len(obs_idx)

            # Remove missing sample columns: X{iv}(:, ind_0) = []
            X_iv_obs = Xs_T[iv][:, obs_idx].copy()
            X_obs_list.append(X_iv_obs)
            n_features = X_iv_obs.shape[0]

            # G{iv} = diag(ind_folds[:,iv]) with missing columns deleted
            # → (n_samples, n_obs) matrix: G[obs_idx[j], j] = 1
            G_iv = np.zeros((n_samples, n_obs))
            G_iv[obs_idx, np.arange(n_obs)] = 1.0
            G_list.append(G_iv)

            # St2{iv} = (X_obs @ X_obs.T + lamb * I)^{-0.5}
            St = X_iv_obs @ X_iv_obs.T + self.lamb * np.eye(n_features)
            eigvals, eigvecs = np.linalg.eigh(St)
            St2_iv = eigvecs @ np.diag(eigvals ** (-0.5)) @ eigvecs.T
            St2_list.append(St2_iv)

            # St3{iv} = St2{iv} * X_obs * G{iv}'
            St3_list.append(St2_iv @ X_iv_obs @ G_iv.T)

            # linshi_GG += ind_folds[:,iv]
            sum_obs_counts += ind_folds[:, iv]

            # Construct KNN graph on observed samples, then embed into full space
            fea = X_iv_obs.T  # (n_obs, n_features)
            W_obs = self._construct_w(fea)  # (n_obs, n_obs)
            W_graph = (W_obs + W_obs.T) * 0.5
            W_full = G_iv @ W_graph @ G_iv.T  # (n_samples, n_samples)
            sum_laplacians += np.diag(W_full.sum(axis=1)) - W_full

            # PCA init: Piv{iv} = PCA1(X_obs.T, options_dim)'
            eigvec = self._pca_init(X_iv_obs, self.n_clusters)  # (n_features, dim)
            Piv.append(eigvec.T)                     # (dim, n_features)

            # XG{iv} = X_obs * G{iv}'
            XG.append(X_iv_obs @ G_iv.T)  # (n_features, n_samples)

            norm_x += np.linalg.norm(X_iv_obs, 'fro') ** 2

        # inv_GS = inv(linshi_LS * beta + diag(linshi_GG))
        inv_GS = np.linalg.inv(sum_laplacians * self.beta + np.diag(sum_obs_counts))

        # Y = rand(dim, numInst)
        rng = np.random.default_rng(self.random_state)
        Y = rng.random((self.n_clusters, n_samples))

        obj = []
        for iter_idx in range(self.max_iter):
            # Update Y: Y = (sum_iv Piv_iv @ XG_iv) @ inv_GS
            H1 = sum(Piv[iv] @ XG[iv] for iv in range(n_mods))
            Y = H1 @ inv_GS

            # Update Piv: closed-form via SVD
            for iv in range(n_mods):
                M = St3_list[iv] @ Y.T  # (n_features, dim)
                M = np.nan_to_num(M)
                # svd(M', 'econ') where M' is (dim, n_features)
                U, _, Vt = np.linalg.svd(M.T, full_matrices=False)
                U = np.nan_to_num(U)
                Vt = np.nan_to_num(Vt)
                # Piv{iv} = U * V' * St2{iv}  (V' in MATLAB = Vt in numpy)
                Piv[iv] = U @ Vt @ St2_list[iv]

            # Compute objective
            obj_val = 0.0
            for iv in range(n_mods):
                diff = Piv[iv] @ X_obs_list[iv] - Y @ G_list[iv]
                obj_val += np.sum(diff ** 2) + self.lamb * np.sum(Piv[iv] ** 2)
            obj_val = (obj_val + self.beta * np.trace(Y @ sum_laplacians @ Y.T)) / norm_x
            obj.append(obj_val)

            if iter_idx > 2 and abs(obj[-1] - obj[-2]) < 1e-5:
                break

        return Y, np.array(obj)


    @staticmethod
    def _pca_init(X_obs, dim):
        data = X_obs.T           # (n_obs, n_features), rows are samples
        data = data - data.mean(axis=0)
        X = data.T               # (n_features, n_obs) — argument passed to mySVD

        n_smp, m_fea = X.shape   # n_smp=n_features, m_fea=n_obs

        if m_fea / n_smp > 1.0713:
            # Case 1: eig(X @ X.T)
            ddata = X @ X.T
            ddata = (ddata + ddata.T) * 0.5      # enforce numerical symmetry
            eigvals, U = np.linalg.eigh(ddata)   # ascending order
            idx = np.argsort(-eigvals)            # sort descending
            eigvals, U = eigvals[idx], U[:, idx]
        else:
            # Case 2: eig(X.T @ X), then U = X @ V / sqrt(eigvals)
            ddata = X.T @ X
            ddata = (ddata + ddata.T) * 0.5
            eigvals, V = np.linalg.eigh(ddata)
            idx = np.argsort(-eigvals)
            eigvals, V = eigvals[idx], V[:, idx]
            U = X @ (V / np.sqrt(np.maximum(eigvals, 0)))

        # Remove near-zero eigenvectors (matching MATLAB's eigIdx filter)
        max_eig = np.max(np.abs(eigvals))
        keep = np.abs(eigvals) / max_eig >= 1e-10
        U = U[:, keep]

        return U[:, :dim]        # (n_features, dim)


    def _construct_w(self, fea):
        n = fea.shape[0]

        # Find k+1 neighbours (including self); octave code does the same then
        # removes self-connections via W = W - diag(diag(W)).
        nbrs = NearestNeighbors(n_neighbors=self.k + 1, algorithm='brute', metric='euclidean').fit(fea)
        distances, indices = nbrs.kneighbors(fea)

        rows = np.repeat(np.arange(n), self.k + 1)
        cols = indices.ravel()
        W = np.zeros((n, n))

        if self.weight_mode == 'binary':
            W[rows, cols] = 1.0

        elif self.weight_mode == 'heatkernel':
            # Compute t as mean of pairwise squared distances, matching MATLAB's EuDist2 + mean
            if n > 3000:
                sub = fea[:3000]
                D2 = np.maximum(
                    np.sum(sub ** 2, axis=1, keepdims=True)
                    + np.sum(sub ** 2, axis=1)
                    - 2 * sub @ sub.T, 0)
            else:
                D2 = np.maximum(
                    np.sum(fea ** 2, axis=1, keepdims=True)
                    + np.sum(fea ** 2, axis=1)
                    - 2 * fea @ fea.T, 0)
            t = D2.mean()
            W[rows, cols] = np.exp(-distances.ravel() ** 2 / (2 * t ** 2))

        elif self.weight_mode == 'cosine':
            norms = np.linalg.norm(fea, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1e-12, norms)
            fea_norm = fea / norms
            W[rows, cols] = np.einsum('ij,ij->i', fea_norm[rows], fea_norm[cols])

        # Remove self-connections and symmetrise (W = max(W, W'))
        np.fill_diagonal(W, 0.0)
        W = np.maximum(W, W.T)
        return W
