# License: BSD-3-Clause

import os
from os.path import dirname
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.impute import SimpleImputer

from ..impute import get_observed_mod_indicator
from ..preprocessing import MMTransformer
from ..utils import check_Xs_y
from .. import octavemodule_installed, oct2py_module_error, Tensor

if octavemodule_installed:
    import oct2py


class OPIMC(BaseEstimator, ClusterMixin):
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
    batch_size : int, default=250
        Size of the chunk.
    random_state : int, default=None
        Determines the randomness. Use an int to make the randomness deterministic.
    engine : str, default='python'
        Engine to use for computing the model. Current options are 'python' or 'octave'.
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
                 tol: float = 1e-6, batch_size: int = 250, random_state:int = None, engine: str ="python",
                 verbose = False, clean_space: bool = True):
        if not isinstance(n_clusters, int):
            raise ValueError(f"Invalid n_clusters. It must be an int. A {type(n_clusters)} was passed.")
        if n_clusters < 2:
            raise ValueError(f"Invalid n_clusters. It must be an greater than 1. {n_clusters} was passed.")
        engines_options = ["octave", "python"]
        if engine not in engines_options:
            raise ValueError(f"Invalid engine. Expected one of {engines_options}. {engine} was passed.")
        if (engine == "octave") and (not octavemodule_installed):
            raise ImportError(oct2py_module_error)
        if not isinstance(max_iter, int):
            raise ValueError(f"Invalid max_iter. It must be an int. A {type(max_iter)} was passed.")
        if max_iter < 1:
            raise ValueError(f"Invalid max_iter. It must be an greater than 0. {max_iter} was passed.")

        self.n_clusters = n_clusters
        self.alpha = alpha
        self.num_passes = num_passes
        self.max_iter = max_iter
        self.tol = tol
        self.batch_size = batch_size
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
        transformed_Xs, observed_mod_indicator = self._processing_xs(Xs)

        if self.engine == "octave":
            options = {"batch_size": self.batch_size, "k": self.n_clusters, "maxiter": self.max_iter,
                       "tol": self.tol, "pass": self.num_passes, "alpha": self.alpha}
            if self.random_state is not None:
                self._oc.rand('seed', self.random_state)
            labels, embeddings = self._oc.OPIMC(transformed_Xs, options, observed_mod_indicator, nout=2)

            if self.clean_space:
                self._clean_space()

            labels = labels[:, 0]

        elif self.engine == "python":
            self.rng = np.random.default_rng(self.random_state)
            labels, embeddings = self._opimc_python(list(transformed_Xs), np.array(observed_mod_indicator, dtype=float))

        labels = pd.factorize(labels)[0]
        self.labels_ = labels
        self.embedding_ = embeddings

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


    @staticmethod
    def _processing_xs(Xs):
        if isinstance(Xs[0], pd.DataFrame):
            transformed_Xs = [X.values for X in Xs]
        elif isinstance(Xs[0], np.ndarray):
            transformed_Xs = Xs
        elif isinstance(Xs[0], Tensor):
            transformed_Xs = [X.numpy() for X in Xs]
        else:
            transformed_Xs = list(Xs)
        observed_mod_indicator = get_observed_mod_indicator(transformed_Xs)
        imputer = MMTransformer(transformer=SimpleImputer(strategy="constant", fill_value=0))
        transformed_Xs = imputer.fit_transform(transformed_Xs)
        transformed_Xs = tuple(X.T for X in transformed_Xs)
        return transformed_Xs, observed_mod_indicator


    @staticmethod
    def _update_v(X_blk, w_blk, U):
        block_len = X_blk[0].shape[1]
        k = U[0].shape[1]
        D = np.zeros((block_len, k))
        for Xi, wi, Ui in zip(X_blk, w_blk, U):
            bb = np.einsum('ij,ij->j', Ui, Ui)        # (k,) squared column norms of Ui
            D += wi[:, None] * (bb - 2.0 * (Xi.T @ Ui))
        labels = D.argmin(axis=1)
        V = np.zeros((block_len, k))
        V[np.arange(block_len), labels] = 1.0
        return V, D


    def _compute_objective(self, R, T, X_blk, w_blk, U, V):
        loss = 0.0
        for Ri, Ti, Xi, wi, Ui in zip(R, T, X_blk, w_blk, U):
            wV = wi[:, None] * V                       # (block_len, k)
            tmp1 = Ri + Xi @ V                         # (n_features_i, k)
            tmp2 = Ti + V.T @ wV                       # (k, k)  symmetric
            loss += (-2.0 * np.sum(Ui * tmp1)          # -2 * trace(Ui.T @ tmp1)
                     + np.sum((Ui.T @ Ui) * tmp2)      # trace(Ui.T @ Ui @ tmp2)
                     + self.alpha * np.sum(Ui ** 2))   # alpha * ||Ui||_F^2
        return loss


    def _opimc_python(self, Xs, ind):
        num_mods = len(Xs)
        total = Xs[0].shape[1]
        num_block = int(np.ceil(total / self.batch_size))

        W = [ind[:, i] for i in range(num_mods)]
        U = [self.rng.random((Xi.shape[0], self.n_clusters)) for Xi in Xs]
        R = [np.zeros((Xi.shape[0], self.n_clusters)) for Xi in Xs]
        T = [np.zeros((self.n_clusters, self.n_clusters)) for _ in range(num_mods)]

        label_total = np.zeros((self.num_passes, total), dtype=int)
        D_full = np.zeros((total, self.n_clusters))

        for pass_idx in range(self.num_passes):
            if pass_idx > 0:
                label_total[pass_idx] = label_total[pass_idx - 1]

            for block_idx in range(num_block):
                start = block_idx * self.batch_size
                end = min(start + self.batch_size, total)
                block_len = end - start

                X_blk = [Xi[:, start:end] for Xi in Xs]
                w_blk = [Wi[start:end] for Wi in W]

                if pass_idx == 0:
                    if block_idx == 0:
                        U[:] = [self.rng.random(Ui.shape) for Ui in U]
                        V = self.rng.random((block_len, self.n_clusters))
                        V /= V.sum(axis=1, keepdims=True)
                    else:
                        V, _ = self._update_v(X_blk, w_blk, U)
                    label_total[0, start:end] = V.argmax(axis=1)
                else:
                    V = np.zeros((block_len, self.n_clusters))
                    V[np.arange(block_len), label_total[pass_idx, start:end]] = 1.0

                log_out = self._compute_objective(R, T, X_blk, w_blk, U, V)

                for iter_count in range(self.max_iter):
                    if pass_idx != 0 and iter_count == 0:
                        V_pre = np.zeros((block_len, self.n_clusters))
                        V_pre[np.arange(block_len), label_total[pass_idx - 1, start:end]] = 1.0
                        for Ti, Ri, Xi, wi in zip(T, R, X_blk, w_blk):
                            Ti -= V_pre.T @ (wi[:, None] * V_pre)
                            Ri -= Xi @ V_pre

                    for i, (Ti, Ri, Xi, wi, Ui) in enumerate(zip(T, R, X_blk, w_blk, U)):
                        wV = wi[:, None] * V
                        VtWV = V.T @ wV
                        TpVtWV = Ti + VtWV
                        U_new = np.linalg.solve(
                            (TpVtWV + self.alpha * np.eye(self.n_clusters)).T,
                            (Ri + Xi @ V).T,
                        ).T
                        check = VtWV if (pass_idx == 0 and block_idx == 0) else TpVtWV
                        zero_cols = np.where(np.diag(check) == 0)[0]
                        if len(zero_cols):
                            if pass_idx == 0 and block_idx == 0:
                                U_new[:, zero_cols] = Xi.mean(axis=1, keepdims=True)
                            else:
                                U_new[:, zero_cols] = Ui[:, zero_cols]
                        U[i] = U_new

                    V, D = self._update_v(X_blk, w_blk, U)
                    label_total[pass_idx, start:end] = V.argmax(axis=1)

                    log_out_new = self._compute_objective(R, T, X_blk, w_blk, U, V)
                    if log_out != 0.0 and abs((log_out_new - log_out) / log_out) < self.tol:
                        break
                    log_out = log_out_new

                D_full[start:end] = D

                for Ri, Ti, Xi, wi in zip(R, T, X_blk, w_blk):
                    wV = wi[:, None] * V
                    Ri += Xi @ V
                    Ti += V.T @ wV

        return label_total[self.num_passes - 1], D_full
