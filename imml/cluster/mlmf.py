# License: BSD-3-Clause

import os
from os.path import dirname
import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import KMeans

from ..utils import check_Xs_y
from .. import octavemodule_installed, oct2py_module_error

if octavemodule_installed:
    import oct2py


class MLMF(BaseEstimator, ClusterMixin):
    r"""
    Multi-layer matrix factorization (MLMF). [#mlmfpaper]_ [#mlmfcode]_

    MLMF learns a shared representation for multi-modal clustering using linear or nonlinear multi-layer matrix
    factorization.

    Parameters
    ----------
    n_clusters : int, default=8
        The number of clusters to generate.
    factorization : str, default="linear"
        Type of factorization that will be applied. Available options are 'linear' and 'nonlinear'.
    lambda1 : float, default=1.
        Regularization parameter for the consensus representation.
    lambda2 : float, default=1.
        Regularization parameter for aligning modality-specific and consensus representations.
    layers : list, default=None
        Layer dimensions. If None, defaults to ``[max(2 * n_clusters, n_clusters + 1), n_clusters]``.
    max_iter : int, default=100
        Maximum number of semi-NMF initialization iterations.
    tol : float, default=1e-5
        Tolerance used by semi-NMF initialization.
    update_h : bool, default=True
        Whether to update H matrices during optimization.
    update_last_h : bool, default=True
        Whether to update the last H matrix during optimization.
    update_z : bool, default=True
        Whether to update Z matrices in the linear factorization.
    nonlinearity : str, default="tanh"
        Nonlinearity used when ``factorization='nonlinear'``. Options are 'tanh', 'square', 'sigmoid', and 'softplus'.
    init_z : object, default=None
        Optional initial Z matrices.
    init_h : object, default=None
        Optional initial H matrices.
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
    loss_ : array-like of shape (n_iter_,)
        Values of the loss function.
    n_iter_ : int
        Number of iterations.

    References
    ----------
    .. [#mlmfpaper] Yingxuan Ren, Fengtao Ren, Bo Yang, Multi-layer matrix factorization for cancer subtyping
                    using full and partial multi-omics dataset, Briefings in Bioinformatics, Volume 26, Issue 5,
                    September 2025, bbaf448, https://doi.org/10.1093/bib/bbaf448.
    .. [#mlmfcode] https://github.com/renyingxuan/MLMF/tree/main

    Example
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from imml.cluster import MLMF
    >>> Xs = [pd.DataFrame(np.random.default_rng(42).random((20, 10))) for i in range(3)]
    >>> estimator = MLMF(n_clusters = 2)
    >>> labels = estimator.fit_predict(Xs)
    """

    def __init__(self, n_clusters: int = 8, factorization: str = "linear", lambda1: float = 1.,
                 lambda2: float = 1., layers: list = None, max_iter: int = 100, tol: float = 1e-5,
                 update_h: bool = True, update_last_h: bool = True, update_z: bool = True,
                 nonlinearity: str = "tanh", init_z=None, init_h=None, random_state:int = None,
                 engine: str ="octave", verbose = False, clean_space: bool = True):
        if not isinstance(n_clusters, int):
            raise ValueError(f"Invalid n_clusters. It must be an int. A {type(n_clusters)} was passed.")
        if n_clusters < 2:
            raise ValueError(f"Invalid n_clusters. It must be an greater than 1. {n_clusters} was passed.")
        if factorization not in ["linear", "nonlinear"]:
            raise ValueError(f"Invalid factorization. Expected one of ['linear', 'nonlinear']. {factorization} was passed.")
        if lambda1 <= 0:
            raise ValueError(f"Invalid lambda1. It must be a positive number. {lambda1} was passed.")
        if lambda2 <= 0:
            raise ValueError(f"Invalid lambda2. It must be a positive number. {lambda2} was passed.")
        if (layers is not None) and (not isinstance(layers, list)):
            raise ValueError(f"Invalid layers. It must be a list or None. {layers} was passed.")
        if isinstance(layers, list) and (
            len(layers) == 0 or any((not isinstance(layer, int)) or layer <= 0 for layer in layers)
        ):
            raise ValueError(f"Invalid layers. It must contain positive integers. {layers} was passed.")
        if factorization == "nonlinear" and isinstance(layers, list) and len(layers) != 2:
            raise ValueError("Invalid layers. Nonlinear MLMF expects exactly two layers.")
        if not isinstance(max_iter, int):
            raise ValueError(f"Invalid max_iter. It must be an int. A {type(max_iter)} was passed.")
        if max_iter <= 0:
            raise ValueError(f"Invalid max_iter. It must be a positive value. {max_iter} was passed.")
        if tol <= 0:
            raise ValueError(f"Invalid tol. It must be a positive value. {tol} was passed.")
        if not isinstance(update_h, bool):
            raise ValueError(f"Invalid update_h. It must be a boolean. {update_h} was passed.")
        if not isinstance(update_last_h, bool):
            raise ValueError(f"Invalid update_last_h. It must be a boolean. {update_last_h} was passed.")
        if not isinstance(update_z, bool):
            raise ValueError(f"Invalid update_z. It must be a boolean. {update_z} was passed.")
        if nonlinearity not in ["tanh", "square", "sigmoid", "softplus"]:
            raise ValueError(
                f"Invalid nonlinearity. Expected one of ['tanh', 'square', 'sigmoid', 'softplus']. {nonlinearity} was passed."
            )
        engines_options = ["octave", "python"]
        if engine not in engines_options:
            raise ValueError(f"Invalid engine. Expected one of {engines_options}. {engine} was passed.")
        if (engine == "octave") and (not octavemodule_installed):
            raise ImportError(oct2py_module_error)

        if layers is None:
            layers = [max(2 * n_clusters, n_clusters + 1), n_clusters]

        self.n_clusters = n_clusters
        self.factorization = factorization
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.layers = layers
        self.max_iter = max_iter
        self.tol = tol
        self.update_h = update_h
        self.update_last_h = update_last_h
        self.update_z = update_z
        self.nonlinearity = nonlinearity
        self.init_z = init_z
        self.init_h = init_h
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
        transformed_Xs = tuple(np.nan_to_num(np.asarray(X).T, nan=0.0) for X in Xs)

        init_z = 0 if self.init_z is None else self.init_z
        init_h = 0 if self.init_h is None else self.init_h

        if self.engine=="octave":
            if self.random_state is not None:
                self._oc.rand('seed', self.random_state)

            if self.factorization == "linear":
                Hc, H, loss = self._oc.deepMF(
                    transformed_Xs, self.layers, self.lambda1, self.lambda2, init_z, init_h,
                    int(self.update_h), int(self.update_last_h), self.max_iter, self.tol,
                    int(self.verbose), int(self.update_z), nout=3
                )
            elif self.factorization == "nonlinear":
                Hc, H, loss = self._oc.deepMF_nonlinear(
                    transformed_Xs, self.layers, self.lambda1, self.lambda2, init_z, init_h,
                    int(self.update_h), int(self.update_last_h), self.max_iter, self.tol,
                    int(self.verbose), self.nonlinearity, nout=3
                )

            if self.clean_space:
                self._clean_space()

        elif self.engine == "python":
            self.rng = np.random.default_rng(self.random_state)
            if self.factorization == "linear":
                Hc, H, loss = self._mlmf_linear(transformed_Xs)
            elif self.factorization == "nonlinear":
                Hc, H, loss = self._mlmf_nonlinear(transformed_Xs)

        embedding = np.asarray(Hc).T
        model = KMeans(n_clusters= self.n_clusters, n_init="auto", random_state= self.random_state)
        self.labels_ = model.fit_predict(X=embedding)
        self.embedding_ = embedding
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


    def _mlmf_linear(self, Xs):
        Xs, masks, counts = self._remove_missing_columns(Xs)
        Xs_norm = [self._normalize_columns(X) for X in Xs]
        n_mods = len(Xs)
        n_layers = len(self.layers)
        E = np.ones((self.layers[-1], self.layers[-1]))
        I = np.eye(self.layers[-1])

        if isinstance(self.init_h, list):
            Z, H = self.init_z, self.init_h
        else:
            Z = [[None for _ in range(n_layers)] for _ in range(n_mods)]
            H = [[None for _ in range(n_layers)] for _ in range(n_mods)]
            for view_idx, X_norm in enumerate(Xs_norm):
                for layer_idx, layer in enumerate(self.layers):
                    V = X_norm if layer_idx == 0 else H[view_idx][layer_idx - 1]
                    z0 = self.init_z[layer_idx] if isinstance(self.init_z, list) else self.init_z
                    max_iter = 1 if isinstance(self.init_z, list) else self.max_iter
                    update_z = False if isinstance(self.init_z, list) else self.update_z
                    Z[view_idx][layer_idx], H[view_idx][layer_idx], _ = self._seminmf(
                        V, layer, z0=z0, max_iter=max_iter, update_z=update_z
                    )

        Hc = self._update_hc(H, masks, counts)
        loss = []
        g_inv = lambda x: x

        for _ in range(20):
            for view_idx, (X, X_norm) in enumerate(zip(Xs, Xs_norm)):
                mask = masks[view_idx]
                Hc_view = Hc[:, mask]
                H_err = [None for _ in range(n_layers)]
                H_err[-1] = H[view_idx][-1]
                for layer_idx in range(n_layers - 2, -1, -1):
                    H_err[layer_idx] = Z[view_idx][layer_idx + 1] @ H_err[layer_idx + 1]

                D = None
                for layer_idx in range(n_layers):
                    if self.update_z:
                        try:
                            if layer_idx == 0:
                                Z[view_idx][layer_idx] = X_norm @ np.linalg.pinv(H_err[0])
                            else:
                                Z[view_idx][layer_idx] = (
                                    np.linalg.pinv(D.T) @ X_norm @ np.linalg.pinv(H_err[layer_idx])
                                )
                        except np.linalg.LinAlgError:
                            if self.verbose:
                                print("Convergence error while updating Z.")

                    if layer_idx == 0:
                        D = Z[view_idx][0].T
                    else:
                        D = Z[view_idx][layer_idx].T @ D

                    if self.update_h and layer_idx < n_layers - 1:
                        A = D @ X_norm
                        Ap = (np.abs(A) + A) / 2
                        An = (np.abs(A) - A) / 2
                        B = D @ D.T
                        Bp = (np.abs(B) + B) / 2
                        Bn = (np.abs(B) - B) / 2
                        H[view_idx][layer_idx] *= np.sqrt(
                            (Ap + Bn @ H[view_idx][layer_idx])
                            / np.maximum(An + Bp @ H[view_idx][layer_idx], 1e-10)
                        )

                    if layer_idx == n_layers - 1 and self.update_last_h:
                        B = D @ X + self.lambda2 * Hc_view
                        C = D @ D.T + self.lambda1 * E + self.lambda2 * I
                        Ba = (np.abs(B) + B) / 2
                        Bb = (np.abs(B) - B) / 2
                        Ca = (np.abs(C) + C) / 2
                        Cb = (np.abs(C) - C) / 2
                        A = H[view_idx][layer_idx]
                        H[view_idx][layer_idx] *= np.sqrt((Ba + Cb @ A) / (Bb + Ca @ A))

            Hc = self._update_hc(H, masks, counts)
            loss.append(
                sum(
                    self._deep_cost(X_norm, Z[idx], H[idx], E, Hc[:, masks[idx]], g_inv)
                    for idx, X_norm in enumerate(Xs_norm)
                )
            )

        return Hc, H, np.asarray(loss)


    def _mlmf_nonlinear(self, Xs):
        Xs, masks, counts = self._remove_missing_columns(Xs)
        n_mods = len(Xs)
        n_layers = len(self.layers)
        g, g_inv, g_inv_diff = self._nonlinear_functions(self.nonlinearity)
        E = np.ones((self.layers[-1], self.layers[-1]))

        if isinstance(self.init_h, list):
            Z, H = self.init_z, self.init_h
        else:
            Z = [[None for _ in range(n_layers)] for _ in range(n_mods)]
            H = [[None for _ in range(n_layers)] for _ in range(n_mods)]
            for view_idx, X in enumerate(Xs):
                for layer_idx, layer in enumerate(self.layers):
                    V = X if layer_idx == 0 else g(H[view_idx][layer_idx - 1])
                    Z[view_idx][layer_idx], H[view_idx][layer_idx], _ = self._seminmf(V, layer)

        Hc = self._update_hc(H, masks, counts)
        loss = []
        for _ in range(30):
            for view_idx, X in enumerate(Xs):
                Hc_view = Hc[:, masks[view_idx]]
                dnorm = self._deep_cost(X, Z[view_idx], H[view_idx], E, Hc_view, g_inv)

                for layer_idx in range(n_layers - 1, -1, -1):
                    if layer_idx == 1:
                        ksi = Z[view_idx][0].T @ X
                        psi = Z[view_idx][0].T @ Z[view_idx][0]

                    if self.update_h and (layer_idx < n_layers - 1 or (layer_idx == n_layers - 1 and self.update_last_h)):
                        if layer_idx == 0:
                            H[view_idx][0] = g_inv(Z[view_idx][1] @ H[view_idx][1])
                            H[view_idx][0][H[view_idx][0] <= 0] = np.finfo(float).eps
                        else:
                            c = g_inv_diff(Z[view_idx][1] @ H[view_idx][1])
                            A = 2 * ksi
                            B = 2 * psi @ g_inv(Z[view_idx][1] @ H[view_idx][1])
                            F = 2 * self.lambda1 * E @ H[view_idx][layer_idx]
                            P = 2 * self.lambda2 * (H[view_idx][layer_idx] - Hc_view)
                            C = Z[view_idx][1].T @ ((B - A) * c) + F + P
                            H[view_idx], _ = self._gd_h(
                                X, Z[view_idx], H[view_idx], C, layer_idx, g_inv, dnorm, E, Hc_view
                            )

                    dnorm = self._deep_cost(X, Z[view_idx], H[view_idx], E, Hc_view, g_inv)

                    if layer_idx == 0:
                        Z[view_idx][0] = X @ np.linalg.pinv(g_inv(Z[view_idx][1] @ H[view_idx][1]))
                    else:
                        c = g_inv_diff(Z[view_idx][1] @ H[view_idx][1])
                        C = (
                            (Z[view_idx][0].T @ (Z[view_idx][0] @ g_inv(Z[view_idx][1] @ H[view_idx][1]) - X) * c)
                            @ H[view_idx][1].T
                        )
                        Z[view_idx], _ = self._gd_z(
                            X, Z[view_idx], H[view_idx], C, layer_idx, g_inv, dnorm, E, Hc_view
                        )

            Hc = self._update_hc(H, masks, counts)
            loss.append(
                sum(
                    self._deep_cost(X, Z[idx], H[idx], E, Hc[:, masks[idx]], g_inv)
                    for idx, X in enumerate(Xs)
                )
            )

        return Hc, H, np.asarray(loss)


    def _seminmf(self, X, k, z0=None, h0=None, max_iter=None, update_z=None):
        z0 = self.init_z if z0 is None else z0
        h0 = self.init_h if h0 is None else h0
        max_iter = self.max_iter if max_iter is None else max_iter
        update_z = self.update_z if update_z is None else update_z

        H = self.rng.random((k, X.shape[1])) if h0 is None or np.isscalar(h0) else np.asarray(h0, dtype=float).copy()
        Z = X @ np.linalg.pinv(H) if z0 is None or np.isscalar(z0) else np.asarray(z0, dtype=float).copy()
        dnorm = np.linalg.norm(X - Z @ H, "fro")

        for i in range(1, max_iter + 1):
            if update_z:
                try:
                    Z = X @ np.linalg.pinv(H)
                except np.linalg.LinAlgError:
                    if self.verbose:
                        print("Error inverting")

            A = Z.T @ X
            Ap = (np.abs(A) + A) / 2
            An = (np.abs(A) - A) / 2
            B = Z.T @ Z
            Bp = (np.abs(B) + B) / 2
            Bn = (np.abs(B) - B) / 2

            if self.update_h:
                H *= np.sqrt((Ap + Bn @ H) / np.maximum(An + Bp @ H, np.finfo(float).eps))

            if i % 10 == 0 or (i + 1) % 10 == 0:
                dnorm = np.sqrt(np.sum((X - Z @ H) ** 2))
                if (i + 1) % 10 == 0:
                    dnorm0 = dnorm
                    continue
                if "dnorm0" in locals() and dnorm0 - dnorm <= self.tol * max(1, dnorm0):
                    break

        return Z, H, dnorm


    @staticmethod
    def _remove_missing_columns(Xs):
        transformed = []
        masks = []
        for X in Xs:
            observed = np.any(X != 0, axis=0)
            transformed.append(X[:, observed])
            masks.append(observed)
        counts = np.sum(masks, axis=0)
        return transformed, masks, counts


    @staticmethod
    def _normalize_columns(X):
        return X / np.sqrt(np.sum(X ** 2, axis=0, keepdims=True))


    @staticmethod
    def _update_hc(H, masks, counts):
        hg = np.zeros((H[0][-1].shape[0], counts.shape[0]))
        for H_view, mask in zip(H, masks):
            hg[:, mask] += H_view[-1]
        return hg / counts


    @staticmethod
    def _deep_recon(Z, H, g_inv):
        out = H[-1]
        for layer_idx in range(len(H) - 1, -1, -1):
            out = g_inv(Z[layer_idx] @ out)
        return out


    def _deep_cost(self, X, Z, H, E, Hc_view, g_inv):
        return (
            np.linalg.norm(X - self._deep_recon(Z, H, g_inv), "fro")
            + self.lambda1 * np.trace(H[-1] @ H[-1].T @ E)
            + self.lambda2 * np.linalg.norm(H[-1] - Hc_view, "fro")
        )


    def _gd_z(self, X, Z, H, c, layer_idx, g_inv, dnorm, E, Hc_view):
        eta = 0.01
        old_z = Z[layer_idx].copy()
        while True:
            eta /= 2
            Z[layer_idx] = old_z - eta * c
            dnorm1 = self._deep_cost(X, Z, H, E, Hc_view, g_inv)

            if eta < 0.00001:
                Z[layer_idx] = old_z
                dnorm1 = dnorm
                break
            if dnorm1 <= dnorm:
                break
        return Z, dnorm1


    def _gd_h(self, X, Z, H, c, layer_idx, g_inv, dnorm, E, Hc_view):
        eta = 0.01
        old_h = H[layer_idx].copy()
        if layer_idx == 0:
            dnorm = np.linalg.norm(X - Z[0] @ H[0], "fro")

        while True:
            eta /= 2
            H[layer_idx] = old_h - eta * c
            H[layer_idx][H[layer_idx] <= 0] = np.finfo(float).eps
            if layer_idx == 0:
                dnorm1 = np.linalg.norm(X - Z[0] @ H[0], "fro")
            else:
                dnorm1 = self._deep_cost(X, Z, H, E, Hc_view, g_inv)

            if eta < 0.00001:
                H[layer_idx] = old_h
                dnorm1 = dnorm
                break
            if dnorm1 <= dnorm:
                break
        return H, dnorm1


    @staticmethod
    def _nonlinear_functions(nonlinearity):
        if nonlinearity == "tanh":
            return (
                lambda x: 3 * np.arctanh(x / 1.7159) / 2,
                lambda x: 1.7159 * np.tanh(2 * x / 3),
                lambda x: 1.7159 * 2 / 3 * (1 / np.cosh((2 * x) / 3) ** 2),
            )
        if nonlinearity == "square":
            return lambda x: x ** 0.5, lambda x: x * x, lambda x: 2 * x
        if nonlinearity == "sigmoid":
            sigmoid = lambda x: 1 / (1 + np.exp(-x))
            return lambda x: np.log(x / (1 - x)), sigmoid, lambda x: sigmoid(x) * (1 - sigmoid(x))
        return lambda x: np.log(np.exp(x) - 1), lambda x: np.log(1 + np.exp(x)), lambda x: np.exp(x) / (1 + np.exp(x))
