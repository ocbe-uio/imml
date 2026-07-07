# License: BSD-3-Clause

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from ..utils import check_Xs_y
from .. import rmodule_installed, rmodule_error

if rmodule_installed:
    from rpy2.robjects.packages import importr, PackageNotInstalledError
    from ..utils import _convert_df_to_r_object
    import rpy2.robjects as ro
    rbase = importr("base")
    try:
        nnTensor = importr("nnTensor")
        nnTensor_installed = True
    except PackageNotInstalledError:
        nnTensor_installed = False
        nnTensor_module_error = "nnTensor needs to be installed in R to use r engine."


class JNMF(TransformerMixin, BaseEstimator):
    r"""
    Joint Non-Negative Matrix Factorization (JNMF).
    [#jnmfpaper1]_ [#jnmfpaper2]_ [#jnmfcode1]_ [#jnmfcode2]_

    JNMF decompose the matrices to low-dimensional factor matrices.

    It can deal with both modality- and feature-wise missing.

    Parameters
    ----------
    n_components : int, default=10
        Number of components to keep.
    init_W : array-like, default=None
        The initial values of factor matrix W, which has n_samples-rows and n_components-columns.
    init_V : array-like, default=None
        A list containing the initial values of multiple factor matrices.
    init_H : array-like, default=None
        A list containing the initial values of multiple factor matrices.
    l1_W : float, default=1e-10
        Paramter for L1 regularitation. This also works as small positive constant to prevent division by zero,
        so should be set as 0.
    l1_V : float, default=1e-10
        Paramter for L1 regularitation. This also works as small positive constant to prevent division by zero,
        so should be set as 0.
    l1_H : float, default=1e-10
        Paramter for L1 regularitation. This also works as small positive constant to prevent division by zero,
        so should be set as 0.
    l2_W : float, default=1e-10
        Parameter for L2 regularitation.
    l2_V : float, default=1e-10
        Parameter for L2 regularitation.
    l2_H : float, default=1e-10
        Parameter for L2 regularitation.
    weights : list, default=None
        Weight vector.
    beta_loss : int, default='Frobenius'
        One of ["Frobenius", "KL", "IS", "PLTF"].
    p : float, default=1.0
        The parameter of Probabilistic Latent Tensor Factorization. Only used when beta_loss="PLTF".
    tol : float, default=1e-10
        Tolerance of the stopping condition.
    max_iter : int, default=100
        Maximum number of iterations to perform.
    random_state : int, default=None
        Determines the randomness. Use an int to make the randomness deterministic.
    verbose : bool, default=False
        Verbosity mode.
    engine : str, default='python'
        Engine to use for computing the model. One of ``["r", "python"]``.

    Attributes
    ----------
    H_ : list of n_mods array-likes objects of shape (n_features_i, n_components)
        List of specific factorization matrix.
    V_ : list of n_mods array-likes objects of shape (n_samples, n_components)
        List of specific factorization matrix.
    reconstruction_err_ : list of float
        Beta-divergence between the training data X and the reconstructed data WH from the fitted model.
    observed_reconstruction_err_ : list of float
        Beta-divergence between the observed values and the reconstructed data WH from the fitted model.
    missing_reconstruction_err_ : list of float
        Beta-divergence between the missing values and the reconstructed data WH from the fitted model.
    relchange_ : list of float
        The relative change of the error.

    References
    ----------
    .. [#jnmfpaper1] Tsuyuzaki et al., (2023). nnTensor: An R package for non-negative matrix/tensor decomposition.
                     Journal of Open Source Software, 8(84), 5015, https://doi.org/10.21105/joss.05015
    .. [#jnmfpaper2] Zi Yang, et al. (2016) A non-negative matrix factorization method for detecting modules in
                     heterogeneous omics multi-modal data, Bioinformatics 32(1), 1-8.
    .. [#jnmfcode1] https://cran.r-project.org/web/packages/nnTensor/vignettes/nnTensor-2.html
    .. [#jnmfcode2] https://github.com/rikenbit/nnTensor

    See Also
    --------
    :class:`~imml.impute.JNMFImputer`
    :class:`~imml.feature_selection.JNMFFeatureSelector`

    Example
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from imml.decomposition import JNMF
    >>> Xs = [pd.DataFrame(np.random.default_rng(42).uniform(size=(20, 10))) for i in range(3)]
    >>> transformer = JNMF(n_components = 5)
    >>> transformed_Xs = transformer.fit_transform(Xs)
    """

    def __init__(self, n_components: int = 10, init_W=None, init_V=None, init_H=None,
                 l1_W: float = 1e-10, l1_V: float = 1e-10, l1_H: float = 1e-10,
                 l2_W: float = 1e-10, l2_V: float = 1e-10, l2_H: float = 1e-10,
                 weights=None, beta_loss: str = "Frobenius", p: float = 1., tol: float = 1e-10,
                 max_iter: int = 100, verbose=0, random_state: int = None,
                 engine: str = "python"):
        engines_options = ["r", "python"]
        if engine not in engines_options:
            raise ValueError(f"Invalid engine. Expected one of {engines_options}. {engine} was passed.")
        if engine == "r":
            if not rmodule_installed:
                raise ImportError(rmodule_error)
            elif not nnTensor_installed:
                raise ImportError(nnTensor_module_error)

        if not isinstance(beta_loss, str):
            raise ValueError(f"Invalid beta_loss. It must be a str. A {type(beta_loss)} was passed.")
        if not isinstance(p, float):
            raise ValueError(f"Invalid float. It must be a float. A {type(float)} was passed.")

        if beta_loss == "Frobenius":
            p = 0
        elif beta_loss == "KL":
            p = 1
        elif beta_loss == "IS":
            p = 1

        self.n_components = n_components
        self.init_W = init_W
        self.init_V = init_V
        self.init_H = init_H
        self.l1_W = l1_W
        self.l1_V = l1_V
        self.l1_H = l1_H
        self.l2_W = l2_W
        self.l2_V = l2_V
        self.l2_H = l2_H
        self.weights = weights
        self.beta_loss = beta_loss
        self.p = p
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = bool(verbose)
        if random_state is None:
            random_state = int(np.random.default_rng().integers(10000))
        self.random_state = random_state
        self.engine = engine
        self.transform_ = None


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
        self :  returns an instance of self.
        """
        Xs = check_Xs_y(Xs, ensure_all_finite='allow-nan')
        if not isinstance(Xs[0], pd.DataFrame):
            self.transform_ = "numpy"
            Xs = [pd.DataFrame(X) for X in Xs]
        else:
            self.transform_ = "pandas"

        if self.engine == "r":
            transformed_Xs, transformed_mask, beta_loss, init_W, init_V, init_H, weights = \
                self._prepare_variables(Xs=Xs, beta_loss=self.beta_loss,
                                        init_W=self.init_W, init_V=self.init_V,
                                        init_H=self.init_H, weights=self.weights)
            if self.random_state is not None:
                rbase.set_seed(self.random_state)

            W, V, H, recerror, train_recerror, test_recerror, relchange = nnTensor.jNMF(
                X=transformed_Xs, M=transformed_mask, J=self.n_components,
                initW=init_W, initV=init_V, initH=init_H,
                fixW=False, fixV=False, fixH=False,
                L1_W=self.l1_W, L1_V=self.l1_V, L1_H=self.l1_H,
                L2_W=self.l2_W, L2_V=self.l2_V, L2_H=self.l2_H,
                w=weights, algorithm=beta_loss, p=self.p,
                thr=self.tol, num_iter=self.max_iter, verbose=self.verbose)

            H = [np.array(mat) for mat in H]
            V = [np.array(mat) for mat in V]

        elif self.engine == "python":
            _, V, H, recerror, train_recerror, test_recerror, relchange = self._jnmf_python(Xs=Xs, fixW=False, fixV=False, fixH=False)

        self.H_ = H
        self.V_ = V
        self.reconstruction_err_ = list(recerror)
        self.observed_reconstruction_err_ = list(train_recerror)
        self.missing_reconstruction_err_ = list(test_recerror)
        self.relchange_ = list(relchange)
        return self


    def transform(self, Xs):
        r"""
        Project data into the learned space.

        Parameters
        ----------
        Xs : list of array-likes objects
            - Xs length: n_mods
            - Xs[i] shape: (n_samples, n_features_i)

            A list of different modalities.

        Returns
        -------
        transformed_X : array-like of shape (n_samples, n_components)
            The shared factor matrix W for the input data.
        """
        Xs = check_Xs_y(Xs, ensure_all_finite='allow-nan')
        if not isinstance(Xs[0], pd.DataFrame):
            Xs = [pd.DataFrame(X) for X in Xs]

        if self.engine == "r":
            transformed_Xs, transformed_mask, beta_loss, init_W, init_V, init_H, weights = \
                self._prepare_variables(Xs=Xs, beta_loss=self.beta_loss,
                                        init_W=self.init_W, init_V=self.init_V,
                                        init_H=self.H_, weights=self.weights)

            if not isinstance(self.H_[0], pd.DataFrame):
                H = [pd.DataFrame(h) for h in self.H_]
            H = _convert_df_to_r_object(H)
            if self.random_state is not None:
                rbase.set_seed(self.random_state)

            transformed_X = nnTensor.jNMF(
                X=transformed_Xs, M=transformed_mask, J=self.n_components,
                initW=init_W, initV=init_V, initH=H,
                fixW=False, fixV=False, fixH=True,
                L1_W=self.l1_W, L1_V=self.l1_V, L1_H=self.l1_H,
                L2_W=self.l2_W, L2_V=self.l2_V, L2_H=self.l2_H,
                w=weights, algorithm=beta_loss, p=self.p,
                thr=self.tol, num_iter=self.max_iter,
                verbose=self.verbose)[0]
            transformed_X = np.array(transformed_X)

        elif self.engine == "python":
            transformed_X = self._jnmf_python(Xs=Xs, fixW=False, fixV=False, fixH=True)[0]

        if self.transform_ == "pandas":
            transformed_X = pd.DataFrame(transformed_X, index=Xs[0].index)

        return transformed_X


    def fit_transform(self, Xs, y=None, **fit_params):
        r"""
        Fit to data, then transform it.

        Parameters
        ----------
        Xs : list of array-likes objects
            - Xs length: n_mods
            - Xs[i] shape: (n_samples_i, n_features_i)

            A list of different modalities.
        y : Ignored
            Not used, present here for API consistency by convention.
        fit_params : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        transformed_X : array-like of shape (n_samples, n_components)
            The shared factor matrix W.
        """
        Xs = check_Xs_y(Xs, ensure_all_finite='allow-nan')
        if not isinstance(Xs[0], pd.DataFrame):
            self.transform_ = "numpy"
            Xs = [pd.DataFrame(X) for X in Xs]
        else:
            self.transform_ = "pandas"

        if self.engine == "r":
            transformed_Xs, transformed_mask, beta_loss, init_W, init_V, init_H, weights = \
                self._prepare_variables(Xs=Xs, beta_loss=self.beta_loss,
                                        init_W=self.init_W, init_V=self.init_V,
                                        init_H=self.init_H, weights=self.weights)
            if self.random_state is not None:
                rbase.set_seed(self.random_state)

            W, V, H, recerror, train_recerror, test_recerror, relchange = nnTensor.jNMF(
                X=transformed_Xs, M=transformed_mask, J=self.n_components,
                initW=init_W, initV=init_V, initH=init_H,
                fixW=False, fixV=False, fixH=False,
                L1_W=self.l1_W, L1_V=self.l1_V, L1_H=self.l1_H,
                L2_W=self.l2_W, L2_V=self.l2_V, L2_H=self.l2_H,
                w=weights, algorithm=beta_loss, p=self.p,
                thr=self.tol, num_iter=self.max_iter, verbose=self.verbose)

            H = [np.array(mat) for mat in H]
            V = [np.array(mat) for mat in V]
            transformed_X = np.array(W)

        elif self.engine == "python":
            transformed_X, V, H, recerror, train_recerror, test_recerror, relchange = self._jnmf_python(Xs=Xs, fixW=False, fixV=False, fixH=False)

        if self.transform_ == "pandas":
            H = [pd.DataFrame(mat, index=X.columns) for X, mat in zip(Xs, H)]
            V = [pd.DataFrame(mat, index=X.index) for X, mat in zip(Xs, V)]
            transformed_X = pd.DataFrame(transformed_X, index=Xs[0].index)

        self.H_ = H
        self.V_ = V
        self.reconstruction_err_ = list(recerror)
        self.observed_reconstruction_err_ = list(train_recerror)
        self.missing_reconstruction_err_ = list(test_recerror)
        self.relchange_ = list(relchange)
        return transformed_X


    @staticmethod
    def _column_norm(X):
        norms = np.linalg.norm(X, axis=0, keepdims=True)
        norms[norms == 0.0] = 1.0
        return X / norms


    @staticmethod
    def _rec_error(X, X_bar, notsqrt=False):
        """Frobenius reconstruction error, ignoring NaN entries.

        Equivalent to R's .recError(notsqrt=notsqrt).
        Returns sum of squared differences when *notsqrt=True*,
        or its square root otherwise.
        """
        diff = X_bar - X
        ssq = np.nansum(diff * diff)
        return ssq if notsqrt else np.sqrt(ssq)


    def _jnmf_python(self, Xs, fixW, fixV, fixH):
        X_list = [X.values.astype(float) for X in Xs]
        M_list = [X.notnull().astype(float).values for X in Xs]
        w = [1.0] * len(Xs) if self.weights is None else list(self.weights)

        eps = np.finfo(float).eps
        K = len(X_list)
        n_samples = X_list[0].shape[0]
        rng = np.random.default_rng(self.random_state)

        # M_NA: 1=originally observed (not NaN), 0=NaN  –  mirrors R's M_NA
        M_NA = [np.where(np.isnan(Xk), 0.0, 1.0) for Xk in X_list]

        # Working copies of the mask (provided M is the training mask)
        M = [mk.astype(float) for mk in M_list]

        # Replace NaN and exact zeros in X with pseudocount (R: pseudocount=.Machine$double.eps)
        X = []
        for Xk in X_list:
            Xk_f = Xk.copy().astype(float)
            Xk_f[np.isnan(Xk_f)] = eps
            Xk_f[Xk_f == 0.0] = eps
            X.append(Xk_f)

        # pM: mask with zeros replaced by pseudocount (used as weight in update numerators/denominators)
        pM = []
        for Mk in M:
            pMk = Mk.copy()
            pMk[pMk == 0.0] = eps
            pM.append(pMk)

        # Normalise weights to sum to 1  (R: w <- w / sum(w))
        w = np.asarray(w, dtype=float)
        w = w / w.sum()

        # ---- Initialise W ----
        if self.init_W is None:
            W_raw = rng.random((n_samples, self.n_components))
            W = self._column_norm(W_raw * W_raw)
        else:
            W = np.asarray(self.init_W, dtype=float).copy()

        # ---- Initialise V (list of K matrices) ----
        if self.init_V is None:
            V = []
            for _ in range(K):
                V_raw = rng.random((n_samples, self.n_components))
                V.append(self._column_norm(V_raw * V_raw))
        else:
            V = [np.asarray(v, dtype=float).copy() for v in self.init_V]

        # ---- Initialise H (list of K matrices) ----
        if self.init_H is None:
            H = []
            for k in range(K):
                H_raw = rng.random((X[k].shape[1], self.n_components))
                H.append(self._column_norm(H_raw * H_raw))
        else:
            H = [np.asarray(h, dtype=float).copy() for h in self.init_H]

        # Expand scalar fix-flags to per-modality lists (R: fixV <- rep(fixV, length=K))
        if isinstance(fixV, bool):
            fixV = [fixV] * K
        if isinstance(fixH, bool):
            fixH = [fixH] * K

        if self.p < 1.0:
            rho = 1.0 / (2.0 - self.p)
        elif self.p <= 2.0:
            rho = 1.0
        else:
            rho = 1.0 / (self.p - 1.0)

        # Error arrays: index 0 is the "offset" seed (thr*10), matching R's RecError[1]
        rec_error = [self.tol * 10.0]
        train_rec_error = [self.tol * 10.0]
        test_rec_error = [self.tol * 10.0]
        rel_change = [self.tol * 10.0]

        iter_idx = 0
        # Loop mirrors R: while (RecError[iter] > thr) && (iter <= num.iter)
        while rec_error[iter_idx] > self.tol and iter_idx < self.max_iter:

            # ---- Pre-update reconstruction (used for ALL updates this iteration) ----
            X_bar = [(W + V[k]) @ H[k].T for k in range(K)]
            pre_error = np.sqrt(sum(self._rec_error(X[k], X_bar[k], notsqrt=True) for k in range(K)))

            # ---- Update W (shared factor) ----
            if not fixW:
                W_numer = np.zeros_like(W)
                W_denom = np.zeros_like(W)
                for k in range(K):
                    pMX = pM[k] * X[k]
                    pMX_bar = pM[k] * X_bar[k]
                    # Numerator: w_k * (pM * X * (pM * X_bar)^(-self.p)) @ H_k
                    W_numer += w[k] * (pMX * pMX_bar ** (-self.p)) @ H[k]
                    # Denominator: w_k * (pM * X_bar)^(1-self.p) @ H_k + L1_W + L2_W * W
                    # Note: L1/L2 terms are added inside the loop (replicating R exactly)
                    W_denom += w[k] * pMX_bar ** (1.0 - self.p) @ H[k] + self.l1_W + self.l2_W * W
                W = self._column_norm(np.maximum(W * W_numer / W_denom, np.finfo(float).eps) ** rho)

            # ---- Update H_k (feature factors) ----
            for k in range(K):
                if not fixH[k]:
                    pMX = pM[k] * X[k]
                    pMX_bar = pM[k] * X_bar[k]
                    # Numerator: ((pM*X).T * (pM*X_bar).T^(-self.p)) @ (W + V_k)
                    Hk_numer = (pMX.T * pMX_bar.T ** (-self.p)) @ (W + V[k])
                    # Denominator: (pM*X_bar).T^(1-self.p) @ W + L1_H + L2_H * H_k
                    # (uses W, not W+V – matches R's formula exactly)
                    Hk_denom = pMX_bar.T ** (1.0 - self.p) @ W + self.l1_H + self.l2_H * H[k]
                    H[k] = H[k] * (Hk_numer / Hk_denom) ** rho

            # ---- Update V_k (sample-level specific factors) ----
            for k in range(K):
                if not fixV[k]:
                    pMX = pM[k] * X[k]
                    pMX_bar = pM[k] * X_bar[k]
                    # Numerator: (pM * X * (pM * X_bar)^(-self.p)) @ H_k
                    Vk_numer = (pMX * pMX_bar ** (-self.p)) @ H[k]
                    # Denominator: (pM * X_bar)^(1-self.p) @ H_k + L1_V + L2_V * V_k
                    Vk_denom = pMX_bar ** (1.0 - self.p) @ H[k] + self.l1_V + self.l2_V * V[k]
                    V[k] = V[k] * (Vk_numer / Vk_denom) ** rho

            # ---- Post-update errors ----
            iter_idx += 1
            X_bar = [(W + V[k]) @ H[k].T for k in range(K)]

            new_rec = np.sqrt(sum(self._rec_error(X[k], X_bar[k], notsqrt=True) for k in range(K)))
            rec_error.append(new_rec)

            # TrainRecError: error on entries where M == M_NA (i.e. the "observed" training set)
            train_rec_error.append(np.sqrt(sum(
                self._rec_error(
                    (1.0 - M_NA[k] + M[k]) * X[k],
                    (1.0 - M_NA[k] + M[k]) * X_bar[k],
                    notsqrt=True)
                for k in range(K))))

            # TestRecError: error on held-out entries (M_NA - M gives 1 where observed but held out)
            test_rec_error.append(np.sqrt(sum(
                self._rec_error(
                    (M_NA[k] - M[k]) * X[k],
                    (M_NA[k] - M[k]) * X_bar[k],
                    notsqrt=True)
                for k in range(K))))

            denom_rel = new_rec if new_rec != 0.0 else 1.0
            rel_change.append(abs(pre_error - new_rec) / denom_rel)

            if self.verbose:
                print(f"{iter_idx} / {self.max_iter} "
                      f"|Previous Error - Error| / Error = {rel_change[-1]}")

            if np.isnan(rel_change[-1]):
                raise ValueError(
                    "NaN generated. Please run again or change the parameters.")

        return (W, V, H,
                np.array(rec_error),
                np.array(train_rec_error),
                np.array(test_rec_error),
                np.array(rel_change))


    @staticmethod
    def _prepare_variables(Xs, beta_loss, init_W, init_V, init_H, weights):
        mask = [X.notnull().astype(int) for X in Xs]
        transformed_Xs = _convert_df_to_r_object(Xs)
        transformed_mask = _convert_df_to_r_object(mask)
        init_W = ro.NULL if init_W is None else init_W
        init_V = ro.NULL if init_V is None else init_V
        init_H = ro.NULL if init_H is None else init_H
        weights = ro.NULL if weights is None else weights
        return transformed_Xs, transformed_mask, beta_loss, init_W, init_V, init_H, weights
