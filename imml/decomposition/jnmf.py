# License: BSD-3-Clause

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

try:
    from rpy2.robjects.packages import importr, PackageNotInstalledError
    from ..utils import check_Xs, _convert_df_to_r_object
    rmodule_installed = True
except ImportError:
    rmodule_installed = False
    rmodule_error = "Module 'r' needs to be installed to use r engine. See https://imml.readthedocs.io/stable/main/installation.html#optional-dependencies"

if rmodule_installed:
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
    [#jnmfpaper1]_ [#jnmfpaper2]_ [#jnmfpaper3]_ [#jnmfpaper4]_ [#jnmfpaper5]_ [#jnmfpaper6]_ [#jnmfcode1]_ [#jnmfcode2]_

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
    p : float, default=None
        The parameter of Probabilistic Latent Tensor Factorization (p=0: Frobenius, p=1: KL, p=2: IS) .
    tol : int, default=1e-10
        Tolerance of the stopping condition.
    max_iter : int, default=100
        Maximum number of iterations to perform.
    random_state : int, default=None
        Determines the randomness. Use an int to make the randomness deterministic.
    verbose : bool, default=False
        Verbosity mode.
    engine : str, default='r'
        Engine to use for computing the model. Currently only 'r' is supported.

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
    .. [#jnmfpaper2] Liviu Badea, (2008) Extracting Gene Expression Profiles Common to Colon and Pancreatic
                     Adenocarcinoma using Simultaneous nonnegative matrix factorization. Pacific Symposium on
                     Biocomputing 13:279-290.
    .. [#jnmfpaper3] Shihua Zhang, et al. (2012) Discovery of multi-dimensional modules by integrative analysis of
                     cancer genomic data. Nucleic Acids Research 40(19), 9379-9391.
    .. [#jnmfpaper4] Zi Yang, et al. (2016) A non-negative matrix factorization method for detecting modules in
                     heterogeneous omics multi-modal data, Bioinformatics 32(1), 1-8.
    .. [#jnmfpaper5] Y. Kenan Yilmaz et al., (2010) Probabilistic Latent Tensor Factorization, International Conference
                     on Latent Variable Analysis and Signal Separation 346-353.
    .. [#jnmfpaper6] N. Fujita et al., (2018) Biomarker discovery by integrated joint non-negative matrix factorization
                     and pathway signature analyses, Scientific Report.
    .. [#jnmfcode1] https://rdrr.io/cran/nnTensor/man/JNMF.html
    .. [#jnmfcode2] https://github.com/rikenbit/nnTensor

    Example
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from imml.decomposition import JNMF
    >>> Xs = [pd.DataFrame(np.random.default_rng(42).uniform(size=(20, 10))) for i in range(3)]
    >>> transformer = JNMF(n_components = 5)
    >>> transformed_Xs = transformer.fit_transform(Xs)
    """

    def __init__(self, n_components : int = 10, init_W = None, init_V = None, init_H = None,
                 l1_W: float = 1e-10, l1_V: float = 1e-10, l1_H: float = 1e-10,
                 l2_W: float = 1e-10, l2_V: float = 1e-10, l2_H: float = 1e-10, weights = None,
                 beta_loss : list = None, p: float = 1., tol: float = 1e-10, max_iter: int = 100,
                 verbose=0, random_state: int = None, engine: str = "r"):
        engines_options = ["r"]
        if engine not in engines_options:
            raise ValueError(f"Invalid engine. Expected one of {engines_options}. {engine} was passed.")
        if engine == "r":
            if not rmodule_installed:
                raise ImportError(rmodule_error)
            elif not nnTensor_installed:
                raise ImportError(nnTensor_module_error)

        if beta_loss is None:
            beta_loss = ["Frobenius", "KL", "IS", "PLTF"]

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


    def fit(self, Xs, y = None):
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
        Xs = check_Xs(Xs, ensure_all_finite='allow-nan')
        if not isinstance(Xs[0], pd.DataFrame):
            self.transform_ = "numpy"
            Xs = [pd.DataFrame(X) for X in Xs]
        else:
            self.transform_ = "pandas"

        if self.engine=="r":
            transformed_Xs, transformed_mask, beta_loss, init_W, init_V, init_H, weights = self._prepare_variables(
                Xs=Xs, beta_loss=self.beta_loss, init_W=self.init_W, init_V=self.init_V, init_H=self.init_H,
                weights=self.weights)
            if self.random_state is not None:
                rbase.set_seed(self.random_state)

            W, V, H, recerror, train_recerror, test_recerror, relchange = nnTensor.jNMF(
                X= transformed_Xs, M=transformed_mask, J=self.n_components,
                initW=init_W, initV=init_V, initH=init_H, fixW=False, fixV=False, fixH=False,
                L1_W=self.l1_W, L1_V=self.l1_V, L1_H=self.l1_H, L2_W=self.l2_W, L2_V= self.l2_V, L2_H=self.l2_H,
                w=weights, algorithm=beta_loss, p=self.p, thr = self.tol, num_iter=self.max_iter, verbose=self.verbose)

            H = [np.array(mat) for mat in H]
            V = [np.array(mat) for mat in V]
            if self.transform_ == "pandas":
                H = [pd.DataFrame(mat, index=X.columns) for X,mat in zip(Xs, H)]
                V = [pd.DataFrame(mat, index=X.index) for X,mat in zip(Xs, V)]

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
        transformed_Xs : list of array-likes objects, shape (n_samples, n_components)
            The projected data.
        """
        Xs = check_Xs(Xs, ensure_all_finite='allow-nan')
        if not isinstance(Xs[0], pd.DataFrame):
            Xs = [pd.DataFrame(X) for X in Xs]

        if self.engine == "r":
            transformed_Xs, transformed_mask, beta_loss, init_W, init_V, init_H, weights = self._prepare_variables(
                Xs=Xs, beta_loss=self.beta_loss, init_W=self.init_W, init_V=self.init_V, init_H=self.H_,
                weights=self.weights)

            if not isinstance(self.H_[0], pd.DataFrame):
                H = [pd.DataFrame(H) for H in self.H_]
            else:
                H = self.H_
            H = _convert_df_to_r_object(H)
            if self.random_state is not None:
                rbase.set_seed(self.random_state)

            transformed_X = nnTensor.jNMF(X= transformed_Xs, M=transformed_mask, J=self.n_components,
                                          initW=init_W, initV=init_V, initH=H,
                                          fixW=False, fixV=False, fixH=True,
                                          L1_W=self.l1_W, L1_V=self.l1_V, L1_H=self.l1_H,
                                          L2_W=self.l2_W, L2_V= self.l2_V, L2_H=self.l2_H,
                                          w=weights, algorithm=beta_loss, p=self.p, thr = self.tol, num_iter=self.max_iter,
                                          verbose=self.verbose)[0]

        transformed_X = np.array(transformed_X)
        if self.transform_ == "pandas":
            transformed_X = pd.DataFrame(transformed_X, index= Xs[0].index)

        return transformed_X


    def fit_transform(self, Xs, y = None, **fit_params):
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
        transformed_X : array-likes objects of shape (n_samples, n_components)
            The projected data.
        """
        Xs = check_Xs(Xs, ensure_all_finite='allow-nan')
        if not isinstance(Xs[0], pd.DataFrame):
            self.transform_ = "numpy"
            Xs = [pd.DataFrame(X) for X in Xs]
        else:
            self.transform_ = "pandas"

        if self.engine=="r":
            transformed_Xs, transformed_mask, beta_loss, init_W, init_V, init_H, weights = self._prepare_variables(
                Xs=Xs, beta_loss=self.beta_loss, init_W=self.init_W, init_V=self.init_V, init_H=self.init_H,
                weights=self.weights)
            if self.random_state is not None:
                rbase.set_seed(self.random_state)

            W, V, H, recerror, train_recerror, test_recerror, relchange = nnTensor.jNMF(
                X= transformed_Xs, M=transformed_mask, J=self.n_components,
                initW=init_W, initV=init_V, initH=init_H, fixW=False, fixV=False, fixH=False,
                L1_W=self.l1_W, L1_V=self.l1_V, L1_H=self.l1_H, L2_W=self.l2_W, L2_V= self.l2_V, L2_H=self.l2_H,
                w=weights, algorithm=beta_loss, p=self.p, thr = self.tol, num_iter=self.max_iter, verbose=self.verbose)

            H = [np.array(mat) for mat in H]
            V = [np.array(mat) for mat in V]
            transformed_X = np.array(W)
            if self.transform_ == "pandas":
                H = [pd.DataFrame(mat, index=X.columns) for X,mat in zip(Xs, H)]
                V = [pd.DataFrame(mat, index=X.index) for X,mat in zip(Xs, V)]
                transformed_X = pd.DataFrame(transformed_X, index=Xs[0].index)

        self.H_ = H
        self.V_ = V
        self.reconstruction_err_ = list(recerror)
        self.observed_reconstruction_err_ = list(train_recerror)
        self.missing_reconstruction_err_ = list(test_recerror)
        self.relchange_ = list(relchange)
        return transformed_X


    @staticmethod
    def _prepare_variables(Xs, beta_loss, init_W, init_V, init_H, weights):
        import rpy2.robjects as ro
        mask = [X.notnull().astype(int) for X in Xs]
        transformed_Xs, transformed_mask = _convert_df_to_r_object(Xs), _convert_df_to_r_object(mask)
        if beta_loss is not None:
            beta_loss = ro.vectors.StrVector(beta_loss)
        init_W = ro.NULL if init_W is None else init_W
        init_V = ro.NULL if init_V is None else init_V
        init_H = ro.NULL if init_H is None else init_H
        weights = ro.NULL if weights is None else weights
        return transformed_Xs, transformed_mask, beta_loss, init_W, init_V, init_H, weights

