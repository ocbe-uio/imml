import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from ..utils import check_Xs, _convert_df_to_r_object


class jNMF(TransformerMixin, BaseEstimator):
    r"""
    Joint Non-negative Matrix Factorization Algorithms (jNMF).

    jNMF decompose the matrices to two low-dimensional factor matrices. It can deal with both view- and
    single-wise missing.

    R library "nnTensor" should be installed. This can be done using the R command: install.packages("nnTensor")

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
        Engine to use for computing the model.

    Attributes
    ----------
    H_ : list of array-like, shape=(n_features_i, n_components)
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
    [paper1] Liviu Badea, (2008) Extracting Gene Expression Profiles Common to Colon and Pancreatic Adenocarcinoma
            using Simultaneous nonnegative matrix factorization. Pacific Symposium on Biocomputing 13:279-290.
    [paper2] Shihua Zhang, et al. (2012) Discovery of multi-dimensional modules by integrative analysis of cancer
            genomic data. Nucleic Acids Research 40(19), 9379-9391.
    [paper3] Zi Yang, et al. (2016) A non-negative matrix factorization method for detecting modules in heterogeneous
            omics multi-modal data, Bioinformatics 32(1), 1-8.
    [paper4] Y. Kenan Yilmaz et al., (2010) Probabilistic Latent Tensor Factorization, International Conference on
            Latent Variable Analysis and Signal Separation 346-353.
    [paper5] N. Fujita et al., (2018) Biomarker discovery by integrated joint non-negative matrix factorization and
            pathway signature analyses, Scientific Report.
    [code] https://rdrr.io/cran/nnTensor/man/jNMF.html

    Examples
    --------
    >>> from imvc.datasets import LoadDataset
    >>> from imvc.decomposition import jNMF
    >>> from imvc.preprocessing import MultiViewTransformer
    >>> from sklearn.pipeline import make_pipeline
    >>> from sklearn.preprocessing import MinMaxScaler
    >>> from sklearn.cluster import KMeans
    >>> Xs = LoadDataset.load_dataset(dataset_name="nutrimouse")
    >>> transformer = jNMF(n_components = 5).set_output(transform="pandas")
    >>> estimator = KMeans(n_clusters = 3)
    >>> pipeline = make_pipeline(MultiViewTransformer(MinMaxScaler().set_output(transform="pandas")), transformer, estimator)
    >>> labels = pipeline.fit_predict(Xs)
    """


    def __init__(self, n_components : int = 10, init_W = None, init_V = None, init_H = None,
                 l1_W: float = 1e-10, l1_V: float = 1e-10, l1_H: float = 1e-10,
                 l2_W: float = 1e-10, l2_V: float = 1e-10, l2_H: float = 1e-10, weights = None,
                 beta_loss = ["Frobenius", "KL", "IS", "PLTF"], p: float = 1., tol: float = 1e-10, max_iter: int = 100,
                 verbose=0, random_state: int = None, engine: str = "r"):
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
        self.random_state = random_state
        self.engine = engine
        self.transform_ = None


    def fit(self, Xs, y = None):
        r"""
        Fit the transformer to the input data.

        Parameters
        ----------
        Xs : list of array-likes
            - Xs length: n_views
            - Xs[i] shape: (n_samples, n_features_i)
            A list of different views.
        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self :  returns and instance of self.
        """
        Xs = check_Xs(Xs, force_all_finite='allow-nan')
        samples = Xs[0].index

        if self.engine=="r":
            from rpy2.robjects.packages import importr
            nnTensor = importr("nnTensor")
            transformed_Xs, transformed_mask, beta_loss, init_W, init_V, init_H, weights = self._prepare_variables(
                Xs=Xs, beta_loss=self.beta_loss, init_W=self.init_W, init_V=self.init_V, init_H=self.init_H,
                weights=self.weights)
            if self.random_state is not None:
                base = importr("base")
                base.set_seed(self.random_state)

            W, V, H, recerror, train_recerror, test_recerror, relchange = nnTensor.jNMF(
                X= transformed_Xs, M=transformed_mask, J=self.n_components,
                initW=init_W, initV=init_V, initH=init_H, fixW=False, fixV=False, fixH=False,
                L1_W=self.l1_W, L1_V=self.l1_V, L1_H=self.l1_H, L2_W=self.l2_W, L2_V= self.l2_V, L2_H=self.l2_H,
                w=weights, algorithm=beta_loss, p=self.p, thr = self.tol, num_iter=self.max_iter, verbose=self.verbose)

            H = [np.array(mat) for mat in H]
            if self.transform_ == "pandas":
                H = [pd.DataFrame(mat) for mat in H]
        else:
            raise ValueError("Only engine=='r' is currently supported.")

        self.H_ = H
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
        Xs : list of array-likes
            - Xs length: n_views
            - Xs[i] shape: (n_samples, n_features_i)
            A list of different views.

        Returns
        -------
        transformed_Xs : list of array-likes, shape (n_samples, n_components)
            The projected data.
        """
        Xs = check_Xs(Xs, force_all_finite='allow-nan')
        samples = Xs[0].index

        if self.engine == "r":
            from rpy2.robjects.packages import importr
            nnTensor = importr("nnTensor")
            transformed_Xs, transformed_mask, beta_loss, init_W, init_V, init_H, weights = self._prepare_variables(
                Xs=Xs, beta_loss=self.beta_loss, init_W=self.init_W, init_V=self.V_, init_H=self.H_, weights=self.weights)
            if self.random_state is not None:
                base = importr("base")
                base.set_seed(self.random_state)

            transformed_X = nnTensor.jNMF(X= transformed_Xs, M=transformed_mask, J=self.n_components,
                                          initW=init_W, initV=init_V, initH=_convert_df_to_r_object(self.H_),
                                          fixW=False, fixV=False, fixH=True,
                                          L1_W=self.l1_W, L1_V=self.l1_V, L1_H=self.l1_H,
                                          L2_W=self.l2_W, L2_V= self.l2_V, L2_H=self.l2_H,
                                          w=weights, algorithm=beta_loss, p=self.p, thr = self.tol, num_iter=self.max_iter,
                                          verbose=self.verbose)[0]

        transformed_X = np.array(transformed_X)
        if self.transform_ == "pandas":
            transformed_X = pd.DataFrame(transformed_X, index= samples)

        return transformed_X


    def set_output(self, *, transform=None):
        self.transform_ = "pandas"
        return self


    def _prepare_variables(self, Xs, beta_loss, init_W, init_V, init_H, weights):
        if self.engine == "r":
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

