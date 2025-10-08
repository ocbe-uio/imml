# License: BSD-3-Clause

import os
import contextlib
import tempfile
import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator

from ._mofa.run.entry_point import entry_point
from ._mofa._mofax import core as mfx
from ._mofa.core._BayesNet import BayesNet, StochasticBayesNet, _ModifiedBayesNet, _ModifiedStochasticBayesNet
from ..utils import check_Xs


class MOFA(TransformerMixin, BaseEstimator):
    r"""
    Multi-Omics Factor Analysis (MOFA). [#mofapaper1]_ [#mofapaper2]_ [#mofacode]_

    MOFA is a factor analysis model that provides a general framework for the integration of (originally, multi-omic
    data sets) incomplete multi-modal datasets, in an unsupervised fashion. Intuitively, MOFA can be viewed as a
    versatile and statistically rigorous generalization of principal component analysis to multi-modal data. Given
    several data matrices with measurements of multiple data types on the same or on overlapping sets of
    samples, MOFA infers an interpretable low-dimensional representation in terms of a few latent factors.

    It can deal with both modality- and feature-wise missing.

    Parameters
    ----------
    n_components : int, default=10
        Number of components to keep.
    impute : bool, default=True
        True if missing values should be imputed.
    data_options : dict, default=None
        Data processing options, such as scale_views and scale_groups.
    data_matrix : dict, default=None
        Keys such as likelihoods, view_names, etc.
    model_options : dict, default=None
        Model options, such as ard_factors or ard_weights.
    train_options : dict, default=None
        Keys such as iter, tolerance.
    stochastic_options : dict, default=None
        Stochastic variational inference options, such as learning rate or batch size.
    covariates : dict, default=None
        Slot to store sample covariate for training in MEFISTO. Keys are sample_cov and covariates_names.
    smooth_options : dict, default=None
        options for smooth inference, such as scale_cov or model_groups.
    random_state : int, default=None
        Determines the randomness. Use an int to make the randomness deterministic.
    verbose : bool, default=False
        Verbosity mode.

    Attributes
    ----------
    mofa_ : mofa object
        Entry point as the original library. This can be used for data analysis and explainability.
    factors_: array-like of shape (n_samples, n_components)
        Factors computed by the model.
    weights_: list of n_mods array-likes objects of shape (n_features_i, n_components)
        Weights of the MOFA model.

    References
    ----------
    .. [#mofapaper1] Argelaguet R, Velten B, Arnol D, Dietrich S, Zenz T, Marioni JC, Buettner F, Huber W, Stegle O
                    (2018). “Multi‐Omics Factor Analysis—a framework for unsupervised integration of multi‐omics data
                    sets.” Molecular Systems Biology, 14. doi:10.15252/msb.20178124.
    .. [#mofapaper2] Argelaguet R, Arnol D, Bredikhin D, Deloro Y, Velten B, Marioni JC, Stegle O (2020). “MOFA+: a
                     statistical framework for comprehensive integration of multi-modal single-cell data.” Genome
                     Biology, 21. doi:10.1186/s13059-020-02015-1.
    .. [#mofacode] https://biofam.github.io/MOFA2/index.html

    Example
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from imml.decomposition import MOFA
    >>> Xs = [pd.DataFrame(np.random.default_rng(42).random((20, 10))) for i in range(3)]
    >>> transformer = MOFA(n_components = 5)
    >>> transformed_Xs = transformer.fit_transform(Xs)
    """

    
    def __init__(self, n_components : int = 10, impute:bool = True,
                 data_options : dict = None, data_matrix : dict = None, model_options : dict = None,
                 train_options : dict = None, stochastic_options : dict = None, covariates : dict = None,
                 smooth_options : dict = None, random_state : int = None, verbose = False):

        if data_options is None:
            data_options = {}
        if data_matrix is None:
            data_matrix = {}
        if model_options is None:
            model_options = {}
        if train_options is None:
            train_options = {}
        if stochastic_options is None:
            stochastic_options = {}
        if covariates is None:
            covariates = {}
        if smooth_options is None:
            smooth_options = {}

        if not isinstance(n_components, int):
            raise ValueError(f"Invalid n_components. It must be an int. A {type(n_components)} was passed.")
        if n_components < 1:
            raise ValueError(f"Invalid n_components. It must be greater or equal to 1. {n_components} was passed.")
        self.n_components = n_components
        self.impute = impute
        self.random_state = random_state
        self.verbose = verbose        
        if self.verbose:
            self.mofa_ = entry_point()
        else:
            with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                self.mofa_ = entry_point()
        self.data_options = data_options
        self.data_matrix = data_matrix
        self.model_options = model_options
        self.train_options = train_options
        self.stochastic_options = stochastic_options
        self.covariates = covariates
        self.smooth_options = smooth_options
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
        if self.verbose:
            self._run_mofa(data = [[X] for X in Xs])
        else:
            with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                self._run_mofa(data = [[X] for X in Xs])
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            with tempfile.TemporaryDirectory() as tmp:
                outfile = os.path.join(tmp, 'tmp.hdf5')
                self.mofa_.save(outfile=outfile, save_data=True, save_parameters=False, expectations=None)
                model = mfx.mofa_model(outfile)
                self.weights_ = model.get_weights(concatenate_views= False)
                self.factors_ = model.get_factors()
                model.close()
        return self


    def transform(self, Xs):
        r"""
        Project data into the learned space.

        Parameters
        ----------
        Xs : list of array-likes objects
            - Xs length: n_mods
            - Xs[i] shape: (n_samples_i, n_features_i)
            A list of different mods.

        Returns
        -------
        transformed_Xs : list of n_mods array-likes objects of shape (n_samples, n_components)
            The projected data.
        """
        Xs = check_Xs(Xs, ensure_all_finite='allow-nan')
        if not isinstance(Xs[0], pd.DataFrame):
            Xs = [pd.DataFrame(X) for X in Xs]

        ws = self.weights_
        winv = [np.linalg.pinv(w) for w in ws]
        transformed_Xs = [np.dot(X, w.T) for X,w in zip(Xs, winv)]

        if self.transform_ == "pandas":
            transformed_Xs = [pd.DataFrame(transformed_X, index=X.index) for X,transformed_X in zip(Xs,transformed_Xs)]
        return transformed_Xs


    def fit_transform(self, Xs, y = None, **fit_params):
        r"""
        Fit to data, then transform it.

        Parameters
        ----------
        Xs : list of array-likes objects
            - Xs length: n_mods
            - Xs[i] shape: (n_samples_i, n_features_i)
            A list of different mods.
        y : Ignored
            Not used, present here for API consistency by convention.
        fit_params : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        transformed_X : array-likes objects of shape (n_samples, n_components)
            The projected data.
        """
        transformed_X = self.fit(Xs).factors_
        if self.transform_ == "pandas":
            transformed_X = pd.DataFrame(transformed_X, index=Xs[0].index)
        return transformed_X

    
    def _run_mofa(self, data):
        self.mofa_.set_data_options(**self.data_options)
        self.mofa_.set_data_matrix(data = data, **self.data_matrix)
        self.mofa_.set_model_options(factors = self.n_components, **self.model_options)
        self.mofa_.set_train_options(seed = self.random_state, verbose = self.verbose, **self.train_options)
        self.mofa_.set_stochastic_options(**self.stochastic_options)
        if self.covariates:
            self.mofa_.set_covariates(**self.covariates)
            self.mofa_.set_smooth_options(**self.smooth_options)
        self.mofa_.build()
        if isinstance(self.mofa_.model, BayesNet):
            self.mofa_.model = _ModifiedBayesNet(self.mofa_.model.dim, self.mofa_.model.nodes)
        elif isinstance(self.mofa_.model, StochasticBayesNet):
            self.mofa_.model = _ModifiedStochasticBayesNet(self.mofa_.model.dim, self.mofa_.model.nodes)
        self.mofa_.run()
        return None

