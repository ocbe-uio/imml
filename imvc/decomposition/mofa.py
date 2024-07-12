import copy
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
    Multi-Omics Factor Analysis (MOFA).

    MOFA is a factor analysis model that provides a general framework for the integration of (originally, multi-omic
    data sets) incomplete multi-view datasets, in an unsupervised fashion. Intuitively, MOFA can be viewed as a
    versatile and statistically rigorous generalization of principal component analysis to multi-views data. Given
    several data matrices with measurements of multiple -views data types on the same or on overlapping sets of
    samples, MOFA infers an interpretable low-dimensional representation in terms of a few latent factors.

    Parameters
    ----------
    factors : int, default=10
        The number of clusters to generate. If it is not provided, it will use the default one from the algorithm.
    impute : bool, default=True
        True if missing values should be imputed.
    data_options : dict (default={})
        Data processing options, such as scale_views and scale_groups.
    data_matrix : dict (default={})
        Keys such as likelihoods, view_names, etc.
    model_options : dict (default={})
        Model options, such as ard_factors or ard_weights.
    train_options : dict (default={})
        Keys such as iter, tolerance.
    stochastic_options : dict (default={})
        Stochastic variational inference options, such as learning rate or batch size.
    covariates : dict (default={})
        Slot to store sample covariate for training in MEFISTO. Keys are sample_cov and covariates_names.
    smooth_options : dict (default={})
        options for smooth inference, such as scale_cov or model_groups.
    random_state : int (default=None)
        Determines the randomness. Use an int to make the randomness deterministic.
    verbose : bool, default=False
        Verbosity mode.

    Attributes
    ----------
    mofa_ : mofa object
        Entry point as the original library. This can be used for data analysis and explainability.
    mofa_model_ : Class around HDF5-based model on disk
        This can be used for data analysis and explainability. It provides utility functions to get factors, weights,
         features, and samples info in the form of Pandas dataframes, and data as a NumPy array.
    weights_: ndarray
        Weights of the MOFA model.

    References
    ----------
    [paper1] Argelaguet R, Velten B, Arnol D, Dietrich S, Zenz T, Marioni JC, Buettner F, Huber W, Stegle O
        (2018). “Multi‐Omics Factor Analysis—a framework for unsupervised integration of multi‐omics data sets.”
        Molecular Systems Biology, 14. doi:10.15252/msb.20178124.
    [paper2] Argelaguet R, Arnol D, Bredikhin D, Deloro Y, Velten B, Marioni JC, Stegle O (2020). “MOFA+: a statistical
        framework for comprehensive integration of multi-modal single-cell data.” Genome Biology, 21.
        doi:10.1186/s13059-020-02015-1.
    [url] https://biofam.github.io/MOFA2/index.html

    Examples
    --------
    >>> from imvc.datasets import LoadDataset
    >>> from imvc.decomposition import MOFA
    >>> Xs = LoadDataset.load_dataset(dataset_name="nutrimouse")
    >>> pipeline = MOFA().fit(Xs)
    >>> transformed_Xs = pipeline.transform(Xs)
    """

    
    def __init__(self, factors : int = 10, impute:bool = True, data_options = {}, data_matrix = {}, model_options = {},
                 train_options = {}, stochastic_options = {}, covariates = {}, smooth_options = {},
                 random_state : int = None, verbose = False):
        self.factors = factors
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
        Xs : list of array-likes
            - Xs length: n_views
            - Xs[i] shape: (n_samples_i, n_features_i)
            A list of different views.
        y : array-like, shape (n_samples,)
            Labels for each sample. Only used by supervised algorithms.

        Returns
        -------
        self :  returns and instance of self.
        """
        Xs = check_Xs(Xs, force_all_finite='allow-nan')
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
                self.factors_ = model.get_factors(concatenate_groups= False)
                self._columns = model._check_factors(np.arange(self.factors).tolist())[1]
                model.close()
        return self


    def transform(self, Xs):
        r"""
        Project data into the learned space.

        Parameters
        ----------
        Xs : list of array-likes
            - Xs length: n_views
            - Xs[i] shape: (n_samples_i, n_features_i)
            A list of different views.

        Returns
        -------
        transformed_Xs : list of array-likes, shape (n_samples, n_features)
            The projected data.
        """
        ws = self.weights_
        winv = [np.linalg.pinv(w) for w in ws]
        transformed_Xs = [np.dot(X, w.T) for X,w in zip(Xs, winv)]
        if self.impute:
            imputed_Xs = copy.deepcopy(Xs)
            for idx, (transformed_X, w) in enumerate(zip(transformed_Xs, ws)):
                imputed_X = np.dot(np.nan_to_num(transformed_X, nan=0.0), w.T)
                imputed_X = pd.DataFrame(imputed_X, columns=imputed_Xs[idx].columns)
                imputed_Xs[idx] = imputed_Xs[idx].fillna(imputed_X)
            transformed_Xs = [np.dot(X, w.T) for X, w in zip(imputed_Xs, winv)]

        if self.transform_ == "pandas":
            transformed_Xs = [pd.DataFrame(transformed_X, columns=self._columns, index=X.index)
                              for X,transformed_X in zip(Xs,transformed_Xs)]
        return transformed_Xs

    
    def _run_mofa(self, data):
        self.mofa_.set_data_options(**self.data_options)
        self.mofa_.set_data_matrix(data = data, **self.data_matrix)
        self.mofa_.set_model_options(factors = self.factors, **self.model_options)
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


    def get_feature_names_out(self, *args, **params):
        return self._columns

    def set_output(self, *, transform=None):
        self.transform_ = "pandas"
        return self


