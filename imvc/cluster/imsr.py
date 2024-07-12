import os
from os.path import dirname
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import KMeans

from ..impute import get_observed_view_indicator
from ..utils import check_Xs


class IMSR(BaseEstimator, ClassifierMixin):
    r"""
    Self-representation Subspace Clustering for Incomplete Multi-view Data (IMSR).

    IMSR performs feature extraction, imputation and self-representation learning to obtain a low-rank regularized
    consensus coefficient matrix.

    It is recommended to normalize (Normalizer or NormalizerNaN in case incomplete views) the data before applying
    this algorithm.

    Parameters
    ----------
    n_clusters : int, default=8
        The number of clusters to generate.
    lbd : float, default=1
        Positive trade-off parameter used for the optimization function. It is recommended to set from 0 to 1.
    gamma : float, default=1
        Positive trade-off parameter used for the optimization function. It is recommended to set from 0 to 1.
    random_state : int, default=None
        Determines the randomness. Use an int to make the randomness deterministic.
    engine : str, default=matlab
        Engine to use for computing the model. Current options are 'matlab'.
.   verbose : bool, default=False
        Verbosity mode.

    Attributes
    ----------
    labels_ : array-like of shape (n_samples,)
        Labels of each point in training data.
    Z_ : np.array
        Consensus coefficient matrix.
    loss_ : array-like of shape (n_views,)
        Value of the loss function.

    References
    ----------
    [paper] Jiyuan Liu, Xinwang Liu, Yi Zhang, Pei Zhang, Wenxuan Tu, Siwei Wang, Sihang Zhou, Weixuan Liang, Siqi
            Wang, and Yuexiang Yang. 2021. Self-Representation Subspace Clustering for Incomplete Multi-view Data. In
            Proceedings of the 29th ACM International Conference on Multimedia (MM '21). Association for Computing
            Machinery, New York, NY, USA, 2726–2734. https://doi.org/10.1145/3474085.3475379.
    [code]  https://github.com/liujiyuan13/IMSR-code_release

    Examples
    --------
    >>> from sklearn.pipeline import make_pipeline
    >>> from imvc.datasets import LoadDataset
    >>> from imvc.cluster import IMSR
    >>> from imvc.preprocessing import NormalizerNaN, MultiViewTransformer
    >>> Xs = LoadDataset.load_dataset(dataset_name="nutrimouse")
    >>> normalizer = NormalizerNaN().set_output(transform="pandas")
    >>> estimator = IMSR(n_clusters = 2)
    >>> pipeline = make_pipeline(MultiViewTransformer(normalizer), estimator)
    >>> labels = pipeline.fit_predict(Xs)
    """

    def __init__(self, n_clusters: int = 8, lbd : float = 1, gamma: float = 1, random_state:int = None,
                 engine: str ="matlab", verbose = False):
        self.n_clusters = n_clusters
        try:
            assert lbd > 0
        except AssertionError:
            raise ValueError("lbd should be a positive value.")
        try:
            assert gamma > 0
        except AssertionError:
            raise ValueError("gamma should be a positive value.")
        self.lbd = lbd
        self.gamma = gamma
        self.random_state = random_state
        self.engine = engine
        self.verbose = verbose


    def fit(self, Xs, y=None):
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
        self :  Fitted estimator.
        """
        Xs = check_Xs(Xs, force_all_finite='allow-nan')

        if self.engine=="matlab":
            import oct2py
            matlab_folder = dirname(__file__)
            matlab_folder = os.path.join(matlab_folder, "_imsr")
            matlab_files = ["IMSC.m", "update_Z.m", "update_X.m", "update_F.m", "init_Z.m",
                            "cal_obj.m", "baseline_spectral_onkernel.m"]
            oc = oct2py.Oct2Py(temp_dir= matlab_folder)
            for matlab_file in matlab_files:
                with open(os.path.join(matlab_folder, matlab_file)) as f:
                    oc.eval(f.read())

            observed_view_indicator = get_observed_view_indicator(Xs)
            if isinstance(observed_view_indicator, pd.DataFrame):
                observed_view_indicator = observed_view_indicator.reset_index(drop=True)
            elif isinstance(observed_view_indicator[0], np.ndarray):
                observed_view_indicator = pd.DataFrame(observed_view_indicator)
            observed_view_indicator = [(1 + missing_view[missing_view == 0].index).to_list() for _, missing_view in observed_view_indicator.items()]
            transformed_Xs = [X.T.values for X in Xs]

            if self.random_state is not None:
                oc.rand('seed', self.random_state)
            Z, obj = oc.IMSC(transformed_Xs, tuple(observed_view_indicator), self.n_clusters, self.lbd, self.gamma, nout=2)
        else:
            raise ValueError("Only engine=='matlab' is currently supported.")

        model = KMeans(n_clusters= self.n_clusters, random_state= self.random_state)
        self.labels_ = model.fit_predict(X= Z)
        self.Z_, self.loss_ = Z, obj

        return self


    def _predict(self, Xs):
        r"""
        Return clustering results for samples.

        Parameters
        ----------
        Xs : list of array-likes
            - Xs length: n_views
            - Xs[i] shape: (n_samples, n_features_i)
            A list of different views.

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
        Xs : list of array-likes
            - Xs length: n_views
            - Xs[i] shape: (n_samples, n_features_i)
            A list of different views.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """

        labels = self.fit(Xs)._predict(Xs)
        return labels
