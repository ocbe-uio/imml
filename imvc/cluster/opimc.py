import os
from os.path import dirname
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

from ..impute import get_observed_view_indicator, simple_view_imputer
from ..utils import check_Xs


class OPIMC(BaseEstimator, ClassifierMixin):
    r"""
    One-Pass Incomplete Multi-View Clustering (OPIMC).

    OPIMC deals with large scale incomplete multi-view clustering problem by considering the instance missing
    information with the help of regularized matrix factorization and weighted matrix factorization.

    It is recommended to normalize (Normalizer or NormalizerNaN in case incomplete views) the data before applying
    this algorithm.

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
    block_size : int, default=50
        Size of the chunk.
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

    References
    ----------
    [paper1] Hu, M., & Chen, S. (2019). One-Pass Incomplete Multi-View Clustering. Proceedings of the AAAI Conference
             on Artificial Intelligence, 33(01), 3838-3845. https://doi.org/10.1609/aaai.v33i01.33013838.
.
    [paper2] Jie Wen, Zheng Zhang, Lunke Fei, Bob Zhang, Yong Xu, Zhao Zhang, Jinxing Li, A Survey on Incomplete
             Multi-view Clustering, IEEE TRANSACTIONS ON SYSTEMS, MAN, AND CYBERNETICS: SYSTEMS, 2022.
    [code]  https://github.com/DarrenZZhang/Survey_IMC

    Examples
    --------
    >>> from sklearn.pipeline import make_pipeline
    >>> from imvc.datasets import LoadDataset
    >>> from imvc.cluster import OPIMC
    >>> from imvc.preprocessing import NormalizerNaN, MultiViewTransformer
    >>> Xs = LoadDataset.load_dataset(dataset_name="nutrimouse")
    >>> normalizer = NormalizerNaN().set_output(transform="pandas")
    >>> estimator = OPIMC(n_clusters = 2)
    >>> pipeline = make_pipeline(MultiViewTransformer(normalizer), estimator)
    >>> labels = pipeline.fit_predict(Xs)
    """

    def __init__(self, n_clusters: int = 8, alpha: float = 10, num_passes: int = 1, max_iter: int = 30,
                 tol: float = 1e-6, block_size: int = 250, random_state:int = None, engine: str ="matlab", verbose = False):
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.num_passes = num_passes
        self.max_iter = max_iter
        self.tol = tol
        self.block_size = block_size
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
            matlab_folder = os.path.join(matlab_folder, "_opimc")
            matlab_files = ["UpdateV.m", "OPIMC.m", "NormalizeFea.m"]
            oc = oct2py.Oct2Py(temp_dir= matlab_folder)
            for matlab_file in matlab_files:
                with open(os.path.join(matlab_folder, matlab_file)) as f:
                    oc.eval(f.read())

            if isinstance(Xs[0], pd.DataFrame):
                transformed_Xs = [X.values for X in Xs]
            elif isinstance(Xs[0], np.ndarray):
                transformed_Xs = Xs
            observed_view_indicator = get_observed_view_indicator(transformed_Xs)
            transformed_Xs = simple_view_imputer(transformed_Xs, value="zeros")
            transformed_Xs = [X.T for X in transformed_Xs]
            transformed_Xs = tuple(transformed_Xs)

            w = tuple([oc.diag(missing_view) for missing_view in observed_view_indicator])
            options = {"block_size": self.block_size, "k": self.n_clusters, "maxiter": self.max_iter,
                       "tol": self.tol, "pass": self.num_passes, "loss": 0, "alpha": self.alpha}
            if self.random_state is not None:
                oc.rand('seed', self.random_state)
            labels = oc.OPIMC(transformed_Xs, w, options, observed_view_indicator)
        else:
            raise ValueError("Only engine=='matlab' is currently supported.")

        self.labels_ = pd.factorize(labels[:,0])[0]

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
