import os
from os.path import dirname

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import KMeans

from ..impute import get_observed_view_indicator
from ..utils import check_Xs, DatasetUtils


class IMSCAGL(BaseEstimator, ClassifierMixin):
    r"""
    Projective Incomplete Multi-View Clustering (PIMVC).

    The objective of PIMVC is to simultaneously discover the projection matrix for each view and establish a unified
    feature representation shared across incomplete multiple views, facilitating clustering. Essentially, PIMVC
    transforms the traditional multi-view matrix factorization model into a multi-view projection learning model. By
    consolidating various view-specific objective losses into a cohesive subspace of equal dimensions, it adeptly
    handles the challenge where a single view might overly influence consensus representation learning due to
    imbalanced information across views stemming from diverse dimensions. Furthermore, to capture the data geometric
    structure, PIMVC incorporates a penalty term for graph regularization.

    It is recommended to normalize (Normalizer or NormalizerNaN in case incomplete views) the data before applying
    this algorithm.

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
    neighbor_mode : str, default='KNN'
        Indicates how to construct the graph. Options are 'KNN' (default), and 'Supervised'.
    weight_mode : str, default='Binary'
        Indicates how to assign weights for each edge in the graph. Options are 'Binary' (default), 'Cosine' and 'HeatKernel'.
    max_iter : int, default=100
        Maximum number of iterations.
    random_state : int, default=None
        Determines the randomness. Use an int to make the randomness deterministic.
    engine : str, default=matlab
        Engine to use for computing the model. Currently only 'matlab' is supported.
.   verbose : bool, default=False
        Verbosity mode.

    Attributes
    ----------
    labels_ : array-like of shape (n_samples,)
        Labels of each point in training data.
    V_ : np.array
        Commont latent feature matrix.
    loss_ : float
        Value of the loss function.

    References
    ----------
    [paper] S. Deng, J. Wen, C. Liu, K. Yan, G. Xu and Y. Xu, "Projective Incomplete Multi-View Clustering," in IEEE
            Transactions on Neural Networks and Learning Systems, doi: 10.1109/TNNLS.2023.3242473.
    [code]  https://github.com/Dshijie/PIMVC

    Examples
    --------
    >>> from sklearn.pipeline import make_pipeline
    >>> from imvc.datasets import LoadDataset
    >>> from imvc.cluster import PIMVC
    >>> from imvc.preprocessing import NormalizerNaN, MultiViewTransformer
    >>> Xs = LoadDataset.load_dataset(dataset_name="nutrimouse")
    >>> normalizer = NormalizerNaN().set_output(transform="pandas")
    >>> estimator = PIMVC(n_clusters = 2)
    >>> pipeline = make_pipeline(MultiViewTransformer(normalizer), estimator)
    >>> labels = pipeline.fit_predict(Xs)
    """

    def __init__(self, n_clusters: int = 8, dele: float = 0.1, lamb: int = 100000, beta: int = 1, k: int = 3,
                 neighbor_mode: str = 'KNN', weight_mode: str = 'Binary', max_iter: int = 100,
                 random_state: int = None, engine: str = "matlab", verbose = False):
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
            matlab_folder = os.path.join(matlab_folder, "_pimvc")
            matlab_files = ["IMSAGL.m", "constructW.m", "EuDist2.m", "PCA1.m", "mySVD.m"]
            oc = oct2py.Oct2Py(temp_dir= matlab_folder)
            for matlab_file in matlab_files:
                with open(os.path.join(matlab_folder, matlab_file)) as f:
                    oc.eval(f.read())

            incomplete_samples = DatasetUtils.get_missing_samples_by_view(Xs=Xs, return_as_list=True)
            w = [pd.DataFrame(np.eye(len(X)), index=X.index, columns=X.index) for X in Xs]
            w = [eye.loc[samples,:].values for eye, samples in zip(w, incomplete_samples)]
            w = tuple(w)

            transformed_Xs = tuple([X.T for X in Xs])

            if self.random_state is not None:
                oc.rand('seed', self.random_state)
            v, loss = oc.IMSAGL(transformed_Xs, w, self.n_clusters, observed_view_indicator, self.lamb, self.beta,
                               self.max_iter,
                               {"NeighborMode": self.neighbor_mode, "WeightMode": self.weight_mode, "k": self.k}, nout=2)
        else:
            raise ValueError("Only engine=='matlab' is currently supported.")

        model = KMeans(n_clusters= self.n_clusters, random_state= self.random_state)
        v = v.T
        self.labels_ = model.fit_predict(X= v)
        self.V_ = v
        self.loss_ = loss

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
