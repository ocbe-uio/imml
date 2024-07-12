import os
from os.path import dirname

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import KMeans
from sklearn.gaussian_process import kernels

from ..impute import get_observed_view_indicator
from ..utils import check_Xs


class EEIMVC(BaseEstimator, ClassifierMixin):
    r"""
    Efficient and Effective Incomplete Multi-view Clustering (EE-IMVC).

    EE-IMVC impute missing views with a consensus clustering matrix that is regularized with prior knowledge.

    Parameters
    ----------
    n_clusters : int, default=8
        The number of clusters to generate.
    kernel : callable, default=kernels.Sum(kernels.DotProduct(), kernels.WhiteKernel())
        Specifies the kernel type to be used in the algorithm.
    lambda_reg : float, default=1.
        Regularization parameter. The algorithm demonstrated stable performance across a wide range of
        this hyperparameter.
    qnorm : float, default=2.
        Regularization parameter. The algorithm demonstrated stable performance across a wide range of
        this hyperparameter.
    random_state : int, default=None
        Determines the randomness. Use an int to make the randomness deterministic.
    engine : str, default=matlab
        Engine to use for computing the model.
.   verbose : bool, default=False
        Verbosity mode.

    Attributes
    ----------
    labels_ : array-like of shape (n_samples,)
        Labels of each point in training data.
    H_ : array-like
        Consensus clustering matrix.
    WP_ : array-like
        p-th permutation matrix.
    HP_ : array-like
        missing part of the p-th base clustering matrix.
    beta_ : array-like
        Adaptive weights of clustering matrices.
    loss_ : float
        Value of the loss function.

    References
    ----------
    [paper] X. Liu et al., "Efficient and Effective Regularized Incomplete Multi-View Clustering," in IEEE
             Transactions on Pattern Analysis and Machine Intelligence, vol. 43, no. 8, pp. 2634-2646, 1 Aug. 2021,
             doi: 10.1109/TPAMI.2020.2974828.
    [code]   https://github.com/xinwangliu/TPAMI_EEIMVC

    Examples
    --------
    >>> from sklearn.pipeline import make_pipeline
    >>> from imvc.datasets import LoadDataset
    >>> from imvc.cluster import EEIMVC
    >>> from sklearn.preprocessing import StandardScaler
    >>> from imvc.preprocessing import MultiViewTransformer
    >>> Xs = LoadDataset.load_dataset(dataset_name="nutrimouse")
    >>> normalizer = StandardScaler().set_output(transform="pandas")
    >>> estimator = EEIMVC(n_clusters = 2)
    >>> pipeline = make_pipeline(MultiViewTransformer(normalizer), estimator)
    >>> labels = estimator.fit_predict(Xs)
    """

    def __init__(self, n_clusters: int = 8, kernel: callable = kernels.Sum(kernels.DotProduct(), kernels.WhiteKernel()),
                 lambda_reg: float = 1., qnorm: float = 2., random_state: int = None,
                 engine: str ="matlab", verbose = False):
        self.n_clusters = n_clusters
        self.qnorm = qnorm
        self.kernel = kernel
        self.lambda_reg = lambda_reg
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
            matlab_folder = os.path.join(matlab_folder, "_eeimvc")
            matlab_files = ['incompleteLateFusionMKCOrthHp_lambda.m', 'mycombFun.m', 'myInitializationHp.m',
                            'mykernelkmeans.m', 'updateBetaAbsentClustering.m', 'updateHPabsentClusteringOrthHp.m',
                            'updateWPabsentClusteringV1.m', 'algorithm2.m']
            oc = oct2py.Oct2Py(temp_dir= matlab_folder)
            for matlab_file in matlab_files:
                with open(os.path.join(matlab_folder, matlab_file)) as f:
                    oc.eval(f.read())

            observed_view_indicator = get_observed_view_indicator(Xs)
            if isinstance(observed_view_indicator, pd.DataFrame):
                observed_view_indicator = observed_view_indicator.reset_index(drop=True)
            elif isinstance(observed_view_indicator[0], np.ndarray):
                observed_view_indicator = pd.DataFrame(observed_view_indicator)
            s = [view[view == 0].index.values for _,view in observed_view_indicator.items()]
            transformed_Xs = [self.kernel(X) for X in Xs]
            transformed_Xs = np.array(transformed_Xs).swapaxes(0, -1)
            transformed_Xs = np.nan_to_num(transformed_Xs, nan=0)
            s = tuple([{"indx": i +1} for i in s])

            if self.random_state is not None:
                oc.rand('seed', self.random_state)
            H_normalized,WP,HP,beta,obj = oc.incompleteLateFusionMKCOrthHp_lambda(transformed_Xs, s, self.n_clusters,
                                                                                  self.qnorm, self.lambda_reg, nout=5)
        else:
            raise ValueError("Only engine=='matlab' is currently supported.")

        model = KMeans(n_clusters= self.n_clusters, random_state= self.random_state)
        self.labels_ = model.fit_predict(X= H_normalized)
        self.H_, self.WP_, self.HP_, self.beta_, self.loss_ = H_normalized, WP, HP, beta, obj

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
