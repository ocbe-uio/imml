import os
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import KMeans

from ..impute import simple_view_imputer
from ..preprocessing import select_complete_samples
from ..utils import check_Xs, DatasetUtils


class SIMCADC(BaseEstimator, ClassifierMixin):
    r"""
    Scalable Incomplete Multiview Clustering with Adaptive Data Completion (SIMC-ADC).

    The SIMC-ADC algorithm captures the complementary information from different views by building a view-specific
    anchor graph. The anchor graph construction and a structure alignment are jointly optimized to enhance
    clustering quality.

    It is recommended to normalize (Normalizer or NormalizerNaN in case incomplete views) the data before applying
    this algorithm.

    Parameters
    ----------
    n_clusters : int, default=8
        The number of clusters to generate.
    lambda_parameter : float, default=1
        Balance the influence between anchor graph generation and alignment term.
    n_anchors : int, default=None
        Number of anchors. If None, use n_clusters.
    beta : float, default=1
        Balance the influence between anchor graph generation and alignment term.
    gamma : float, default=1
        Balance the influence between anchor graph generation and alignment term.
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
    U_ : np.array
        Basis matrix.
    V_ : np.array
        Commont latent feature matrix.
    A_ : np.array
        Learned anchors.
    Z_ : np.array
        View-specific anchor graph.
    loss_ : float
        Value of the loss function.
    iter_ : int
        Number of iterations.

    References
    ----------
    [paper] He, W.-J., Zhang, Z., & Wei, Y. (2023). Scalable incomplete multi-view clustering with adaptive data
            completion. Information Sciences, 649, 119562. doi:10.1016/j.ins.2023.119562.
    [code]  https://github.com/DarrenZZhang/INS23-SIMC_ADC

    Examples
    --------
    >>> from sklearn.pipeline import make_pipeline
    >>> from imvc.datasets import LoadDataset
    >>> from imvc.cluster import SIMCADC
    >>> from imvc.preprocessing import NormalizerNaN, MultiViewTransformer
    >>> Xs = LoadDataset.load_dataset(dataset_name="nutrimouse")
    >>> normalizer = NormalizerNaN().set_output(transform="pandas")
    >>> estimator = SIMCADC(n_clusters = 2)
    >>> pipeline = make_pipeline(MultiViewTransformer(normalizer), estimator)
    >>> labels = pipeline.fit_predict(Xs)
    """

    def __init__(self, n_clusters: int = 8, lambda_parameter: float = 1, n_anchors: int = None,
                 beta: float = 1, gamma: float = 1, eps: float = 1e-25, random_state:int = None,
                 engine: str ="matlab", verbose = False):
        self.n_clusters = n_clusters
        self.lambda_parameter = lambda_parameter
        self.beta = beta
        self.gamma = gamma
        self.eps = eps
        self.n_anchors = n_clusters if n_anchors is None else n_anchors
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
            matlab_folder = os.path.join("imvc", "cluster", "_simcadc")
            matlab_files = ["SIMC.m", "EProjSimplex_new.m"]
            oc = oct2py.Oct2Py(temp_dir= matlab_folder)
            for matlab_file in matlab_files:
                with open(os.path.join(matlab_folder, matlab_file)) as f:
                    oc.eval(f.read())

            mean_view_profile = [X.mean(axis=0).to_frame(X_id) for X_id, X in enumerate(select_complete_samples(Xs))]
            incomplete_samples = DatasetUtils.get_missing_samples_by_view(Xs=Xs, return_as_list=True)
            mean_view_profile = [pd.DataFrame(np.repeat(means, len(incom), axis= 1), columns=incom).values for means, incom in
                                  zip(mean_view_profile, incomplete_samples)]

            transformed_Xs = simple_view_imputer(Xs, value="zeros")
            transformed_Xs, mean_view_profile = tuple(transformed_Xs), tuple(mean_view_profile)

            w = [pd.DataFrame(np.eye(len(X)), index=X.index, columns=X.index) for X in Xs]
            w = [eye.loc[samples,:].values for eye, samples in zip(w, incomplete_samples)]
            w = tuple(w)

            n_incomplete_samples_view = list(len(incomplete_sample) for incomplete_sample in incomplete_samples)

            if self.random_state is not None:
                oc.rand('seed', self.random_state)
            u,v,a,w,z,iter,obj = oc.SIMC(transformed_Xs, len(Xs[0]), self.lambda_parameter,
                                                self.n_clusters, self.n_anchors, w, n_incomplete_samples_view,
                                                mean_view_profile, self.beta, self.gamma, nout=7)
        else:
            raise ValueError("Only engine=='matlab' is currently supported.")

        model = KMeans(n_clusters= self.n_clusters, random_state= self.random_state)
        self.labels_ = model.fit_predict(X= u)
        # todo
        self.U_ = u
        self.V_ = v
        self.A_ = a
        self.Z_, self.loss_, self.iter_ = z, obj, iter

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

