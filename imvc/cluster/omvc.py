import os
from os.path import dirname
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import KMeans

from ..utils import check_Xs, DatasetUtils


class OMVC(BaseEstimator, ClassifierMixin):
    r"""
    Online multi-view clustering (OMVC).

    OMVC aims to learn latent feature matrices for all views while driving them towards a consensus. To enhance the
    robustness of these learned matrices, it incorporates lasso regularization. Additionally, to mitigate the impact of
    incomplete data, it introduces dynamic weight adjustment.

    Parameters
    ----------
    n_clusters : int, default=8
        The number of clusters to generate.
    max_iter : int, default=3
        Maximum number of iterations.
    tol : float, default=1e-4
        Tolerance of the stopping condition.
    block_size : int, default=50
        Size of the chunk.
    n_pass : int, default=1
        Number of passes.
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
        Latent feature matrix.
    U_star_loss_ : np.array
        Common consensus, latent feature matrix across all the views.
    loss_ : float
        Value of the loss function.

    References
    ----------
    [paper] W. Shao, L. He, C. -t. Lu and P. S. Yu, "Online multi-view clustering with incomplete views," 2016 IEEE
             International Conference on Big Data (Big Data), Washington, DC, USA, 2016, pp. 1012-1017,
             doi: 10.1109/BigData.2016.7840701.
    [code]  https://github.com/software-shao/online-multiview-clustering-with-incomplete-view

    Examples
    --------
    >>> from sklearn.pipeline import make_pipeline
    >>> from imvc.datasets import LoadDataset
    >>> from imvc.cluster import OMVC
    >>> from sklearn.preprocessing import StandardScaler
    >>> from imvc.preprocessing import MultiViewTransformer
    >>> Xs = LoadDataset.load_dataset(dataset_name="nutrimouse")
    >>> normalizer = StandardScaler().set_output(transform="pandas")
    >>> estimator = OMVC(n_clusters = 2)
    >>> pipeline = make_pipeline(MultiViewTransformer(normalizer), estimator)
    >>> labels = estimator.fit_predict(Xs)

    """

    def __init__(self, n_clusters: int = 8, max_iter: int = 200, tol: float = 1e-4, decay: float = 1,
                 block_size: int = 50, n_pass: int = 1, random_state:int = None,
                 engine: str ="matlab", verbose = False):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.decay = decay
        self.block_size = block_size
        self.n_pass = n_pass
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
            matlab_folder = os.path.join(matlab_folder, "_omvc")
            matlab_files = ["objective_ONMF_Multi.m", "ONMF_Multi_PGD_search.m"]
            oc = oct2py.Oct2Py(temp_dir= matlab_folder)
            for matlab_file in matlab_files:
                with open(os.path.join(matlab_folder, matlab_file)) as f:
                    oc.eval(f.read())

            n_views = len(Xs)
            ones = np.ones((n_views, 1))
            option = {"k": self.n_clusters, "maxiter": self.max_iter, "tol": self.tol, "num_cluster": self.n_clusters,
                      "decay": self.decay, "alpha": 1e-2*ones, "beta": 1e-7*ones,
                      "pass": self.n_pass}

            if isinstance(Xs[0], pd.DataFrame):
                transformed_Xs = [X.values for X in Xs]
            elif isinstance(Xs[0], np.ndarray):
                transformed_Xs = Xs
            missing_samples_by_view = DatasetUtils.get_missing_samples_by_view(Xs=transformed_Xs, return_as_list=True)
            missing_samples_by_view = tuple([np.array(missing_samples)+1 for missing_samples in missing_samples_by_view])
            transformed_Xs = [np.nan_to_num(np.clip(X, a_min=0, a_max=None), nan=0.0) for X in transformed_Xs]
            transformed_Xs = [X/(X.sum().sum()) for X in transformed_Xs]

            if self.random_state is not None:
                oc.rand('seed', self.random_state)
            u, v, u_star_loss, loss = oc.ONMF_Multi_PGD_search(transformed_Xs, option, len(Xs[0]),
                                                               missing_samples_by_view, self.block_size, nout=4)
            u_star_loss = u_star_loss[self.n_pass-1]
        else:
            raise ValueError("Only engine=='matlab' is currently supported.")

        model = KMeans(n_clusters= self.n_clusters, random_state= self.random_state)
        self.labels_ = model.fit_predict(X= u_star_loss)
        self.U_ = u
        self.V_ = v
        self.U_star_loss_ = u_star_loss
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
