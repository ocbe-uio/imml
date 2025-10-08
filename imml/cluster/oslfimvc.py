# License: BSD-3-Clause

import os
from os.path import dirname
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import KMeans
from sklearn.gaussian_process import kernels

from ..impute import get_observed_mod_indicator
from ..utils import check_Xs

matlabmodule_installed = False
oct2py_module_error = "Module 'matlab' needs to be installed. See https://imml.readthedocs.io/stable/main/installation.html#optional-dependencies"
try:
    import oct2py
    matlabmodule_installed = True
except ImportError:
    pass


class OSLFIMVC(BaseEstimator, ClassifierMixin):
    r"""
    One-Stage Incomplete Multi-View Clustering via Late Fusion (OS-LF-IMVC). [#oslfimvcpaper]_ [#oslfimvccode]_

    OS-LF-IMVC integrates the processes of imputing incomplete views and clustering into a cohesive optimization
    procedure. This approach enables the direct utilization of the learned consensus partition matrix to enhance
    the final clustering task.

    Parameters
    ----------
    n_clusters : int, default=8
        The number of clusters to generate.
    kernel : callable, default=kernels.Sum(kernels.DotProduct(), kernels.WhiteKernel())
        Specifies the kernel type to be used in the algorithm.
    lambda_reg : float, default=1.
        Regularization parameter. The algorithm demonstrated stable performance across a wide range of
        this hyperparameter.
    random_state : int, default=None
        Determines the randomness. Use an int to make the randomness deterministic.
    engine : str, default=matlab
        Engine to use for computing the model. Current options are 'matlab'.
    verbose : bool, default=False
        Verbosity mode.
    clean_space : bool, default=True
        If engine is 'matlab' and clean_space is True, the session will be closed after fitting the model.

    Attributes
    ----------
    labels_ : array-like of shape (n_samples,)
        Labels of each point in training data.
    embedding_ : np.array
        Consensus clustering matrix to be used as input for the KMeans clustering step.
    WP_ : array-like of shape (n_clusters, n_clusters, n_mods)
        p-th permutation matrix.
    C_ : array-like of shape (n_clusters, n_clusters)
        Centroids.
    beta_ : array-like of shape (n_mods,)
        Adaptive weights of clustering matrices.
    loss_ : array-like of shape (n_iter\_,)
        Values of the loss function.
    n_iter_ : int
        Number of iterations.

    References
    ----------
    .. [#oslfimvcpaper] Yi Zhang, Xinwang Liu, Siwei Wang, Jiyuan Liu, Sisi Dai, and En Zhu. 2021. One-Stage Incomplete
                        Multi-view Clustering via Late Fusion. In Proceedings of the 29th ACM International Conference
                        on Multimedia (MM '21). Association for Computing Machinery, New York, NY, USA, 2717–2725.
                        https://doi.org/10.1145/3474085.3475204.
    .. [#oslfimvccode] https://github.com/ethan-yizhang/OSLF-IMVC

    Example
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from imml.cluster import OSLFIMVC
    >>> Xs = [pd.DataFrame(np.random.default_rng(42).random((20, 10))) for i in range(3)]
    >>> estimator = OSLFIMVC(n_clusters = 2)
    >>> labels = estimator.fit_predict(Xs)

    """

    def __init__(self, n_clusters: int = 8, kernel: callable = kernels.Sum(kernels.DotProduct(), kernels.WhiteKernel()),
                 lambda_reg: float = 1., random_state:int = None, engine: str ="matlab",
                 verbose = False, clean_space: bool = True):
        if not isinstance(n_clusters, int):
            raise ValueError(f"Invalid n_clusters. It must be an int. A {type(n_clusters)} was passed.")
        if n_clusters < 2:
            raise ValueError(f"Invalid n_clusters. It must be an greater than 1. {n_clusters} was passed.")
        engines_options = ["matlab"]
        if engine not in engines_options:
            raise ValueError(f"Invalid engine. Expected one of {engines_options}. {engine} was passed.")
        if (engine == "matlab") and (not matlabmodule_installed):
            raise ImportError(oct2py_module_error)

        self.n_clusters = n_clusters
        self.kernel = kernel
        self.lambda_reg = lambda_reg
        self.random_state = random_state
        self.engine = engine
        self.verbose = verbose
        self.clean_space = clean_space

        if self.engine == "matlab":
            matlab_folder = dirname(__file__)
            matlab_folder = os.path.join(matlab_folder, "_" + (os.path.basename(__file__).split(".")[0]))
            self._matlab_folder = matlab_folder
            matlab_files = [x for x in os.listdir(matlab_folder) if x.endswith(".m")]
            self._oc = oct2py.Oct2Py(temp_dir= matlab_folder)
            for matlab_file in matlab_files:
                with open(os.path.join(matlab_folder, matlab_file)) as f:
                    self._oc.eval(f.read())
            self._oc.eval("pkg load statistics")


    def fit(self, Xs, y=None):
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
        self :  Fitted estimator.
        """
        Xs = check_Xs(Xs, ensure_all_finite='allow-nan')

        if self.engine=="matlab":
            observed_mod_indicator = get_observed_mod_indicator(Xs)
            if isinstance(observed_mod_indicator, pd.DataFrame):
                observed_mod_indicator = observed_mod_indicator.reset_index(drop=True)
            elif isinstance(observed_mod_indicator[0], np.ndarray):
                observed_mod_indicator = pd.DataFrame(observed_mod_indicator)
            s = [modality[modality == 0].index.values for _,modality in observed_mod_indicator.items()]
            transformed_Xs = [self.kernel(X) for X in Xs]
            transformed_Xs = np.array(transformed_Xs).swapaxes(0, -1)
            s = tuple([{"indx": i +1} for i in s])

            if self.random_state is not None:
                self._oc.rand('seed', self.random_state)
            U, C, WP, beta, obj = self._oc.OS_LF_IMVC_alg(transformed_Xs, s, self.n_clusters, self.lambda_reg, nout=5)
            beta = beta[:,0]
            obj = obj[0]

            if self.clean_space:
                self._clean_space()

        model = KMeans(n_clusters= self.n_clusters, n_init= "auto", random_state= self.random_state)
        self.labels_ = model.fit_predict(X= U)
        self.embedding_, self.WP_, self.C_, self.beta_, self.loss_ = U, WP, C, beta, obj
        self.n_iter_ = len(self.loss_)

        return self


    def _predict(self, Xs):
        r"""
        Return clustering results for samples.

        Parameters
        ----------
        Xs : list of array-likes objects
            - Xs length: n_mods
            - Xs[i] shape: (n_samples, n_features_i)

            A list of different modalities.

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
        Xs : list of array-likes objects
            - Xs length: n_mods
            - Xs[i] shape: (n_samples, n_features_i)

            A list of different modalities.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """

        labels = self.fit(Xs)._predict(Xs)
        return labels


    def _clean_space(self):
        [os.remove(os.path.join(self._matlab_folder, x)) for x in ["reader.mat", "writer.mat"]]
        self._oc.exit()
        del self._oc
        return None


