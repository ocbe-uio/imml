# License: BSD-3-Clause

import os
from os.path import dirname
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import KMeans
from sklearn.gaussian_process import kernels

from ..utils import check_Xs_y
from ..explore import get_missing_samples_by_mod
from .. import octavemodule_installed, oct2py_module_error

if octavemodule_installed:
    import oct2py


class MKKMIK(BaseEstimator, ClusterMixin):
    r"""
    Multiple Kernel K-Means with Incomplete Kernels (MKKM-IK). [#mkkmikpaper]_ [#mkkmikcode]_

    MKKM-IK integrates imputation and clustering into a single optimization procedure. Thus, the clustering result
    guides the missing kernel imputation, and the latter is used to conduct the subsequent clustering. Both procedures
    will be performed until convergence.

    Parameters
    ----------
    n_clusters : int, default=8
        The number of clusters to generate.
    kernel : callable, default=None
        Specifies the kernel type to be used in the algorithm. It uses dot product kernel by default.
    kernel_initialization : str, default="zeros"
        Specifies the algorithm to initialize the kernel. It should be one of ['zeros', 'mean', 'knn', 'em', 'laplacian'].
    lambda_reg : float, default=1.
        Regularization parameter. The algorithm demonstrated stable performance across a wide range of
        this hyperparameter.
    qnorm : float, default=2.
        Regularization parameter. The algorithm demonstrated stable performance across a wide range of
        this hyperparameter.
    random_state : int, default=None
        Determines the randomness. Use an int to make the randomness deterministic.
    engine : str, default='octave'
        Engine to use for computing the model. Current options are 'octave'.
    verbose : bool, default=False
        Verbosity mode.
    clean_space : bool, default=True
        If engine is 'octave' and clean_space is True, the session will be closed after fitting the model.

    Attributes
    ----------
    labels_ : array-like of shape (n_samples,)
        Labels of each point in training data.
    embedding_ : array-like of shape (n_samples, n_clusters)
        Consensus clustering matrix to be used as input for the KMeans clustering step.
    gamma_ : array-like of shape (n_mods,)
        Kernel weights.
    KA_ : array-like of shape (n_samples, n_mods)
        Kernel sub-matrix.
    loss_ : array-like of shape (n_iter\_,)
        Values of the loss function.
    n_iter_ : int
        Number of iterations.

    References
    ----------
    .. [#mkkmikpaper] X. Liu et al., "Multiple Kernel k-Means with Incomplete Kernels," in IEEE Transactions on Pattern
                      Analysis and Machine Intelligence, vol. 42, no. 5, pp. 1191-1204, 1 May 2020,
                      doi: 10.1109/TPAMI.2019.2892416.
    .. [#mkkmikcode] https://github.com/wangsiwei2010/multiple_kernel_clustering_with_absent_kernel

    Example
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from imml.cluster import MKKMIK
    >>> Xs = [pd.DataFrame(np.random.default_rng(42).random((20, 10))) for i in range(3)]
    >>> estimator = MKKMIK(n_clusters = 2)
    >>> labels = estimator.fit_predict(Xs)

    """

    def __init__(self, n_clusters: int = 8, kernel_initialization: str = "zeros",
                 kernel: callable = None,
                 qnorm: float = 2., random_state: int = None, engine: str = "octave",
                 verbose=False, clean_space: bool = True):

        if not isinstance(n_clusters, int):
            raise ValueError(f"Invalid n_clusters. It must be an int. A {type(n_clusters)} was passed.")
        if n_clusters < 2:
            raise ValueError(f"Invalid n_clusters. It must be an greater than 1. {n_clusters} was passed.")
        engines_options = ["octave"]
        if engine not in engines_options:
            raise ValueError(f"Invalid engine. Expected one of {engines_options}. {engine} was passed.")
        if (engine == "octave") and (not octavemodule_installed):
            raise ImportError(oct2py_module_error)
        kernel_initializations = ['zeros', 'mean', 'knn', 'em', 'laplacian']
        if kernel_initialization not in kernel_initializations:
            raise ValueError(f"Invalid kernel_initialization. Expected one of: {kernel_initializations}")

        if kernel is None:
            kernel = kernels.Sum(kernels.DotProduct(), kernels.WhiteKernel())

        self.n_clusters = n_clusters
        self.kernel_initialization = kernel_initialization
        self.qnorm = qnorm
        self.kernel = kernel
        self.random_state = random_state
        self.engine = engine
        self.verbose = verbose
        self.kernel_initializations = {"zeros": "algorithm2", "mean": "algorithm3", "knn": "algorithm0",
                                       "em": "algorithm6", "laplacian": "algorithm4"}
        self.clean_space = clean_space

        if self.engine == "octave":
            octave_folder = dirname(__file__)
            octave_folder = os.path.join(octave_folder, "_" + (os.path.basename(__file__).split(".")[0]))
            self._octave_folder = octave_folder
            octave_files = [x for x in os.listdir(octave_folder) if x.endswith(".m")]
            self._oc = oct2py.Oct2Py(temp_dir= octave_folder)
            for octave_file in octave_files:
                with open(os.path.join(octave_folder, octave_file)) as f:
                    self._oc.eval(f.read())

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
        Xs = check_Xs_y(Xs, ensure_all_finite='allow-nan')

        if self.engine == "octave":
            if isinstance(Xs[0], pd.DataFrame):
                transformed_Xs = [X.values for X in Xs]
            elif isinstance(Xs[0], np.ndarray):
                transformed_Xs = Xs
            s = get_missing_samples_by_mod(Xs=transformed_Xs, return_as_list=True)
            s = tuple([{"indx": pd.Series(i).add(1).to_list()} for i in s])

            transformed_Xs = [self.kernel(X) for X in transformed_Xs]
            transformed_Xs = np.array(transformed_Xs).swapaxes(0, -1)
            kernel = self.kernel_initializations[self.kernel_initialization]

            if self.random_state is not None:
                self._oc.rand('seed', self.random_state)
            H_normalized,gamma,obj,KA = self._oc.myabsentmultikernelclustering(transformed_Xs, s, self.n_clusters,
                                                                         self.qnorm, kernel, nout=4)
            KA = KA[:, 0]
            obj = obj[0]

            if self.clean_space:
                self._clean_space()

        model = KMeans(n_clusters=self.n_clusters, n_init="auto", random_state=self.random_state)
        self.labels_ = model.fit_predict(X=H_normalized)
        self.embedding_, self.gamma_, self.KA_, self.loss_ = H_normalized, gamma, KA, obj
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
        [os.remove(os.path.join(self._octave_folder, x)) for x in ["reader.mat", "writer.mat"]]
        self._oc.exit()
        del self._oc
        return None

