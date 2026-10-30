# License: BSD-3-Clause

import os
from os.path import dirname
import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import KMeans
from sklearn.gaussian_process import kernels
from scipy.sparse.linalg import eigsh
from numpy.linalg import svd

from ..impute import get_observed_mod_indicator
from ..utils import check_Xs_y, as_pd_X, as_np_Xs
from .. import octavemodule_installed, oct2py_module_error

if octavemodule_installed:
    import oct2py


class OSLFIMVC(BaseEstimator, ClusterMixin):
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
    engine : str, default='python'
        Engine to use for computing the model. Current options are 'octave' or 'python'.
    verbose : bool, default=False
        Verbosity mode.
    clean_space : bool, default=True
        If engine is 'octave' and clean_space is True, the session will be closed after fitting the model.

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
                 lambda_reg: float = 1., random_state:int = None, engine: str ="python",
                 verbose = False, clean_space: bool = True):
        if not isinstance(n_clusters, int):
            raise ValueError(f"Invalid n_clusters. It must be an int. A {type(n_clusters)} was passed.")
        if n_clusters < 2:
            raise ValueError(f"Invalid n_clusters. It must be an greater than 1. {n_clusters} was passed.")
        engines_options = ["octave", "python"]
        if engine not in engines_options:
            raise ValueError(f"Invalid engine. Expected one of {engines_options}. {engine} was passed.")
        if (engine == "octave") and (not octavemodule_installed):
            raise ImportError(oct2py_module_error)

        self.n_clusters = n_clusters
        self.kernel = kernel
        self.lambda_reg = lambda_reg
        self.random_state = random_state
        self.engine = engine
        self.verbose = verbose
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
        Xs = check_Xs_y(Xs, ensure_all_finite='allow-nan')

        observed_mod_indicator = as_pd_X(get_observed_mod_indicator(Xs))
        s = [modality[modality == 0].index.values for _, modality in observed_mod_indicator.items()]
        Xs_numpy = as_np_Xs(Xs)
        transformed_Xs = [self.kernel(X) for X in Xs_numpy]
        transformed_Xs = np.array(transformed_Xs).swapaxes(0, -1)
        s = tuple([{"indx": i + 1} for i in s])

        if self.engine=="octave":
            if self.random_state is not None:
                self._oc.rand('seed', self.random_state)
            U, C, WP, beta, obj = self._oc.OS_LF_IMVC_alg(transformed_Xs, s, self.n_clusters, self.lambda_reg, nout=5)
            beta = beta[:,0]
            obj = obj[0]

            if self.clean_space:
                self._clean_space()

        elif self.engine=="python":
            U, C, WP, beta, obj = self._os_lf_imvc_alg(transformed_Xs, s, self.n_clusters, self.lambda_reg)

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
        [os.remove(os.path.join(self._octave_folder, x)) for x in ["reader.mat", "writer.mat"]]
        self._oc.exit()
        del self._oc
        return None


    def _top_eigenvectors(self, K, n_components):
        r"""
        Return eigenvectors associated with the largest algebraic eigenvalues.

        This mirrors MATLAB/Octave ``eigs(K, k, 'LA')`` for the symmetric
        kernel matrices used by OS-LF-IMVC.
        """
        K = (K + K.T) / 2
        n_samples = K.shape[0]

        if n_components >= n_samples:
            eigvals, eigvecs = np.linalg.eigh(K)
        else:
            kwargs = {}
            if self.random_state is not None:
                kwargs["v0"] = np.random.default_rng(self.random_state).uniform(size=n_samples)
            eigvals, eigvecs = eigsh(K, n_components, which="LA", **kwargs)

        order = np.argsort(eigvals)[::-1][:n_components]
        return np.real(eigvecs[:, order])


    @staticmethod
    def _normalize_rows(X):
        r"""
        Normalize rows to unit Euclidean norm, leaving zero rows unchanged.
        """
        norms = np.sqrt(np.sum(X ** 2, axis=1, keepdims=True))
        return np.divide(X, norms, out=np.zeros_like(X, dtype=float), where=norms > 0)


    @staticmethod
    def _orth(C):
        r"""
        Return a square orthonormal basis equivalent to MATLAB ``orth`` for
        full-rank square centroid matrices.
        """
        U, _, _ = svd(C, full_matrices=True)
        return U


    @staticmethod
    def _my_comb_fun(Y, beta):
        r"""
        Combine view-specific kernels using adaptive weights.
        """
        return np.tensordot(Y, beta.ravel(), axes=(2, 0))


    @staticmethod
    def _my_solving(B):
        r"""
        Solve the discrete assignment update by selecting the largest entry
        in each row.
        """
        Y = np.zeros_like(B)
        Y[np.arange(B.shape[0]), np.argmax(B, axis=1)] = 1
        return Y


    def _initialize_kh(self, KH, S):
        r"""
        Zero kernel rows and columns for missing samples in each view.
        """
        KH = np.array(KH, dtype=float, copy=True)
        for p in range(KH.shape[2]):
            missing = np.asarray(S[p]["indx"], dtype=int).ravel() - 1
            if missing.size == 0:
                continue
            KH[missing, :, p] = 0
            KH[:, missing, p] = 0
        return KH


    def _my_initialization(self, KH, S, n_clusters):
        r"""
        Initialize per-view base partition matrices and permutation matrices.
        """
        numker = KH.shape[2]
        num = KH.shape[0]
        HP = np.zeros(shape=(num, n_clusters, numker))
        WP = np.zeros(shape=(n_clusters, n_clusters, numker))

        all_indices = np.arange(num)
        for p in range(numker):
            missing = np.asarray(S[p]["indx"], dtype=int).ravel() - 1
            obs_index = np.setdiff1d(all_indices, missing, assume_unique=False)
            KAp = KH[np.ix_(obs_index, obs_index, [p])][:, :, 0]
            KAp = (KAp + KAp.T) / 2 + 1e-8 * np.eye(len(obs_index))
            HP[obs_index, :, p] = self._top_eigenvectors(KAp, n_clusters)
            WP[:, :, p] = np.eye(n_clusters)

        return HP, WP


    def _my_kernel_kmeans(self, K, n_clusters):
        r"""
        Kernel k-means spectral relaxation used by the original code.
        """
        return self._top_eigenvectors(K, n_clusters)


    def _my_initialization_c(self, KH, n_clusters):
        r"""
        Initialize the consensus centroid matrix.
        """
        numker = KH.shape[2]
        sigma0 = np.ones(shape=(numker, 1)) / numker
        avg_kernel = self._my_comb_fun(KH, sigma0)
        H = self._my_kernel_kmeans(avg_kernel, n_clusters)
        H = self._normalize_rows(H)

        model = KMeans(n_clusters=n_clusters, max_iter=200, n_init=30, random_state=self.random_state)
        model.fit(H)
        return self._orth(model.cluster_centers_)


    @staticmethod
    def _update_wp_oslfimvc(HP, Y, C):
        r"""
        Update view-specific permutation matrices.
        """
        k = HP.shape[1]
        numker = HP.shape[2]
        WP = np.zeros(shape=(k, k, numker))

        Hstar = Y @ C
        for p in range(numker):
            Tp = HP[:, :, p].T @ Hstar
            Up, _, Vp = svd(Tp, full_matrices=False)
            WP[:, :, p] = Up @ Vp

        return WP


    @staticmethod
    def _update_beta_oslfimvc(HP, WP, Y, C):
        r"""
        Update adaptive view weights.
        """
        numker = WP.shape[2]
        Hstar = Y @ C
        HHPWP = np.zeros(shape=(numker, 1))

        for p in range(numker):
            HHPWP[p] = np.real(np.trace(Hstar.T @ (HP[:, :, p] @ WP[:, :, p])))

        beta_norm = np.linalg.norm(HHPWP)
        if beta_norm == 0:
            return np.ones(shape=(numker, 1)) * np.sqrt(1 / numker)

        beta = HHPWP / beta_norm
        beta[beta < np.finfo(float).eps] = 0
        beta_norm = np.linalg.norm(beta)
        if beta_norm == 0:
            return np.ones(shape=(numker, 1)) * np.sqrt(1 / numker)
        return beta / beta_norm


    def _os_lf_imvc_alg(self, KH, S, n_clusters, lambda_reg):
        r"""
        Runs the OS-LF-IMVC clustering algorithm.

        This is a direct Python translation of the original
        ``OS_LF_IMVC_alg.m`` implementation. The preprocessing step from the
        original demo script (``kcenter``/``knorm`` before simulating
        incompleteness) is intentionally not included, matching the estimator
        API where callers may preprocess incomplete views themselves.
        """
        num = KH.shape[1]
        numker = KH.shape[2]
        max_iter = 100

        KH = self._initialize_kh(KH, S)
        HP, WP = self._my_initialization(KH, S, n_clusters)
        HP00 = HP.copy()
        beta = np.ones(shape=(numker, 1)) * np.sqrt(1 / numker)
        C = self._my_initialization_c(KH, n_clusters)
        KC = self._my_comb_fun(KH, beta)
        H0 = self._my_kernel_kmeans((KC + KC.T) / 2, n_clusters)

        RpHpwp = np.zeros(shape=(num, n_clusters))
        for p in range(numker):
            RpHpwp += beta[p] * (HP[:, :, p] @ WP[:, :, p])
        RpHpwp_lambda = RpHpwp + lambda_reg * H0

        obj = []
        all_indices = np.arange(num)
        flag = True
        iteration = 0
        while flag:
            iteration += 1

            Y = self._my_solving(RpHpwp_lambda @ C.T)

            CB = Y.T @ RpHpwp_lambda
            Uh, _, Vh = svd(CB, full_matrices=False)
            C = Uh @ Vh

            WP = self._update_wp_oslfimvc(HP, Y, C)

            for p in range(numker):
                missing = np.asarray(S[p]["indx"], dtype=int).ravel() - 1
                obs_index = np.setdiff1d(all_indices, missing, assume_unique=False)

                if missing.size > 0:
                    HB = Y[missing, :] @ C @ WP[:, :, p].T
                    Uh, _, Vh = svd(HB, full_matrices=False)
                    HP[missing, :, p] = Uh @ Vh
                HP[obs_index, :, p] = HP00[obs_index, :, p]

            beta = self._update_beta_oslfimvc(HP, WP, Y, C)

            RpHpwp = np.zeros(shape=(num, n_clusters))
            for p in range(numker):
                RpHpwp += beta[p] * (HP[:, :, p] @ WP[:, :, p])
            RpHpwp_lambda = RpHpwp + lambda_reg * H0

            obj.append(np.real(np.trace((Y @ C).T @ RpHpwp_lambda)))
            if ((iteration > 2) and (abs((obj[iteration - 1] - obj[iteration - 2]) / obj[iteration - 1]) < 1e-4)
                    or (iteration > max_iter)):
                flag = False

        return Y, C, WP, beta.ravel(), obj
