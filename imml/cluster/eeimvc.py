# License: BSD-3-Clause

import os
from contextlib import contextmanager
from os.path import dirname
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import KMeans
from sklearn.gaussian_process import kernels
from scipy.sparse.linalg import eigs
from numpy.linalg import svd

from ..impute import get_observed_mod_indicator
from ..utils import check_Xs

matlabmodule_installed = False
oct2py_module_error = "Module 'matlab' needs to be installed. See https://imml.readthedocs.io/stable/main/installation.html#optional-dependencies"
try:
    import oct2py
    matlabmodule_installed = True
except ImportError:
    pass


@contextmanager
def fixed_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


class EEIMVC(BaseEstimator, ClassifierMixin):
    r"""
    Efficient and Effective Incomplete Multi-view Clustering (EE-IMVC). [#eeimvcpaper]_ [#eeimvccode]_

    EE-IMVC impute missing views with a consensus clustering matrix that is regularized with prior knowledge.

    Parameters
    ----------
    n_clusters : int, default=8
        The number of clusters to generate.
    kernel : callable, default=None
        Specifies the kernel type to be used in the algorithm. By default, it applies dot product kernel.
    lambda_reg : float, default=1.
        Regularization parameter. The algorithm demonstrated stable performance across a wide range of
        this hyperparameter.
    qnorm : float, default=2.
        Regularization parameter. The algorithm demonstrated stable performance across a wide range of
        this hyperparameter.
    random_state : int, default=None
        Determines the randomness. Use an int to make the randomness deterministic.
    engine : str, default=python
        Engine to use for computing the model. Current options are 'matlab' or 'python'.
    verbose : bool, default=False
        Verbosity mode.
    clean_space : bool, default=True
        If engine is 'matlab' and clean_space is True, the session will be closed after fitting the model.

    Attributes
    ----------
    labels_ : array-like of shape (n_samples,)
        Labels of each point in training data.
    embedding_ : array-like of shape (n_samples, n_clusters)
        Consensus clustering matrix to be used as input for the KMeans clustering step.
    WP_ : array-like of shape (n_clusters, n_clusters, n_mods)
        p-th permutation matrix.
    HP_ : array-like of shape (n_samples, n_clusters, n_mods)
        missing part of the p-th base clustering matrix.
    beta_ : array-like of shape (n_mods,)
        Adaptive weights of clustering matrices.
    loss_ : array-like of shape (n_iter\_,)
        Values of the loss function.
    n_iter_ : int
        Number of iterations.

    References
    ----------
    .. [#eeimvcpaper] X. Liu et al., "Efficient and Effective Regularized Incomplete Multi-View Clustering," in
                        IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 43, no. 8, pp. 2634-2646,
                        1 Aug. 2021, doi: 10.1109/TPAMI.2020.2974828.
    .. [#eeimvccode] https://github.com/xinwangliu/TPAMI_EEIMVC

    Example
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from imml.cluster import EEIMVC
    >>> Xs = [pd.DataFrame(np.random.default_rng(42).random((20, 10))) for i in range(3)]
    >>> estimator = EEIMVC(n_clusters = 2)
    >>> labels = estimator.fit_predict(Xs)
    """

    def __init__(self, n_clusters: int = 8, kernel: callable = None,
                 lambda_reg: float = 1., qnorm: float = 2., random_state: int = None,
                 engine: str ="python", verbose = False, clean_space: bool = True):
        if not isinstance(n_clusters, int):
            raise ValueError(f"Invalid n_clusters. It must be an int. A {type(n_clusters)} was passed.")
        if n_clusters < 2:
            raise ValueError(f"Invalid n_clusters. It must be an greater than 1. {n_clusters} was passed.")
        engines_options = ["matlab", "python"]
        if engine not in engines_options:
            raise ValueError(f"Invalid engine. Expected one of {engines_options}. {engine} was passed.")
        if (engine == "matlab") and (not matlabmodule_installed):
            raise ImportError(oct2py_module_error)

        if kernel is None:
            kernel = kernels.Sum(kernels.DotProduct(), kernels.WhiteKernel())

        self.n_clusters = n_clusters
        self.qnorm = qnorm
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
            if isinstance(Xs[0], pd.DataFrame):
                transformed_Xs = [X.values for X in Xs]
            elif isinstance(Xs[0], np.ndarray):
                transformed_Xs = Xs
            observed_mod_indicator = get_observed_mod_indicator(transformed_Xs)
            if isinstance(observed_mod_indicator, np.ndarray):
                observed_mod_indicator = pd.DataFrame(observed_mod_indicator)
            s = [modality[modality == 0].index.values for _,modality in observed_mod_indicator.items()]
            transformed_Xs = [self.kernel(X) for X in transformed_Xs]
            transformed_Xs = np.array(transformed_Xs).swapaxes(0, -1)
            s = tuple([{"indx": i +1} for i in s])

            if self.random_state is not None:
                self._oc.rand('seed', self.random_state)
            H_normalized,WP,HP,beta,obj = self._oc.incompleteLateFusionMKCOrthHp_lambda(transformed_Xs, s,
                                                                                        self.n_clusters, self.qnorm,
                                                                                        self.lambda_reg, nout=5)
            beta = beta[:,0]
            obj = obj[0]

            if self.clean_space:
                self._clean_space()

        elif self.engine=="python":
            observed_mod_indicator = get_observed_mod_indicator(Xs)
            if isinstance(observed_mod_indicator, pd.DataFrame):
                observed_mod_indicator = observed_mod_indicator.reset_index(drop=True)
            elif isinstance(observed_mod_indicator[0], np.ndarray):
                observed_mod_indicator = pd.DataFrame(observed_mod_indicator)
            s = [modality[modality == 0].index.values for _, modality in observed_mod_indicator.items()]
            transformed_Xs = [self.kernel(X) for X in Xs]
            transformed_Xs = np.array(transformed_Xs).swapaxes(0, -1)
            transformed_Xs = np.nan_to_num(transformed_Xs, nan=0)
            s = tuple([{"indx": i + 1} for i in s])

            H_normalized, WP, HP, beta, obj = self._incomplete_late_fusion_MKCOrthHp_lamba(transformed_Xs, s,
                                                                                      self.n_clusters,
                                                                                      self.qnorm, self.lambda_reg)

        model = KMeans(n_clusters= self.n_clusters, n_init="auto", random_state= self.random_state)
        self.labels_ = model.fit_predict(X=H_normalized)
        self.embedding_, self.WP_, self.HP_, self.beta_, self.loss_ = H_normalized, WP, HP, beta, obj
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



    def _my_initialization_Hp(self, KH, S, n_clusters):
        r"""
        Initialize HP and WP variable.

        Parameters
        ----------
        KH: 3-D array of shape(n_samples, n_samples, kernels)
        S: tuple of shape (n_mods)
            - S[i]['indx']: array of missing values column
        n_clusters: int
            The number of clusters.

        Returns
        -------
        HP: 3-d array of shape (n_samples, n_clusters, n_mods)
        WP: 3-d array of shape (n_clusters, n_clusters, n_mods)
        """
        numker = KH.shape[2]
        num = KH.shape[0]
        HP = np.zeros(shape=(num, n_clusters, numker))
        WP = np.zeros(shape=(n_clusters, n_clusters, numker))

        for p in range(numker):
            KH_tmp = KH[:, :, p]
            HP_tmp = HP[:, :, p]

            obs_index = np.setdiff1d(ar1=[i for i in range(num)], ar2=[i-1 for i in S[p]['indx'].T])
            KAp = KH_tmp[np.ix_(obs_index, obs_index)]
            KAp = (KAp + KAp.T) / 2 + 1e-8 * np.eye(len(obs_index))
            if self.random_state is not None:
                v0 = np.random.default_rng(self.random_state).uniform(size=min(KAp.shape))
                with fixed_seed(self.random_state):
                    _, Hp = eigs(KAp, n_clusters, which='LR', v0=v0)
            else:
                _, Hp = eigs(KAp, n_clusters, which='LR')

            HP_tmp[np.ix_(obs_index), :] = Hp
            HP[:, :, p] = HP_tmp
            WP[:, :, p] = np.eye(n_clusters)

        return HP, WP

    def _algorithm2(self, KH, S):
        r"""
        Process KH with the missing index.

        Parameters
        ----------
        KH: 3-D array of shape(n_samples, n_samples, kernels)
        S: tuple of shape (n_mods)
            - S[i]['indx']: array of missing values column

        Returns
        -------
        KH2: 3-D array of shape (n_samples, n_samples, n_mods)
        """
        num = KH.shape[0]
        numker = KH.shape[2]
        KH2 = np.zeros(shape=(num, num, numker))

        for p in range(numker):
            KH_tmp = KH[:, :, p]
            KH2_tmp = KH2[:, :, p]

            obs_index = np.setdiff1d(ar1=[i for i in range(num)], ar2=[i-1 for i in S[p]['indx'].T])
            KAp = KH_tmp[np.ix_(obs_index, obs_index)]
            KH2_tmp[np.ix_(obs_index, obs_index)] = (KAp + KAp.T)/2
            KH2[:, :, p] = KH2_tmp

        return KH2


    def _my_comb_fun(self, Y, beta):
        r"""
        Process data with beta values

        Parameters
        ----------
        Y: 3-D array of shape(n_samples, n_samples, kernels)
        beta: list of float (len=n_mods)

        Returns
        -------
        cF: 2-D array of shape (n_samples, n_samples)
        """
        m = Y.shape[2]
        n = Y.shape[0]
        cF = np.zeros(shape=(n, n))

        for p in range(m):
            cF += Y[:, :, p] * beta[p]

        return cF


    def _my_kernal_kmeans(self, K, n_clusters):
        r"""
        Determines eigenvectors.

        Parameters
        ----------
        K: 2-D array of shape (n_samples, n_samples)
        n_clusters: int
            The number of clusters.

        Returns
        -------
        H_normalized: 2-D array of shape (n_samples, n_clusters)
        """
        K = (K + K.T) / 2
        if self.random_state is not None:
            v0 = np.random.default_rng(self.random_state).uniform(size=min(K.shape))
            with fixed_seed(self.random_state):
                _, H = eigs(K, n_clusters, which='LR', v0=v0)
        else:
            _, H = eigs(K, n_clusters, which='LR')
        obj = np.trace(np.matmul(H.T, np.matmul(K, H))) - np.trace(K)
        H_normalized = H
        return H_normalized


    def _update_WP_absent_clustering_V1(self, HP, Hstar):
        r"""
        Update the WP variable.

        Parameters
        ----------
        HP: 3-D array of shape (n_samples, n_clusters, n_mods)
        Hstar: 2-D array of shape (n_samples, n_clusters)

        Returns
        -------
        WP: 3-d array of shape (n_clusters, n_clusters, n_mods)
        """
        k = HP.shape[1]
        numker = HP.shape[2]
        WP = np.zeros(shape=(k, k, numker))
        for p in range(numker):
            Tp = np.matmul(HP[:, :, p].T, Hstar)
            Up, Sp, Vp = np.linalg.svd(Tp, full_matrices=False)
            V = Vp.T.conj()
            WP[:, :, p] = np.matmul(Up, V.T)

        return WP


    def _update_HP_absent_clustering_OrthHp(self, WP, Hstar, S, HP00):
        r"""
        Update the HP variable.

        Parameters
        ----------
        WP: 3-D array of shape (n_clusters, n_clusters, n_views)
        Hstar: 2-D array of shape (n_samples, n_clusters)
        S: tuple of shape (n_views)
            - S[i]['indx']: array of missing values column
        HP00: 3-D array of shape (n_samples, n_clusters, n_views)

        Returns
        -------
        HP: 3-D array of shape (n_samples, n_clusters, n_views)
        """
        num = Hstar.shape[0]
        k = Hstar.shape[1]
        numker = WP.shape[2]
        HP = np.zeros(shape=(num, k, numker))

        for p in range(numker):
            mis_indx = [i-1 for i in S[p]['indx'].T]
            obs_indx = np.setdiff1d(ar1=[i for i in range(num)], ar2=mis_indx)

            if len(mis_indx) > 0:
                Vp = np.matmul(Hstar[np.ix_(mis_indx), :], WP[:, :, p].T)
                Up, Sp, Vp = svd(Vp, full_matrices=False)
                V = Vp.T.conj()
                HP[mis_indx, :, p] = np.matmul(Up, V.T)

            HP_tmp = HP[:, :, p]
            HP00_tmp = HP00[:, :, p]

            HP_tmp[np.ix_(obs_indx), :] = HP00_tmp[np.ix_(obs_indx), :]
            HP[:, :, p] = HP_tmp

        return HP


    def _update_beta_absent_clustering(self, HP, WP, Hstar, qnorm):
        r"""
        Update the beta variable

        Parameters
        ----------
        HP: 3-D array of shape (n_samples, n_clusters, n_views)
        WP: 3-D array of shape (n_clusters, n_clusters, n_views)
        Hstar: 2-D array of shape (n_samples, n_clusters)
        qnorm: float, default=2.0

        Returns
        -------
        beta: list of float (len=n_views)
        """
        numker = WP.shape[2]
        HHPWP = np.zeros(shape=(numker, 1))

        for p in range(numker):
            HHPWP[p] = np.trace(np.matmul(Hstar.T, np.matmul(HP[:, :, p], WP[:, :, p])))

        beta = HHPWP**(1/qnorm-1) / np.sum(HHPWP**(qnorm/(qnorm-1)))**(1/qnorm)
        return beta


    def _incomplete_late_fusion_MKCOrthHp_lamba(self, KH, S, n_clusters, qnorm, lambda_reg):
        r"""
        Runs the EEIMVC clustering algorithm.

        Parameters
        ----------
        KH: 3-D array of shape(n_samples, n_samples, n_views)
        S: tuple of shape (n_views)
            - S[i]['indx']: array of missing values column
        n_clusters: int
            The number of clusters.
        qnorm: float, default=2.0
        lambda_: float, default=1.0

        Returns
        -------
        H_normalized: list of array-likes objects of shape (n_samples, n_clusters)
        WP: 3-D array of shape (n_clusters, n_clusters, n_views)
        HP: 3-D array of shape (n_samples, n_clusters, n_views)
        beta: list of float (len=n_views)
        obj: list of float
        """
        num = KH.shape[1]
        numker = KH.shape[2]
        maxIter = 100
        HP, WP = self._my_initialization_Hp(KH, S, n_clusters)
        HP00 = HP
        beta = np.ones(shape=(numker, 1)) * (1/numker)**(1/qnorm)
        KA = self._algorithm2(KH, S)
        KC = self._my_comb_fun(KA, beta)
        H0 = self._my_kernal_kmeans(KC, n_clusters)

        flag = 1
        iter = 0
        RpHpwp = np.zeros(shape=(num, n_clusters))
        for p in range(numker):
            RpHpwp += beta[p] * (HP[:, :, p] @ WP[:, :, p])

        RpHpwp_lambda = RpHpwp + (lambda_reg * H0)
        obj = []
        obj.append(0)
        while flag:
            iter += 1
            Uh, Sh, Vh = svd(RpHpwp_lambda, full_matrices=False)
            V = Vh.T.conj()

            Hstar = Uh @ V.T
            WP = self._update_WP_absent_clustering_V1(HP, Hstar)
            HP = self._update_HP_absent_clustering_OrthHp(WP, Hstar, S, HP00)
            beta = self._update_beta_absent_clustering(HP, WP, Hstar, qnorm)

            RpHpwp = np.zeros(shape=(num, n_clusters))
            for p in range(numker):
                RpHpwp += beta[p] * np.matmul(HP[:, :, p], WP[:, :, p])

            RpHpwp_lambda = RpHpwp + (lambda_reg * H0)
            obj.append(np.trace(np.matmul(Hstar.T, RpHpwp_lambda)))

            if (iter > 2) and (np.abs((obj[iter] - obj[iter-1]) / obj[iter])) < 1e-4 or (iter > maxIter):
                flag = 0


        H_normalized = np.real(Hstar / np.tile(A=np.sqrt(np.sum(Hstar**2, 1)), reps=(n_clusters, 1)).T)
        return H_normalized, WP, HP, beta, obj
