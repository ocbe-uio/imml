# License: BSD-3-Clause

import os
from os.path import dirname

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import KMeans
from scipy.linalg import svd
from scipy.stats import zscore

from ..impute import simple_mod_imputer
from ..preprocessing import select_complete_samples
from ..utils import check_Xs
from ..explore import get_missing_samples_by_mod

matlabmodule_installed = False
oct2py_module_error = "Module 'matlab' needs to be installed. See https://imml.readthedocs.io/stable/main/installation.html#optional-dependencies"
try:
    import oct2py
    matlabmodule_installed = True
except ImportError:
    pass


class SIMCADC(BaseEstimator, ClassifierMixin):
    r"""
    Scalable Incomplete Multiview Clustering with Adaptive Data Completion (SIMC-ADC). [#simcadcpaper]_ [#simcadccode]_

    The SIMC-ADC algorithm captures the complementary information from different views by building a view-specific
    anchor graph. The anchor graph construction and a structure alignment are jointly optimized to enhance
    clustering quality.

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
    V_ : array-like of shape (n_clusters, n_clusters)
        Commont latent feature matrix.
    A_ : array-like of shape (n_clusters, n_clusters)
        Learned anchors.
    Z_ : array-like of shape (n_clusters, n_samples)
        modality-specific anchor graph.
    loss_ : array-like of shape (n_iter\_,)
        Values of the loss function.
    n_iter_ : int
        Number of iterations.

    References
    ----------
    .. [#simcadcpaper] He, W.-J., Zhang, Z., & Wei, Y. (2023). Scalable incomplete multi-view clustering with adaptive
                       data completion. Information Sciences, 649, 119562. doi:10.1016/j.ins.2023.119562.
    .. [#simcadccode] https://github.com/DarrenZZhang/INS23-SIMC_ADC

    Example
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from imml.cluster import SIMCADC
    >>> Xs = [pd.DataFrame(np.random.default_rng(42).random((20, 10))) for i in range(3)]
    >>> estimator = SIMCADC(n_clusters = 2)
    >>> labels = estimator.fit_predict(Xs)
    """

    def __init__(self, n_clusters: int = 8, lambda_parameter: float = 1, n_anchors: int = None,
                 beta: float = 1, gamma: float = 1, eps: float = 1e-25, random_state:int = None,
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

        self.n_clusters = n_clusters
        self.lambda_parameter = lambda_parameter
        self.beta = beta
        self.gamma = gamma
        self.eps = eps
        self.n_anchors = n_clusters if n_anchors is None else n_anchors
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
            if not isinstance(Xs[0], pd.DataFrame):
                Xs = [pd.DataFrame(X) for X in Xs]
            mean_mod_profile = [X.mean(axis=0).to_frame(X_id) for X_id, X in enumerate(select_complete_samples(Xs))]
            incomplete_samples = get_missing_samples_by_mod(Xs=Xs, return_as_list=True)
            mean_mod_profile = [pd.DataFrame(np.tile(means, len(incom))).values for means, incom in
                                  zip(mean_mod_profile, incomplete_samples)]

            transformed_Xs = simple_mod_imputer(Xs, value="zeros")
            transformed_Xs, mean_mod_profile = tuple(transformed_Xs), tuple(mean_mod_profile)

            w = [pd.DataFrame(np.eye(len(X)), index=X.index, columns=X.index) for X in Xs]
            w = [eye.loc[samples,:].values for eye, samples in zip(w, incomplete_samples)]
            w = tuple(w)

            n_incomplete_samples_mod = list(len(incomplete_sample) for incomplete_sample in incomplete_samples)

            # if self.random_state is not None:
            #     self._oc.rand('seed', self.random_state)
            u,v,a,w,z,iter,obj = self._oc.SIMC(transformed_Xs, len(Xs[0]), self.lambda_parameter,
                                                self.n_clusters, self.n_anchors, w, n_incomplete_samples_mod,
                                                mean_mod_profile, self.beta, self.gamma, nout=7)
            obj = obj[0]

            if self.clean_space:
                self._clean_space()

        elif self.engine=="python":
            if not isinstance(Xs[0], pd.DataFrame):
                Xs = [pd.DataFrame(X) for X in Xs]
            mean_mod_profile = [X.mean(axis=0).to_frame(X_id) for X_id, X in enumerate(select_complete_samples(Xs))]
            incomplete_samples = get_missing_samples_by_mod(Xs=Xs, return_as_list=True)
            mean_mod_profile = [pd.DataFrame(np.tile(means, len(incom))).values for means, incom in
                                 zip(mean_mod_profile, incomplete_samples)]

            transformed_Xs = simple_mod_imputer(Xs, value="zeros")
            # transformed_Xs, mean_view_profile = tuple(transformed_Xs), tuple(mean_view_profile)

            w = [pd.DataFrame(np.eye(len(X)), index=X.index, columns=X.index) for X in Xs]
            w = [eye.loc[samples, :].values for eye, samples in zip(w, incomplete_samples)]
            # w = tuple(w)

            n_incomplete_samples_mod = list(len(incomplete_sample) for incomplete_sample in incomplete_samples)

            if self.random_state is not None:
                np.random.seed(self.random_state)
            u, v, a, w, z, iter, obj = self._SIMC(transformed_Xs, len(Xs[0]), self.lambda_parameter,
                                                     self.n_clusters, self.n_anchors, w, n_incomplete_samples_mod,
                                                     mean_mod_profile, self.beta, self.gamma)

        model = KMeans(n_clusters= self.n_clusters, n_init= "auto", random_state= self.random_state)
        self.labels_ = model.fit_predict(X=u)
        self.embedding_ = u
        self.V_ = v
        self.A_ = a
        self.Z_, self.loss_, self.iter_ = z, obj, iter
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



    def _eproj_simplex_new(self, v, k=1):
        r"""
        Adjust the v variable if needed.

        Parameters
        ----------
        v: list of length (n_clusters)
        k: int, default=1

        Returns
        -------
        v0: list of length (n_clusters)
        """
        ft = 1
        n = len(v)

        v0 = v - np.mean(v) + k / n
        vmin = np.min(v0)

        if vmin < 0:
            lambda_m = 0
            f = 1

            while abs(f) > 1e-10:
                v1 = v0 - lambda_m
                posidx = v1 > 0
                npos = np.sum(posidx)

                g = -npos
                f = np.sum(v1[posidx]) - k

                lambda_m -= f / g
                ft += 1

                if ft > 100:
                    return np.maximum(v1, 0)

            return np.maximum(v1, 0)

        else:
            # If no adjustment is needed, return the shifted vector
            return v0


    def _SIMC(self, Y, num_sample, lambda_parameter, n_clusters, n_anchors, N, Ne, E, beta, gamma):
        r"""
        Runs the SIMCADC algorithm.

        Parameters
        ----------
        Y: list of array-likes objects of length (n_mods)
            - Y[i] shape: (n_samples, n_features_i)
        num_sample: int
            Number of samples
        lambda_parameter : float, default=1
            Balance the influence between anchor graph generation and alignment term.
        n_clusters : int, default=8
            The number of clusters to generate.
        n_anchors : int, default=None
            Number of anchors. If None, use n_clusters.
        N: list of arrays-likes of length (n_mods)
            - N[i] shape: (missing_view_columns, n_features_i)
        Ne: list of arrays-likes of length (n_mods)
            Number of missing modality columns
        E: list of arrays-likes of length (n_mods)
            - E[i] shape: (n_features_i, missing_view_columns)
        beta: float, default=1.0
        gamma: float, default=1.0

        Returns
        -------
        UU: list of array-likes objects of shape (n_samples, n_clusters)
        V: list of array-likes objects of shape (n_clusters, n_clusters)
        A: list of array-likes objects of shape (n_clusters, n_anchors)
        W: list of array-likes objects of length (n_view)
            - W[i] shape: (n_features_i, n_clusters)
        Z_final: list of array-likes objects of shape (n_anchors, n_samples)
        iter: int
            Number of iterations.
        obj: List of floats of length (iter)
        """
        # Initialize parameters
        maxIter = 50
        num_view = len(Y)

        W = [None] * num_view
        A = np.zeros(shape=(n_clusters, n_anchors))
        Z_final = np.zeros(shape=(n_anchors, num_sample))
        NNT = [None] * num_view
        B = [None] * num_view
        Z = [None] * num_view
        R = [None] * num_view

        # Preprocessing Y and initializing
        for i in range(num_view):
            Y[i] = np.nan_to_num(zscore(Y[i], ddof=1, axis=0).T, nan=0)
            di = Y[i].shape[0]
            W[i] = np.zeros(shape=(di, n_clusters))
            NNT[i] = np.matmul(N[i], N[i].T)
            B[i] = np.zeros(shape=(di, num_sample))
            Z[i] = np.zeros(shape=(n_anchors, num_sample))
            R[i] = np.eye(n_anchors)

        Z_final[:, :n_anchors] = np.eye(n_anchors)
        alpha = (np.ones(shape=(1, num_view)) / num_view)[0]

        flag = 1
        iter = 0
        obj = []

        while flag:
            iter += 1

            X = [Y[iv] + np.matmul(E[iv], N[iv]) for iv in range(num_view)]
            X = [np.nan_to_num(i, nan=0) for i in X]

            # Update W_i
            for iv in range(num_view):
                AZ = np.matmul(A, Z[iv])
                C = np.matmul(X[iv], AZ.T)
                U, _, Vt = svd(C, full_matrices=False)
                Vt = Vt.T.conj()
                W[iv] = np.matmul(U, Vt.T)

            # Update A
            part1 = sum(alpha[ia] ** 2 * np.matmul(W[ia].T, np.matmul(X[ia], Z[ia].T)) for ia in range(num_view))
            Unew, _, Vnew = svd(part1, full_matrices=False)
            Vnew = Vnew.T.conj()
            A = np.matmul(Unew, Vnew.T)

            # Update Z_i
            for iv in range(num_view):
                C1 = alpha[iv] ** 2 * np.matmul(X[iv].T, np.matmul(W[iv], A)) + gamma * np.matmul(Z_final.T, R[iv])
                C2 = alpha[iv] ** 2 + gamma
                C1 = C1.T
                for ii in range(num_sample):
                    ut = C1[:, ii] / C2
                    Z[iv][:, ii] = self._eproj_simplex_new(ut)

            # Update Z_final
            C3 = sum(gamma * np.matmul(Z[iv].T, R[iv].T) for iv in range(num_view))
            C3 = C3.T
            C4 = num_view * gamma + 1
            for ii in range(num_sample):
                ut = C3[:, ii] / C4
                Z_final[:, ii] = self._eproj_simplex_new(ut)

            # Update E
            for iv in range(num_view):
                B[iv] = Y[iv] - np.matmul(W[iv], np.matmul(A, Z_final))
                E[iv] = np.matmul(-B[iv], np.matmul(N[iv].T, np.linalg.inv(NNT[iv] + beta * np.ones((Ne[iv], Ne[iv])))))

            # Update R
            for iv in range(num_view):
                Z_iv_final = np.matmul(Z[iv], Z_final.T)
                U_R, _, V_R = svd(Z_iv_final, full_matrices=False)
                V_R = V_R.T.conj()
                R[iv] = np.matmul(U_R, V_R.T)

            # Update alpha
            for iv in range(num_view):
                alpha[iv] = np.sqrt(1 / (np.linalg.norm(B[iv] + np.matmul(E[iv], N[iv]), 'fro') + np.finfo('float').eps))

            # Objective calculation
            term1 = sum(alpha[iv] ** 2 * np.linalg.norm(X[iv] - np.matmul(W[iv], np.matmul(A, Z_final)), 'fro') ** 2 for iv in range(num_view))
            term2 = lambda_parameter * np.linalg.norm(Z_final, 'fro') ** 2
            obj.append(term1 + term2)

            if iter > 9 and (abs((obj[iter - 2] - obj[iter - 1]) / obj[iter - 2]) < 1e-3 or iter > maxIter or obj[
                iter - 1] < 1e-10):
                UU, _, V = svd(Z_final.T, full_matrices=False)
                V = V.T.conj()
                flag = 0

        UU = UU / np.sqrt(np.sum(UU ** 2, axis=1, keepdims=True))
        return UU, V, A, W, Z_final, iter, obj
