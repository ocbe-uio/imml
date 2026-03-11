# License: BSD-3-Clause

import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.utils import check_array, check_consistent_length, check_symmetric


def make_affinity(*data, metric='sqeuclidean', K=20, mu=0.5, normalize=True):
    r"""
    Constructs affinity (i.e., similarity) matrix from `data`

    Performs columnwise normalization on `data`, computes distance matrix based
    on provided `metric`, and then constructs affinity matrix. Uses a scaled
    exponential similarity kernel to determine the weight of each edge based on
    the distance matrix. Optional hyperparameters `K` and `mu` determine the
    extent of the scaling (see `Notes`).

    Parameters
    ----------
    *data : (N, M) array_like
        Raw data array, where `N` is samples and `M` is features. If multiple
        arrays are provided then affinity matrices will be generated for each.
    metric : str or list-of-str, optional
        Distance metric to compute. Must be one of available metrics in
        :py:func`scipy.spatial.distance.pdist`. If multiple arrays a provided
        an equal number of metrics may be supplied. Default: 'sqeuclidean'
    K : (0, N) int, optional
        Number of neighbors to consider when creating affinity matrix. See
        `Notes` of :py:func`snf.compute.affinity_matrix` for more details.
        Default: 20
    mu : (0, 1) float, optional
        Normalization factor to scale similarity kernel when constructing
        affinity matrix. See `Notes` of :py:func`snf.compute.affinity_matrix`
        for more details. Default: 0.5
    normalize : bool, optional
        Whether to normalize (i.e., zscore) `arr` before constructing the
        affinity matrix. Each feature (i.e., column) is normalized separately.
        Default: True

    Returns
    -------
    affinity : (N, N) numpy.ndarray or list of numpy.ndarray
        Affinity matrix (or matrices, if multiple inputs provided)

    Notes
    -----
    The scaled exponential similarity kernel, based on the probability density
    function of the normal distribution, takes the form:

    .. math::

       \mathbf{W}(i, j) = \frac{1}{\sqrt{2\pi\sigma^2}}
                          \ exp^{-\frac{\rho^2(x_{i},x_{j})}{2\sigma^2}}

    where :math:`\rho(x_{i},x_{j})` is the Euclidean distance (or other
    distance metric, as appropriate) between patients :math:`x_{i}` and
    :math:`x_{j}`. The value for :math:`\\sigma` is calculated as:

    .. math::

       \sigma = \mu\ \frac{\overline{\rho}(x_{i},N_{i}) +
                           \overline{\rho}(x_{j},N_{j}) +
                           \rho(x_{i},x_{j})}
                          {3}

    where :math:`\overline{\rho}(x_{i},N_{i})` represents the average value
    of distances between :math:`x_{i}` and its neighbors :math:`N_{1..K}`,
    and :math:`\mu\in(0, 1)\subset\mathbb{R}`.

    Examples
    --------
    >>> from snf import datasets
    >>> simdata = datasets.load_simdata()

    >>> from snf import compute
    >>> aff = compute.make_affinity(simdata.data[0], K=20, mu=0.5)
    >>> aff.shape
    (200, 200)
    """

    affinity = []
    for inp, met in _check_data_metric(data, metric):
        # normalize data, taking into account potentially missing data
        if normalize:
            mask = np.isnan(inp).all(axis=1)
            zarr = np.zeros_like(inp)
            zarr[mask] = np.nan
            zarr[~mask] = np.nan_to_num(stats.zscore(inp[~mask], ddof=1))
        else:
            zarr = inp

        # construct distance matrix using `metric` and make affinity matrix
        distance = cdist(zarr, zarr, metric=met)
        affinity += [affinity_matrix(distance, K=K, mu=mu)]

    # match input type (if only one array provided, return array not list)
    if len(data) == 1 and not isinstance(data[0], list):
        affinity = affinity[0]

    return affinity


def _check_data_metric(data, metric):
    """
    Confirms inputs to `make_affinity()` are appropriate

    Parameters
    ----------
    data : (F,) list of (M, N) array_like
        Input data arrays. All arrays should have same first dimension
    metric : str or (F,) list of str
        Input distance metrics. If provided as a list, should be the same
        length as `data`

    Yields
    ------
    data, metric : numpy.ndarray, str
        Tuples of an input data array and the corresponding distance metric
    """

    # make sure all inputs are the same length
    check_consistent_length(*list(_flatten(data)))

    # check provided metric -- if not a list, make it so
    if not isinstance(metric, (list, tuple)):
        metric = [metric] * len(data)

    # expand provided data arrays and metric so that it's 1:1
    for d, m in zip(data, metric):
        # if it's an iterable, recurse down
        if isinstance(d, (list, tuple)):
            yield from _check_data_metric(d, m)
        else:
            yield check_array(d, ensure_all_finite=False), m




def affinity_matrix(dist, *, K=20, mu=0.5):
    r"""
    Calculates affinity matrix given distance matrix `dist`

    Uses a scaled exponential similarity kernel to determine the weight of each
    edge based on `dist`. Optional hyperparameters `K` and `mu` determine the
    extent of the scaling (see `Notes`).

    You'd probably be best to use :py:func`snf.compute.make_affinity` instead
    of this, as that command also handles normalizing the inputs and creating
    the distance matrix.

    Parameters
    ----------
    dist : (N, N) array_like
        Distance matrix
    K : (0, N) int, optional
        Number of neighbors to consider. Default: 20
    mu : (0, 1) float, optional
        Normalization factor to scale similarity kernel. Default: 0.5

    Returns
    -------
    W : (N, N) np.ndarray
        Affinity matrix

    Notes
    -----
    The scaled exponential similarity kernel, based on the probability density
    function of the normal distribution, takes the form:

    .. math::

       \mathbf{W}(i, j) = \frac{1}{\sqrt{2\pi\sigma^2}}
                          \ exp^{-\frac{\rho^2(x_{i},x_{j})}{2\sigma^2}}

    where :math:`\rho(x_{i},x_{j})` is the Euclidean distance (or other
    distance metric, as appropriate) between patients :math:`x_{i}` and
    :math:`x_{j}`. The value for :math:`\\sigma` is calculated as:

    .. math::

       \sigma = \mu\ \frac{\overline{\rho}(x_{i},N_{i}) +
                           \overline{\rho}(x_{j},N_{j}) +
                           \rho(x_{i},x_{j})}
                          {3}

    where :math:`\overline{\rho}(x_{i},N_{i})` represents the average value
    of distances between :math:`x_{i}` and its neighbors :math:`N_{1..K}`,
    and :math:`\mu\in(0, 1)\subset\mathbb{R}`.

    Examples
    --------
    >>> from snf import datasets
    >>> simdata = datasets.load_simdata()

    We need to construct a distance matrix before we can create a similarity
    matrix using :py:func:`snf.compute.affinity_matrix`:

    >>> from scipy.spatial.distance import cdist
    >>> dist = cdist(simdata.data[0], simdata.data[0])

    >>> from snf import compute
    >>> aff = compute.affinity_matrix(dist)
    >>> aff.shape
    (200, 200)
    """

    # check inputs
    dist = check_array(dist, ensure_all_finite=False)
    dist = check_symmetric(dist, raise_warning=False)

    # get mask for potential NaN values and set diagonals zero
    mask = np.isnan(dist)
    dist[np.diag_indices_from(dist)] = 0

    # sort array and get average distance to K nearest neighbors
    T = np.sort(dist, axis=1)
    TT = np.vstack(T[:, 1:K + 1].mean(axis=1) + np.spacing(1))

    # compute sigma (see equation in Notes)
    sigma = (TT + TT.T + dist) / 3
    msigma = np.ma.array(sigma, mask=mask)  # mask for NaN
    sigma = sigma * np.ma.greater(msigma, np.spacing(1)).data + np.spacing(1)

    # get probability density function with scale = mu*sigma and symmetrize
    scale = (mu * np.nan_to_num(sigma)) + mask
    W = stats.norm.pdf(np.nan_to_num(dist), loc=0, scale=scale)
    W[mask] = np.nan
    W = check_symmetric(W, raise_warning=False)

    return W


def _find_dominate_set(W, K=20):
    """
    Retains `K` strongest edges for each sample in `W`

    Parameters
    ----------
    W : (N, N) array_like
        Input data
    K : (0, N) int, optional
        Number of neighbors to retain. Default: 20

    Returns
    -------
    Wk : (N, N) np.ndarray
        Thresholded version of `W`
    """

    # let's not modify W in place
    Wk = W.copy()

    # determine percentile cutoff that will keep only `K` edges for each sample
    # remove everything below this cutoff
    cutoff = 100 - (100 * (K / len(W)))
    Wk[Wk < np.percentile(Wk, cutoff, axis=1, keepdims=True)] = 0

    # normalize by strength of remaining edges
    Wk = Wk / np.nansum(Wk, axis=1, keepdims=True)

    return Wk


def _flatten(messy):
    """
    Flattens a messy list of mixed iterables / not-iterables

    Parameters
    ----------
    messy : list of ???
        Combined list of iterables / non-iterables

    Yields
    ------
    data : ???
        Entries from `messy`

    Notes
    -----
    Thanks to https://stackoverflow.com/a/2158532 :chef-kissing-fingers-emoji:
    """

    for m in messy:
        if isinstance(m, (list, tuple)):
            yield from _flatten(m)
        else:
            yield m


def get_n_clusters(arr, n_clusters=range(2, 6)):
    """
    Finds optimal number of clusters in `arr` via eigengap method

    Parameters
    ----------
    arr : (N, N) array_like
        Input array (e.g., the output of :py:func`snf.compute.snf`)
    n_clusters : array_like
        Numbers of clusters to choose between

    Returns
    -------
    opt_cluster : int
        Optimal number of clusters
    second_opt_cluster : int
        Second best number of clusters
    """

    # confirm inputs are appropriate
    n_clusters = check_array(n_clusters, ensure_2d=False)
    n_clusters = n_clusters[n_clusters > 1]

    # don't overwrite provided array!
    graph = arr.copy()

    graph = (graph + graph.T) / 2
    graph[np.diag_indices_from(graph)] = 0
    degree = graph.sum(axis=1)
    degree[np.isclose(degree, 0)] += np.spacing(1)
    di = np.diag(1 / np.sqrt(degree))
    laplacian = di @ (np.diag(degree) - graph) @ di

    # perform eigendecomposition and find eigengap
    eigs = np.sort(np.linalg.eig(laplacian)[0])
    eigengap = np.abs(np.diff(eigs))
    eigengap = eigengap * (1 - eigs[:-1]) / (1 - eigs[1:])
    n = eigengap[n_clusters - 1].argsort()[::-1]

    return n_clusters[n[:2]]
