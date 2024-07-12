import numpy as np
import pandas as pd
from sklearn.utils import check_array


def check_Xs(Xs, enforce_views=None, copy=False, force_all_finite="allow-nan",return_dimensions=False):
    r"""
    Checks Xs and ensures it to be a list of 2D matrices.

    Parameters
    ----------
    Xs : nd-array, list
        Input data.
    enforce_views : int, (default=not checked)
        If provided, ensures this number of views in Xs. Otherwise not checked.
    copy : boolean, (default=False)
        If True, the returned Xs is a copy of the input Xs,
        and operations on the output will not affect
        the input.
        If False, the returned Xs is a view of the input Xs,
        and operations on the output will change the input.
    force_all_finite : bool or 'allow-nan', default='allow-nan'
        Whether to raise an error on np.inf, np.nan, pd.NA in array. The possibilities are:
        - True: Force all values of array to be finite.
        - False: accepts np.inf, np.nan, pd.NA in array.
        - 'allow-nan': accepts only np.nan and pd.NA values in array. Values
          cannot be infinite.
    return_dimensions : boolean, (default=False)
        If True, the function also returns the dimensions of the multiview
        dataset. The dimensions are n_views, n_samples, n_features where
        n_samples and n_views are respectively the number of views and the
        number of samples, and n_features is a list of length n_views
        containing the number of features of each view.

    References
    ----------
    Adapted from mvlearn package.

    Returns
    -------
    Xs_converted : object
        The converted and validated Xs (list of data arrays).
    n_views : int
        The number of views in the dataset. Returned only if
        ``return_dimensions`` is ``True``.
    n_samples : int
        The number of samples in the dataset. Returned only if
        ``return_dimensions`` is ``True``.
    n_features : list
        List of length ``n_views`` containing the number of features in
        each view. Returned only if ``return_dimensions`` is ``True``.
    """
    if not isinstance(Xs, list):
        if not isinstance(Xs, np.ndarray):
            msg = f"If not list, input must be of type np.ndarray,\
                not {type(Xs)}"
            raise ValueError(msg)
        if Xs.ndim == 2:
            Xs = [Xs]
        else:
            Xs = list(Xs)

    n_views = len(Xs)
    if n_views == 0:
        msg = "Length of input list must be greater than 0"
        raise ValueError(msg)

    if n_views == 1:
        msg = "Must provide at least two data matrices"
        raise ValueError(msg)
    if enforce_views is not None and n_views != enforce_views:
        msg = "Wrong number of views. Expected {} but found {}".format(
            enforce_views, n_views
        )
        raise ValueError(msg)

    if not len(set([X.shape[0] for X in Xs])) == 1:
        msg = "All views must have the same number of samples"
        raise ValueError(msg)

    pandas_format = True if isinstance(Xs[0],pd.DataFrame) else False
    if pandas_format:
        Xs = [pd.DataFrame(check_array(X, allow_nd=False, copy=copy, force_all_finite=force_all_finite),
                           index=X.index, columns=X.columns) for X_idx, X in enumerate(Xs)]
    else:
        Xs = [check_array(X, allow_nd=False, copy=copy, force_all_finite=force_all_finite) for X in Xs]

    if return_dimensions:
        n_samples = Xs[0].shape[0]
        n_features = [X.shape[1] for X in Xs]
        return Xs, n_views, n_samples, n_features
    else:
        return Xs


def _convert_df_to_r_object(dataframe):
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr
    from rpy2.robjects import pandas2ri
    base = importr('base')
    with (ro.default_converter + pandas2ri.converter).context():
        r_from_pd_df = ro.conversion.get_conversion().py2rpy(dataframe)
    return base.lapply(r_from_pd_df, base.as_matrix)
