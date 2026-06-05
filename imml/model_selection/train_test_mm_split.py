# License: BSD-3-Clause

import numpy as np
from sklearn.model_selection import train_test_split

from ..model_selection._multi_modal_dataset import _MultiModalDataset
from ..utils import check_Xs_y


def train_test_mm_split(Xs, y=None, **kwargs):
    """
    Split multi-modal datasets and labels into train and test sets.

    Similar to sklearn's `train_test_split
    <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html>`_, but works with
    lists of arrays/data (Xs) and single arrays (y). Ensures that all X in a Xs get the same train/test split indices.

    Parameters
    ----------
    Xs : list of array-like
        - Xs length: n_mods
        - Xs[i] shape: (n_samples, n_features_i)

        A list of different modalities.
    y : array-like of shape (n_samples,), optional
        Target vector relative to Xs.

    **kwargs : dict
        Additional keyword arguments to pass to sklearn's train_test_split.
    Returns
    -------
    tuple
        Splitting results in the same order as inputs:
        - For each list input (Xs): (list_train, list_test)
        - For each array input (y): (array_train, array_test)

    Example
    --------
    >>> import numpy as np
    >>> from imml.model_selection import train_test_mm_split
    >>> Xs = [np.random.rand(100, 10), np.random.rand(100, 20)]
    >>> y = np.random.randint(0, 2, 100)
    >>> Xs_train, Xs_test, y_train, y_test = train_test_mm_split(Xs, y, train_size=0.7, random_state=42)
    """
    check_Xs_y(Xs=Xs, y=y)

    Xs = _MultiModalDataset(Xs)
    idxs = np.arange(len(Xs))
    tr, te = train_test_split(idxs, **kwargs)
    Xtr = Xs[tr].to_list()
    Xte = Xs[te].to_list()
    output = (Xtr, Xte)
    if y is not None:
        if hasattr(y, "iloc"):
            ytr = y.iloc[tr]
            yte = y.iloc[te]
        else:
            ytr = y[tr]
            yte = y[te]
        output = (*output, ytr, yte)
    return output
