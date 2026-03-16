# License: BSD-3-Clause

import numpy as np

from ..model_selection import _MultiModalDataset
from ..utils import check_Xs_y


class MMSplitter():
    """
    Generic bridge between scikit-learn splitters and multi-modal inputs.

    This helper receives any scikit-learn splitter (such as `StratifiedKFold
    <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html>`_) and
    yields splits.
    A single set of train/test indices is computed by the splitter and applied to every modality, guaranteeing aligned
    partitions across all modalities.

    Parameters
    ----------
    splitter : object
        Any object implementing scikit-learn's splitter interface, for example
        ``KFold``, ``StratifiedKFold``, ``GroupKFold`` or ``ShuffleSplit``.
    return_type : str, default="split"
        Controls what each yielded item contains: "split" returns the actual partition sets, while "indices" return
        the indices of the partition sets.

    Example
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import StratifiedKFold
    >>> from imml.model_selection import MMSplitter
    >>> Xs = [np.random.rand(100, 10), np.random.rand(100, 20)]
    >>> y = np.random.randint(0, 2, 100)
    >>> splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    >>> for Xs_train, Xs_test, y_train, y_test in MMSplitter(splitter=splitter).split(Xs, y):
    ...     pass
    """

    def __init__(self, splitter, return_type: str = "all"):
        if return_type not in ["split", "indices"]:
            raise ValueError(f"return_type must be one of ['split', 'indices'], got {return_type}")

        self.splitter = splitter
        self.return_type = return_type


    def get_n_splits(self, X=None, y=None, groups=None):
        """
        Returns the number of splitting iterations as set with the n_splits param.

        Parameters
        ----------
        Xs : Always ignored, exists for API compatibility.
        y : Always ignored, exists for API compatibility.
        groups : Always ignored, exists for API compatibility.

        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations.
        """

        return self.splitter.get_n_splits()


    def split(self, Xs, y=None, groups=None):
        """
        Generate indices to split data into training and test set.

        Parameters
        ----------
        Xs : list of array-like
            - Xs length: n_mods
            - Xs[i] shape: (n_samples, n_features_i)

            A list of different modalities.
        y : array-like of shape (n_samples,), optional
            Target vector relative to Xs.
        groups : array-like, optional
            Group labels passed to ``splitter.split(...)``.

        Returns
        -------
        tuple
            One tuple per split according to ``return_type``.
        """
        check_Xs_y(Xs=Xs, y=y)

        Xs = _MultiModalDataset(Xs)
        idxs = np.arange(len(Xs))
        for tr, te in self.splitter.split(idxs, y=y, groups=groups):
            if self.return_type == "split":
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
            elif self.return_type == "indices":
                output = (tr, te)
            yield output

