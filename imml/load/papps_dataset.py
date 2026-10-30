# License: BSD-3-Clause

from .. import deepmodule_installed, deepmodule_error, Dataset
from ..impute import get_missing_mod_indicator
from ..utils import as_tensor_Xs, check_Xs_y, as_tensor_X

if deepmodule_installed:
    import torch


class PAPPSDataset(Dataset):
    r"""
    Dataset adapter for full-dataset PA-PPS training.

    Parameters
    ----------
    Xs : list of array-like, each of shape (n_samples, n_features)
        Dense numeric views. A view is considered missing for a sample when
        its entire row contains NaN values. Individual NaNs in an otherwise
        observed row are replaced with zero.
    y : array-like of shape (n_samples,)
        Integer class labels. NaN labels are treated as unlabeled.
    class_distribution : array-like of shape (n_classes,), default=None
        Optional global count per class. If omitted, counts are calculated
        from the observed values in ``y`` and scaled to the dataset size.

    Notes
    -----
    Use this dataset with a ``DataLoader`` whose batch size is the full dataset
    size and with ``shuffle=False``. PA-PPS maintains whole-dataset pseudo-label
    state and performs its own mini-batch optimization inside each epoch.
    """

    def __init__(self, Xs, y, class_distribution=None):
        if not deepmodule_installed:
            raise ImportError(deepmodule_error)
        Xs = check_Xs_y(Xs=Xs, y=y, ensure_all_finite="allow-nan")
        if y is None:
            raise ValueError("Invalid y. It cannot be None.")
        if len(y) != len(Xs[0]):
            raise ValueError("Invalid y. It must have the same number of samples as each view.")
        Xs = as_tensor_Xs(Xs)
        y = as_tensor_X(y).float()
        observed_y = torch.isfinite(y)

        if class_distribution is None:
            if not observed_y.any():
                raise ValueError("At least one training label must be observed.")
            observed_labels = y[observed_y]
            if (observed_labels != observed_labels.round()).any() or (observed_labels < 0).any():
                raise ValueError("Observed y values must be non-negative integer class labels.")
            num_classes = max(int(observed_labels.max()) + 1, 2)
            counts = torch.bincount(observed_labels.long(), minlength=num_classes).float()
            counts = counts * (len(Xs[0]) / counts.sum())
        else:
            counts = torch.as_tensor(class_distribution, dtype=torch.float32).detach().cpu().reshape(-1)
            if counts.numel() < 2 or not torch.isfinite(counts).all() or (counts < 0).any() \
                    or (counts != counts.round()).any():
                raise ValueError("class_distribution must contain at least two finite non-negative integer counts.")
            num_classes = len(counts)
            if counts.sum() != len(Xs[0]):
                raise ValueError("class_distribution must sum to the number of samples.")

        if not observed_y.any():
            raise ValueError("At least one training label must be observed.")
        if (y[observed_y] != y[observed_y].round()).any() or (y[observed_y] < 0).any():
            raise ValueError("Observed y values must be non-negative integer class labels.")
        if (y[observed_y] >= num_classes).any():
            raise ValueError("Observed y values must be smaller than the number of classes.")
        self.missing_mod_indicator = get_missing_mod_indicator(Xs)
        self.Xs = Xs
        self.y = y
        self.y_indicator = observed_y
        self.class_distribution = counts

    def __len__(self):
        return len(self.Xs[0])

    def __getitem__(self, idx):
        return (
            [X[idx] for X in self.Xs],
            self.y[idx],
            self.class_distribution,
        )