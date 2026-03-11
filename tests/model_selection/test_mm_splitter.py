import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit

from imml import deepmodule_installed
from imml.model_selection import MMSplitter

if deepmodule_installed:
    import torch


@pytest.fixture
def sample_data():
    X = np.random.default_rng(42).random((30, 10))
    X = pd.DataFrame(X)
    X1, X2, X3 = X.iloc[:, :3].copy(), X.iloc[:, 3:5].copy(), X.iloc[:, 5:10].copy()
    Xs_pandas, Xs_numpy = [X1, X2, X3], [X1.values, X2.values, X3.values]
    y_numpy = np.random.default_rng(42).choice(2, size=len(Xs_pandas[0]))
    y_pandas = pd.Series(y_numpy)
    output = (Xs_pandas, y_pandas, Xs_numpy, y_numpy)
    if deepmodule_installed:
        Xs_torch = [torch.from_numpy(X) for X in Xs_numpy]
        y_torch = torch.from_numpy(y_numpy)
        output = (*output, Xs_torch, y_torch)
    return output


def test_mmsplitter_with_y(sample_data):
    for i in range(0, len(sample_data) - 1, 2):
        Xs = sample_data[i]
        y = sample_data[i+1]
        splitter = StratifiedShuffleSplit(n_splits=5, random_state=42)
        gen = MMSplitter(splitter=splitter, return_type="split")
        assert gen.get_n_splits() == 5
        Xs_tr, Xs_te, ytr, yte = next(gen.split(Xs=Xs, y=y))
        assert isinstance(Xs_tr, list) and isinstance(Xs_te, list)
        assert len(Xs_tr) == len(Xs_te) == len(Xs)
        # Check lengths per modality are consistent and match y
        n_tr = len(ytr)
        n_te = len(yte)
        for X_tr, X_te in zip(Xs_tr, Xs_te):
            assert len(X_tr) == n_tr
            assert len(X_te) == n_te


def test_mmsplitter_indices_and_y_none(sample_data):
    for i in range(0, len(sample_data) - 1, 2):
        Xs = sample_data[i]
        y = sample_data[i+1]
        # return indices
        splitter = StratifiedShuffleSplit(n_splits=5, random_state=42)
        gen = MMSplitter(splitter=splitter, return_type="indices")
        tr, te = next(gen.split(Xs=Xs, y=y))
        assert len(tr) + len(te) == len(Xs[0])
        # y=None path
        splitter = ShuffleSplit(n_splits=5, random_state=42)
        gen = MMSplitter(splitter=splitter, return_type="split")
        Xs_tr, Xs_te = next(gen.split(Xs=Xs, y=None))
        assert isinstance(Xs_tr, list) and isinstance(Xs_te, list)
        assert len(Xs_tr) == len(Xs_te) == len(Xs)


def test_mmsplitter_bad_return_type_raises(sample_data):
    Xs, y = sample_data[:2]
    with pytest.raises(ValueError):
        splitter = StratifiedShuffleSplit(n_splits=5, random_state=42)
        gen = MMSplitter(splitter=splitter, return_type="bad")


if __name__ == "__main__":
    pytest.main()
