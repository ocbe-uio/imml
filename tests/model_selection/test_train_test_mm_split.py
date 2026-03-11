import numpy as np
import pandas as pd
import pytest

from imml import deepmodule_installed
from imml.model_selection import train_test_mm_split

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

    return Xs_pandas, y_pandas, Xs_numpy, y_numpy


def test_train_test_mm_split_with_y(sample_data):
    for i in range(0, len(sample_data) - 1, 2):
        Xs = sample_data[i]
        y = sample_data[i+1]
        Xs_train, Xs_test, y_train, y_test = train_test_mm_split(
            Xs, y, train_size=0.6, random_state=0, shuffle=True
        )
        # Check modality count preserved
        assert isinstance(Xs_train, list) and isinstance(Xs_test, list)
        assert len(Xs_train) == len(Xs_test) == len(Xs)
        # Check lengths consistent across modalities and match y
        n_tr, n_te = len(y_train), len(y_test)
        assert n_tr + n_te == len(y)
        for Xm_tr, Xm_te in zip(Xs_train, Xs_test):
            assert len(Xm_tr) == n_tr
            assert len(Xm_te) == n_te


def test_train_test_mm_split_y_none(sample_data):
    for i in range(0, len(sample_data) - 1, 2):
        Xs = sample_data[i]
        Xs_train, Xs_test = train_test_mm_split(Xs, y=None, test_size=0.25, random_state=1, shuffle=True)
        assert isinstance(Xs_train, list) and isinstance(Xs_test, list)
        assert len(Xs_train) == len(Xs_test) == len(Xs)
        # Check split sizes add up
        n = len(Xs[0])
        n_tr = len(Xs_train[0])
        n_te = len(Xs_test[0])
        assert n_tr + n_te == n


if __name__ == "__main__":
    pytest.main()
