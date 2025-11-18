import numpy as np
import pandas as pd
import pytest

from imml.model_selection import multi_train_test_split
from imml import deepmodule_installed, Tensor

if deepmodule_installed:
    import torch


@pytest.fixture
def sample_data():
    X = np.random.default_rng(42).random((5, 5))
    X = pd.DataFrame(X)
    X1, X2 = X.iloc[:, :3].copy(), X.iloc[:, 3:].copy()
    Xs_pandas, Xs_numpy = [X1, X2], [X1.values, X2.values]
    output = (Xs_pandas, Xs_numpy,)
    if deepmodule_installed:
        Xs_torch = [torch.from_numpy(X) for X in Xs_numpy]
        output = output + (Xs_torch,)
    return output


def test_list_input(sample_data):
    for Xs in sample_data:
        n = len(Xs[0])
        y = np.array([0] * n)
        y[len(y)//2:] = 1
        if isinstance(Xs[0], pd.DataFrame):
            y = pd.Series(y)
        elif isinstance(Xs[0], Tensor):
            y = torch.from_numpy(y)
        Xs_train, Xs_test, y_train, y_test = multi_train_test_split(Xs, y, train_size=0.6, shuffle=True)
        X1_train, X2_train = Xs_train
        X1_test, X2_test = Xs_test
        assert len(X1_train) == int(0.6 * n)
        assert len(X1_test) == n - int(0.6 * n)
        assert len(set(len(X) for X in Xs)) == 1
        if isinstance(Xs[0], pd.DataFrame):
            assert Xs[0].index.equals(Xs[1].index)
            assert Xs[0].index.equals(y.index)


def test_random_state(sample_data):
    for Xs in sample_data:
        n = len(Xs[0])
        y = np.array([0] * n)
        y[len(y)//2:] = 1
        if isinstance(Xs[0], pd.DataFrame):
            y = pd.Series(y)
        elif isinstance(Xs[0], Tensor):
            y = torch.from_numpy(y)
        Xs_train, Xs_test, y_train, y_test = multi_train_test_split(Xs, y, test_size=0.4, stratify=y,
                                                                    random_state=42)
        X1_train, X2_train = Xs_train
        X1_test, X2_test = Xs_test
        assert len(X1_train) == int(0.6 * n)
        assert len(X1_test) == n - int(0.6 * n)
        assert len(set(len(X) for X in Xs)) == 1
        if isinstance(Xs[0], pd.DataFrame):
            assert Xs[0].index.equals(Xs[1].index)
            assert Xs[0].index.equals(y.index)


def test_three_input(sample_data):
    for Xs in sample_data:
        n = len(Xs[0])
        y = np.array([0] * n)
        y[len(y)//2:] = 1
        if isinstance(Xs[0], pd.DataFrame):
            y = pd.Series(y)
        elif isinstance(Xs[0], Tensor):
            y = torch.from_numpy(y)
        Xs_train, Xs_test, Xs_train1, Xs_test1, y_train, y_test = multi_train_test_split(Xs, Xs, y,
                                                                                         train_size=0.6,
                                                                                         shuffle=True)
        X1_train, X2_train = Xs_train
        X1_test, X2_test = Xs_test
        assert len(X1_train) == int(0.6 * n)
        assert len(X1_test) == n - int(0.6 * n)
        assert len(set(len(X) for X in Xs)) == 1
        if isinstance(Xs[0], pd.DataFrame):
            assert Xs[0].index.equals(Xs[1].index)
            assert Xs[0].index.equals(y.index)

            assert X1_train.equals(Xs_train1[0])
            assert X2_train.equals(Xs_train1[1])
            assert X1_test.equals(Xs_test1[0])
            assert X2_test.equals(Xs_test1[1])
