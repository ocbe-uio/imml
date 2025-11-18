import sys
import numpy as np
import pandas as pd
import pytest

from imml.visualize import plot_summary
from imml import deepmodule_installed

if deepmodule_installed:
    import torch


@pytest.fixture
def sample_data():
    X = np.random.default_rng(42).random((30, 5))
    X = pd.DataFrame(X)
    X1, X2 = X.iloc[:, :2], X.iloc[:, 2:]
    Xs_pandas, Xs_numpy = [X1, X2], [X1.values, X2.values]
    output = (Xs_pandas, Xs_numpy,)
    if deepmodule_installed:
        Xs_torch = [torch.from_numpy(X) for X in Xs_numpy]
        output = output + (Xs_torch,)
    return output


@pytest.mark.skipif(sys.platform.startswith("win"), reason="Plot tests never ends on Windows")
def test_plot_combinations(sample_data):
    for Xs in sample_data:
        ax = plot_summary(Xs)
        assert ax is not None


@pytest.mark.skipif(sys.platform.startswith("win"), reason="Plot tests never ends on Windows")
def test_invalid_params(sample_data):
    with pytest.raises(ValueError, match="Invalid figsize."):
        plot_summary(sample_data[0], figsize=2)
    with pytest.raises(ValueError, match="Invalid summary."):
        plot_summary(summary=1)
