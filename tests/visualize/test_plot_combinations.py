import sys
import pytest
import numpy as np
import pandas as pd

from imml.visualize import plot_combinations


@pytest.fixture
def sample_data():
    X = np.random.default_rng(42).random((5, 10))
    X = pd.DataFrame(X)
    X1, X2, X3 = X.iloc[:, :3].copy(), X.iloc[:, 3:5].copy(), X.iloc[:, 5:].copy()
    X1.loc[[2,4], :] = np.nan
    X2.loc[1, :] = np.nan
    Xs_pandas, Xs_numpy = [X1, X2, X3], [X1.values, X2.values, X3.values]
    return Xs_pandas, Xs_numpy


@pytest.mark.skipif(sys.platform.startswith("win"), reason="Plot tests never ends on Windows")
def test_plot_combinations(sample_data):
    for Xs in sample_data:
        fig, axes = plot_combinations(Xs)
        assert fig is not None and axes is not None
        ax = axes[0, 1]
        assert ax.get_xlabel() == ""
        assert ax.get_ylabel() == "Intersection size"
        ax = axes[1, 0]
        assert ax.get_xlabel() == "Set size"
        assert ax.get_ylabel() == ""
        ax = axes[1, 1]
        assert ax.get_ylabel() == ""


@pytest.mark.skipif(sys.platform.startswith("win"), reason="Plot tests never ends on Windows")
def test_invalid_params(sample_data):
    with pytest.raises(ValueError, match="Invalid figsize."):
        plot_combinations(sample_data[0], figsize=2)
    with pytest.raises(ValueError, match="Invalid modalities."):
        plot_combinations(sample_data[0], modalities="1")
    with pytest.raises(ValueError, match="Invalid modalities."):
        plot_combinations(sample_data[0], modalities=[1, 2, 3])
    with pytest.raises(ValueError, match="Invalid modalities."):
        plot_combinations(sample_data[0], modalities=["1"])
    with pytest.raises(ValueError, match="Invalid max_combs."):
        plot_combinations(sample_data[0], max_combs="1")
