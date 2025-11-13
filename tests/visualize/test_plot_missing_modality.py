import sys
import pytest
import numpy as np
import pandas as pd

from imml.visualize import plot_missing_modality


@pytest.fixture
def sample_data():
    X = np.random.default_rng(42).random((5, 5))
    X = pd.DataFrame(X)
    X1, X2 = X.iloc[:, :3].copy(), X.iloc[:, 3:].copy()
    X1.loc[[2,4], :] = np.nan
    X2.loc[1, :] = np.nan
    Xs_pandas, Xs_numpy = [X1, X2], [X1.values, X2.values]
    return Xs_pandas, Xs_numpy


@pytest.mark.skipif(sys.platform.startswith("win"), reason="Plot tests never ends on Windows")
def test_plot_missing_modality(sample_data):
    for Xs in sample_data:
        fig, ax = plot_missing_modality(Xs)
        assert fig is not None and ax is not None
        assert ax.get_xlabel() == "Modality"
        assert ax.get_ylabel() == "Samples"
        xticks = ax.get_xticks()
        assert len(xticks) == len(Xs)


@pytest.mark.skipif(sys.platform.startswith("win"), reason="Plot tests never ends on Windows")
def test_invalid_params(sample_data):
    with pytest.raises(ValueError, match="Invalid figsize."):
        plot_missing_modality(sample_data[0], figsize=2)
    with pytest.raises(ValueError, match="Invalid sort."):
        plot_missing_modality(sample_data[0], sort=2)


if __name__ == "__main__":
    pytest.main()