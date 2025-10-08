import pytest
rpy2 = pytest.importorskip("rpy2")
import numpy as np
import pandas as pd
from rpy2.robjects.packages import importr, PackageNotInstalledError
rbase = importr("base")

from imml.impute import JNMFImputer

try:
    nnTensor = importr("nnTensor")
    nnTensor_installed = True
except PackageNotInstalledError:
    nnTensor_installed = False
estimator = JNMFImputer


@pytest.fixture
def sample_data():
    X = np.random.default_rng(42).random((50, 270))
    X = pd.DataFrame(X)
    X1, X2 = X.iloc[:, :100].copy(), X.iloc[:, 100:].copy()
    X1.loc[[2,4], :] = np.nan
    X2.loc[1, :] = np.nan
    Xs_pandas, Xs_numpy = [X1, X2], [X1.values, X2.values]
    return Xs_pandas, Xs_numpy


@pytest.mark.skipif(not nnTensor_installed, reason="nnTensor is not installed.")
def test_default_params(sample_data):
    transformer = JNMFImputer(random_state=42)
    for Xs in sample_data:
        transformed_Xs = transformer.fit_transform(Xs)
        assert len(transformed_Xs) == len(Xs)
        assert transformed_Xs[0].shape == Xs[0].shape


@pytest.mark.skipif(not nnTensor_installed, reason="nnTensor is not installed.")
def test_transform(sample_data):
    transformer = JNMFImputer(random_state=42)
    for Xs in sample_data:
        transformer.fit(Xs)
        transformed_Xs = transformer.transform(Xs)
        assert len(transformed_Xs) == len(Xs)
        assert transformed_Xs[0].shape == Xs[0].shape
        assert not any([np.isnan(transformed_X).any().any() for transformed_X in transformed_Xs])


if __name__ == "__main__":
    pytest.main()