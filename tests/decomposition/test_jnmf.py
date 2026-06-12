import importlib
import sys
from unittest.mock import patch
import pytest
import numpy as np
import pandas as pd

from imml.ampute import Amputer
from imml.decomposition import JNMF

try:
    from rpy2.robjects.packages import importr, PackageNotInstalledError
    rpy2_installed = True
    try:
        nnTensor = importr("nnTensor")
        nnTensor_installed = True
    except PackageNotInstalledError:
        nnTensor_installed = False
except ImportError:
    rpy2_installed = False
    nnTensor_installed = False

estimator = JNMF


@pytest.fixture
def sample_data():
    X = np.random.default_rng(42).random((20, 10))
    X = pd.DataFrame(X)
    X1, X2, X3 = X.iloc[:, :3], X.iloc[:, 3:5], X.iloc[:, 5:]
    Xs_pandas, Xs_numpy = [X1, X2, X3], [X1.values, X2.values, X3.values]
    return Xs_pandas, Xs_numpy


def test_r_module_installed():
    if nnTensor_installed:
        estimator(engine="r")
        with patch.dict(sys.modules, {"rpy2": None}):
            import imml as imml_mock
            import imml.decomposition.jnmf as module_mock
            importlib.reload(imml_mock)
            importlib.reload(module_mock)
            with pytest.raises(ImportError, match="Module 'r' needs to be installed."):
                estimator(engine="r")
        importlib.reload(imml_mock)
        importlib.reload(module_mock)
    else:
        with pytest.raises(ImportError, match="Module 'r' needs to be installed."):
            estimator(engine="r")


def test_default_params(sample_data):
    transformer = estimator(random_state=42)
    for Xs in sample_data:
        transformer.fit(Xs)
        assert hasattr(transformer, 'H_')
        assert hasattr(transformer, 'reconstruction_err_')
        assert hasattr(transformer, 'observed_reconstruction_err_')
        assert hasattr(transformer, 'missing_reconstruction_err_')
        assert hasattr(transformer, 'relchange_')


def test_param_randomstate(sample_data):
    random_state = 42
    for engine in ["python", "r"]:
        if (engine == "r") and (not nnTensor_installed):
            continue
        transformed_X = estimator(engine=engine, random_state=random_state).fit_transform(sample_data[0])
        np.testing.assert_array_equal(transformed_X, estimator(engine=engine, random_state=random_state).fit_transform(sample_data[0]))


def test_fit(sample_data):
    n_components = 5
    for engine in ["python", "r"]:
        if (engine == "r") and (not nnTensor_installed):
            continue
        for Xs in sample_data:
            transformer = estimator(n_components=n_components, engine=engine, random_state=42)
            transformer.fit(Xs)
            assert len(transformer.H_) == len(Xs)
            assert transformer.H_[0].shape == (Xs[0].shape[1], n_components)


def test_transform(sample_data):
    n_components = 5
    for engine in ["python", "r"]:
        if (engine == "r") and (not nnTensor_installed):
            continue
        for Xs in sample_data:
            transformer = estimator(n_components=n_components, engine=engine, random_state=42)
            n_samples = len(Xs[0])
            transformer.fit(Xs)
            transformed_X = transformer.transform(Xs)
            assert transformed_X.shape == (n_samples, n_components)
            assert len(transformer.H_) == len(Xs)
            assert transformer.H_[0].shape == (Xs[0].shape[1], n_components)


def test_missing_values(sample_data):
    n_components = 5
    for engine in ["python", "r"]:
        if (engine == "r") and (not nnTensor_installed):
            continue
        for Xs in sample_data:
            Xs = Amputer(p=0.3, random_state=42).fit_transform(Xs)
            transformer = estimator(n_components=n_components, engine=engine, random_state=42)
            n_samples = len(Xs[0])
            transformed_X = transformer.fit_transform(Xs)
            assert not np.isnan(transformed_X).any().any()
            assert transformed_X.shape == (n_samples, n_components)
            assert len(transformer.H_) == len(Xs)
            assert transformer.H_[0].shape == (Xs[0].shape[1], n_components)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
