import importlib
import sys
from unittest.mock import patch
import pytest
import numpy as np
import pandas as pd

from imml.ampute import Amputer
from imml.cluster import NEMO

try:
    from rpy2.robjects.packages import importr, PackageNotInstalledError
    rmodule_installed = True
except ImportError:
    rmodule_installed = False

if rmodule_installed:
    rbase = importr("base")
    try:
        snftool = importr("SNFtool")
        snftool_installed = True
    except PackageNotInstalledError:
        snftool_installed = False
else:
    snftool_installed = False

estimator = NEMO


@pytest.fixture
def sample_data():
    X = np.random.default_rng(42).random((50, 20))
    X = pd.DataFrame(X)
    X1, X2, X3 = X.iloc[:, :10], X.iloc[:, 10:15], X.iloc[:, 15:]
    Xs_pandas, Xs_numpy = [X1, X2, X3], [X1.values, X2.values, X3.values]
    return Xs_pandas, Xs_numpy

@pytest.mark.skipif(not snftool_installed, reason="SNFtool is not installed.")
def test_rmodule_installed():
    if rmodule_installed:
        estimator(engine="r")
        with patch.dict(sys.modules, {"rpy2": None}):
            import imml.cluster.nemo as module_mock
            importlib.reload(module_mock)
            with pytest.raises(ImportError, match="Module 'r' needs to be installed to use r engine."):
                estimator(engine="r")
        importlib.reload(module_mock)
    else:
        with pytest.raises(ImportError, match="Module 'r' needs to be installed to use r engine."):
            estimator(engine="r")

@pytest.mark.skipif(not rmodule_installed, reason="Module 'r' needs to be installed to use r engine.")
def test_r_dependencies_not_installed():
    if rmodule_installed:
        if snftool_installed:
            estimator(engine="r")
        else:
            with pytest.raises(ImportError, match="SNFtool needs to be installed in R to use r engine."):
                estimator(engine="r")

def test_param_randomstate(sample_data):
    random_state = 42
    for engine in ["r", "python"]:
        if (engine == "r") and (not rmodule_installed or not snftool_installed):
            continue
        else:
            labels = estimator(engine=engine, random_state=random_state).fit_predict(sample_data[0])
            assert all(labels == estimator(engine=engine, random_state=random_state).fit_predict(sample_data[0]))

def test_default_params(sample_data):
    model = estimator(random_state=42)
    for Xs in sample_data:
        n_samples = len(Xs[0])
        labels = model.fit_predict(Xs)
        assert labels is not None
        assert len(labels) == n_samples
        assert len(np.unique(labels)) == model.n_clusters_
        assert min(labels) == 0
        assert max(labels) == (model.n_clusters - 1)
        assert not np.isnan(labels).any()
        assert not np.isnan(model.embedding_).any().any()
        assert model.embedding_.shape == (n_samples, model.n_clusters_)
        assert model.affinity_matrix_.shape == (n_samples, n_samples)
        assert len(model.num_neighbors_) == len(Xs)
        assert model.num_neighbors_[0] > 0

def test_custom_parameters(sample_data):
    for engine in ["r", "python"]:
        if (engine == "r") and (not rmodule_installed or not snftool_installed):
            continue
        else:
            model = estimator(n_clusters=list(range(2, 4)), num_neighbors=3, random_state=42, engine=engine)
        for Xs in sample_data:
            n_samples = len(Xs[0])
            labels = model.fit_predict(Xs)
            assert labels is not None
            assert len(labels) == n_samples
            assert len(np.unique(labels)) == model.n_clusters_
            assert min(labels) == 0
            assert max(labels) == (model.n_clusters_ - 1)
            assert not np.isnan(labels).any()
            if engine == "python":
                assert model.embedding_.shape == (n_samples, model.n_clusters_)
            assert model.affinity_matrix_.shape == (n_samples, n_samples)
            assert len(model.num_neighbors_) == len(Xs)
            assert model.num_neighbors_[0] > 0
    estimator(n_clusters=list(range(2, 4)), num_neighbors=[3, 3, 3], random_state=42).fit_predict(Xs)
    estimator(n_clusters=None)

def test_invalid_params(sample_data):
    with pytest.raises(ValueError, match="Invalid engine."):
        estimator(engine='invalid')

def test_fit_predict(sample_data):
    n_clusters = 3
    for engine in ["r", "python"]:
        if (engine == "r") and (not rmodule_installed or not snftool_installed):
            continue
        else:
            model = estimator(n_clusters=n_clusters, engine=engine, random_state=42)
        for Xs in sample_data:
            n_samples = len(Xs[0])
            labels = model.fit_predict(Xs)
            assert labels is not None
            assert len(np.unique(labels)) == model.n_clusters_
            assert min(labels) == 0
            assert max(labels) == (n_clusters - 1)
            assert not np.isnan(labels).any()
            if engine == "python":
                assert not np.isnan(model.embedding_).any().any()
                assert model.embedding_.shape == (n_samples, model.n_clusters_)
            assert model.affinity_matrix_.shape == (n_samples, n_samples)
            assert len(model.num_neighbors_) == len(Xs)
            assert model.num_neighbors_[0] > 0

def test_missing_values_handling(sample_data):
    n_clusters = 3
    for engine in ["r", "python"]:
        if (engine == "r") and (not rmodule_installed or not snftool_installed):
            continue
        for Xs in sample_data:
            model = estimator(n_clusters=n_clusters, engine=engine, random_state=42)
            Xs = Amputer(p= 0.3, random_state=42).fit_transform(Xs)
            n_samples = len(Xs[0])
            labels = model.fit_predict(Xs)
            assert labels is not None
            assert len(np.unique(labels)) == model.n_clusters_
            assert min(labels) == 0
            assert max(labels) == (n_clusters - 1)
            assert not np.isnan(labels).any()
            if engine == "python":
                assert not np.isnan(model.embedding_).any().any()
                assert model.embedding_.shape == (n_samples, model.n_clusters_)
            assert model.affinity_matrix_.shape == (n_samples, n_samples)
            assert len(model.num_neighbors_) == len(Xs)
            assert model.num_neighbors_[0] > 0

if __name__ == "__main__":
    pytest.main()