import importlib
import sys
from unittest.mock import patch
import pytest
import numpy as np

from imml.ampute import Amputer
from imml.cluster import NEMO
from tests._sample_data import make_sample_data

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
ENGINES = ["python"]
if rmodule_installed and snftool_installed:
    ENGINES += ["r"]

sample_data = make_sample_data(n_samples=50, n_features=20, splits=(10, 15))

@pytest.mark.skipif(not snftool_installed, reason="SNFtool is not installed.")
def test_rmodule_installed():
    if rmodule_installed:
        estimator(engine="r")
        with patch.dict(sys.modules, {"rpy2": None}):
            import imml as imml_mock
            import imml.cluster.nemo as module_mock
            importlib.reload(imml_mock)
            importlib.reload(module_mock)
            with pytest.raises(ImportError, match="Module 'r' needs to be installed to use r engine."):
                estimator(engine="r")
        importlib.reload(imml_mock)
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

@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("Xs", sample_data)
def test_param_randomstate(Xs, engine):
    random_state = 42
    labels = estimator(engine=engine, random_state=random_state).fit_predict(Xs)
    assert all(labels == estimator(engine=engine, random_state=random_state).fit_predict(Xs))


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("Xs", sample_data)
def test_default_params(Xs, engine):
    model = estimator(random_state=42, engine=engine)
    n_samples = len(Xs[0])
    labels = model.fit_predict(Xs)
    assert labels is not None
    assert len(labels) == n_samples
    assert len(np.unique(labels)) == model.n_clusters_
    assert min(labels) == 0
    assert max(labels) == (model.n_clusters - 1)
    assert not np.isnan(labels).any()
    if engine == "python":
        assert not np.isnan(model.embedding_).any().any()
        assert model.embedding_.shape == (n_samples, model.n_clusters_)
    assert model.affinity_matrix_.shape == (n_samples, n_samples)
    assert len(model.num_neighbors_) == len(Xs)
    assert model.num_neighbors_[0] > 0


@pytest.mark.parametrize("Xs", sample_data)
@pytest.mark.parametrize("engine", ENGINES)
def test_custom_parameters(Xs, engine):
    model = estimator(n_clusters=list(range(2, 4)), num_neighbors=3, random_state=42, engine=engine)
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

def test_invalid_params():
    with pytest.raises(ValueError, match="Invalid engine."):
        estimator(engine='invalid')

@pytest.mark.parametrize("Xs", sample_data)
@pytest.mark.parametrize("engine", ENGINES)
def test_fit_predict(Xs, engine):
    n_clusters = 3
    model = estimator(n_clusters=n_clusters, engine=engine, random_state=42)
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

@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("Xs", sample_data)
def test_missing_values_handling(Xs, engine):
    n_clusters = 3
    model = estimator(n_clusters=n_clusters, engine=engine, random_state=42)
    Xs = Amputer(p=0.3, random_state=42).fit_transform(Xs)
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
