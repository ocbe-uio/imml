import pytest
import importlib
import sys
from unittest.mock import patch
import numpy as np

from imml.ampute import Amputer
from imml.cluster import OSLFIMVC as estimator
from imml import octavemodule_installed
from tests import ENGINES
from tests._sample_data import make_sample_data

sample_data = make_sample_data(n_samples=40, n_features=10, splits=(3, 5))


def test_octave_not_installed():
    if octavemodule_installed:
        estimator(engine="octave")
        with patch.dict(sys.modules, {"oct2py": None}):
            import imml as imml_mock
            import imml.cluster.oslfimvc as module_mock
            importlib.reload(imml_mock)
            importlib.reload(module_mock)
            with pytest.raises(ImportError, match="Module 'octave' needs to be installed."):
                estimator(engine="octave")
        importlib.reload(imml_mock)
        importlib.reload(module_mock)
    else:
        with pytest.raises(ImportError, match="Module 'octave' needs to be installed."):
            estimator(engine="octave")


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("Xs", sample_data)
def test_default_params(Xs, engine):
    model = estimator(random_state=42, engine=engine)
    n_samples = len(Xs[0])
    labels = model.fit_predict(Xs)
    assert labels is not None
    assert len(labels) == n_samples
    assert len(np.unique(labels)) == model.n_clusters
    assert min(labels) == 0
    assert max(labels) == (model.n_clusters - 1)
    assert not np.isnan(labels).any()
    assert not np.isnan(model.embedding_).any().any()
    assert model.embedding_.shape == (n_samples, model.n_clusters)
    assert model.WP_.shape == (model.n_clusters, model.n_clusters, len(Xs))
    assert len(model.beta_) == len(Xs)
    assert model.n_iter_ > 0


@pytest.mark.parametrize("Xs", sample_data)
def test_default_engine(Xs):
    model = estimator(random_state=42)
    labels = model.fit_predict(Xs)
    assert labels is not None
    assert model.engine == "python"
    assert model.embedding_.shape == (len(Xs[0]), model.n_clusters)


def test_invalid_params():
    with pytest.raises(ValueError, match="Invalid engine."):
        estimator(engine='invalid')
    with pytest.raises(ValueError, match="Invalid n_clusters."):
        estimator(n_clusters='invalid')
    with pytest.raises(ValueError, match="Invalid n_clusters."):
        estimator(n_clusters=0)


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("Xs", sample_data)
def test_fit_predict(Xs, engine):
    n_clusters = 3
    model = estimator(n_clusters=n_clusters, random_state=42, engine=engine)
    n_samples = len(Xs[0])
    labels = model.fit_predict(Xs)
    assert labels is not None
    assert len(labels) == n_samples
    assert len(np.unique(labels)) == n_clusters
    assert min(labels) == 0
    assert max(labels) == (n_clusters - 1)
    assert not np.isnan(labels).any()
    assert not np.isnan(model.embedding_).any().any()
    assert model.embedding_.shape == (n_samples, n_clusters)
    assert model.WP_.shape == (model.n_clusters, model.n_clusters, len(Xs))
    assert len(model.beta_) == len(Xs)
    assert model.n_iter_ > 0


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("Xs", sample_data)
def test_missing_values_handling(Xs, engine):
    n_clusters = 2
    model = estimator(n_clusters=n_clusters, random_state=42, engine=engine)
    Xs = Amputer(p= 0.3, random_state=42).fit_transform(Xs)
    n_samples = len(Xs[0])
    labels = model.fit_predict(Xs)
    assert labels is not None
    assert len(labels) == n_samples
    assert len(np.unique(labels)) == n_clusters
    assert min(labels) == 0
    assert max(labels) == (n_clusters - 1)
    assert not np.isnan(labels).any()
    assert not np.isnan(model.embedding_).any().any()
    assert model.embedding_.shape == (n_samples, n_clusters)
    assert model.WP_.shape == (model.n_clusters, model.n_clusters, len(Xs))
    assert len(model.beta_) == len(Xs)
    assert model.n_iter_ > 0


if __name__ == "__main__":
    pytest.main()
