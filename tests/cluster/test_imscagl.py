import pytest
import importlib
import sys
from unittest.mock import patch
import numpy as np
import pandas as pd

from imml.ampute import Amputer
from imml.cluster import IMSCAGL as estimator
from imml import octavemodule_installed


@pytest.fixture
def sample_data():
    X = np.random.default_rng(42).random((8, 5))
    X = pd.DataFrame(X)
    X1, X2, X3 = X.iloc[:, :2], X.iloc[:, 2:4], X.iloc[:, 4:]
    Xs_pandas, Xs_numpy = [X1, X2, X3], [X1.values, X2.values, X3.values]
    return Xs_pandas, Xs_numpy


def test_octave_not_installed():
    if octavemodule_installed:
        estimator(engine="octave")
        with patch.dict(sys.modules, {"oct2py": None}):
            import imml as imml_mock
            import imml.cluster.imscagl as module_mock
            importlib.reload(imml_mock)
            importlib.reload(module_mock)
            with pytest.raises(ImportError, match="Module 'octave' needs to be installed."):
                estimator(engine="octave")
        importlib.reload(imml_mock)
        importlib.reload(module_mock)
    else:
        with pytest.raises(ImportError, match="Module 'octave' needs to be installed."):
            estimator(engine="octave")


def test_default_params(sample_data):
    for Xs in sample_data:
        model = estimator(random_state=42)
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


def test_param_randomstate(sample_data):
    random_state = 42
    for engine in ["python", "octave"]:
        if (engine == "octave") and not octavemodule_installed:
            continue
        labels = estimator(engine=engine, random_state=random_state).fit_predict(sample_data[0])
        assert all(labels == estimator(engine=engine, random_state=random_state).fit_predict(sample_data[0]))


def test_invalid_params(sample_data):
    with pytest.raises(ValueError, match="Invalid engine."):
        estimator(engine='invalid')
    with pytest.raises(ValueError, match="Invalid n_clusters."):
        estimator(n_clusters='invalid')
    with pytest.raises(ValueError, match="Invalid n_clusters."):
        estimator(n_clusters=0)
    with pytest.raises(ValueError, match="Invalid lambda1."):
        estimator(lambda1=-1)
    with pytest.raises(ValueError, match="Invalid lambda2."):
        estimator(lambda2=-1)
    with pytest.raises(ValueError, match="Invalid lambda3."):
        estimator(lambda3=-1)
    with pytest.raises(ValueError, match="Invalid k."):
        estimator(k=-1)
    with pytest.raises(ValueError, match="Invalid neighbor_mode."):
        estimator(neighbor_mode='invalid')
    with pytest.raises(ValueError, match="Invalid weight_mode."):
        estimator(weight_mode='invalid')
    with pytest.raises(ValueError, match="Invalid max_iter."):
        estimator(max_iter=-1)
    with pytest.raises(ValueError, match="Invalid miu."):
        estimator(miu=-1)
    with pytest.raises(ValueError, match="Invalid rho."):
        estimator(rho=-1)
    with pytest.raises(ValueError, match="Invalid clean_space."):
        estimator(clean_space='invalid')


def test_fit_predict(sample_data):
    n_clusters = 3
    for engine in ["python", "octave"]:
        if (engine == "octave") and not octavemodule_installed:
            continue
        for Xs in sample_data:
            model = estimator(n_clusters=n_clusters, engine=engine, random_state=42)
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


def test_missing_values_handling(sample_data):
    n_clusters = 2
    for engine in ["python", "octave"]:
        if (engine == "octave") and not octavemodule_installed:
            continue
        for Xs in sample_data:
            model = estimator(n_clusters=n_clusters, engine=engine, random_state=42)
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


if __name__ == "__main__":
    pytest.main()
