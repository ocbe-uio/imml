import importlib
import sys
from unittest.mock import patch
import pytest
import numpy as np
import pandas as pd

from imml.ampute import Amputer
from imml.cluster import MLMF as estimator
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
            import imml.cluster.mlmf as module_mock
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
    for engine in ["python", "octave"]:
        if (engine == "octave") and not octavemodule_installed:
            continue
        for Xs in sample_data:
            model = estimator(engine=engine, random_state=42)
            n_samples = len(Xs[0])
            labels = model.fit_predict(Xs)
            assert labels is not None
            assert len(labels) == n_samples
            assert len(np.unique(labels)) == model.n_clusters
            assert min(labels) == 0
            assert max(labels) == (model.n_clusters - 1)
            assert not np.isnan(labels).any()
            assert not np.isnan(model.embedding_).any().any()
            assert model.embedding_.shape[0] == n_samples
            assert model.n_iter_ > 0


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
    with pytest.raises(ValueError, match="Invalid factorization."):
        estimator(factorization='invalid')
    with pytest.raises(ValueError, match="Invalid n_clusters."):
        estimator(n_clusters='invalid')
    with pytest.raises(ValueError, match="Invalid n_clusters."):
        estimator(n_clusters=0)
    with pytest.raises(ValueError, match="Invalid lambda1."):
        estimator(lambda1=-0.5)
    with pytest.raises(ValueError, match="Invalid lambda2."):
        estimator(lambda2=-0.5)
    with pytest.raises(ValueError, match="Invalid layers."):
        estimator(layers=[0, 2])
    with pytest.raises(ValueError, match="Invalid layers."):
        estimator(layers='invalid')
    with pytest.raises(ValueError, match="Invalid layers."):
        estimator(factorization="nonlinear", layers=[4, 3, 2])
    with pytest.raises(ValueError, match="Invalid max_iter."):
        estimator(max_iter=1.5)
    with pytest.raises(ValueError, match="Invalid max_iter."):
        estimator(max_iter=0)
    with pytest.raises(ValueError, match="Invalid tol."):
        estimator(tol=0)
    with pytest.raises(ValueError, match="Invalid update_h."):
        estimator(update_h=1)
    with pytest.raises(ValueError, match="Invalid update_last_h."):
        estimator(update_last_h=1)
    with pytest.raises(ValueError, match="Invalid update_z."):
        estimator(update_z=1)
    with pytest.raises(ValueError, match="Invalid nonlinearity."):
        estimator(nonlinearity="invalid")


def test_fit_predict(sample_data):
    n_clusters = 3
    for engine in ["python", "octave"]:
        if (engine == "octave") and not octavemodule_installed:
            continue
        for Xs in sample_data:
            model = estimator(n_clusters=n_clusters, engine=engine, random_state=42)
            n_samples = len(Xs[0])
            labels = model.fit_predict(Xs)
            assert len(labels) == n_samples
            assert len(np.unique(labels)) == n_clusters
            assert min(labels) == 0
            assert max(labels) == (n_clusters - 1)
            assert not np.isnan(labels).any()
            assert not np.isnan(model.embedding_).any().any()
            assert model.embedding_.shape == (n_samples, n_clusters)
            assert model.n_iter_ > 0


def test_missing_values_handling(sample_data):
    n_clusters = 2
    for engine in ["python", "octave"]:
        if (engine == "octave") and not octavemodule_installed:
            continue
        for Xs in sample_data:
            Xs = Amputer(p= 0.3, random_state=42).fit_transform(Xs)
            model = estimator(n_clusters=n_clusters, engine=engine, random_state=42)
            n_samples = len(Xs[0])
            Xs = Amputer(p= 0.3, random_state=42).fit_transform(Xs)
            labels = model.fit_predict(Xs)
            assert len(labels) == n_samples
            assert len(np.unique(labels)) == n_clusters
            assert min(labels) == 0
            assert max(labels) == (n_clusters - 1)
            assert not np.isnan(labels).any()
            assert not np.isnan(model.embedding_).any().any()
            assert model.embedding_.shape == (n_samples, n_clusters)
            assert model.n_iter_ > 0


def test_nonlinear_fit_predict(sample_data):
    n_clusters = 2
    for engine in ["python", "octave"]:
        if (engine == "octave") and not octavemodule_installed:
            continue
        model = estimator(n_clusters=n_clusters, factorization="nonlinear", engine=engine, random_state=42)
        n_samples = len(sample_data[0][0])
        labels = model.fit_predict(sample_data[0])
        assert len(labels) == n_samples
        assert len(np.unique(labels)) == n_clusters
        assert not np.isnan(labels).any()
        assert not np.isnan(model.embedding_).any().any()
        assert model.embedding_.shape == (n_samples, n_clusters)
        assert model.n_iter_ > 0


def test_python_initial_factors_and_helpers(monkeypatch):
    Xs = (
        np.array([[1., 2., 3., 4.], [2., 3., 4., 5.]]),
        np.array([[2., 1., 4., 3.], [3., 2., 5., 4.]]),
    )
    layers = [3, 2]
    init_z = [
        [np.ones((2, 3)) * 0.2, np.ones((3, 2)) * 0.3],
        [np.ones((2, 3)) * 0.4, np.ones((3, 2)) * 0.5],
    ]
    init_h = [
        [np.ones((3, 4)) * 0.6, np.ones((2, 4)) * 0.7],
        [np.ones((3, 4)) * 0.8, np.ones((2, 4)) * 0.9],
    ]

    model = estimator(
        n_clusters=2, engine="python", layers=layers, init_z=init_z,
        init_h=[[H.copy() for H in view] for view in init_h], random_state=42,
        verbose=True
    )

    def raise_linalg_error(_):
        raise np.linalg.LinAlgError("forced")

    monkeypatch.setattr(np.linalg, "pinv", raise_linalg_error)
    Hc, H, loss = model._mlmf_linear(Xs)
    assert Hc.shape == (2, 4)
    assert len(H) == len(Xs)
    assert len(loss) == 20
    assert np.isfinite(loss).all()

    monkeypatch.undo()
    model = estimator(
        n_clusters=2, factorization="nonlinear", engine="python", layers=layers,
        init_z=init_z, init_h=[[H.copy() for H in view] for view in init_h],
        random_state=42
    )
    Hc, H, loss = model._mlmf_nonlinear(Xs)
    assert Hc.shape == (2, 4)
    assert len(H) == len(Xs)
    assert len(loss) == 30
    assert np.isfinite(loss).all()


def test_python_private_numeric_branches(monkeypatch):
    model = estimator(n_clusters=2, engine="python", random_state=42, verbose=True)
    model.rng = np.random.default_rng(42)

    Z, H, dnorm = model._seminmf(
        np.ones((2, 2)),
        2,
        z0=np.ones((2, 2)),
        h0=np.ones((2, 2)),
        max_iter=1,
        update_z=False,
    )
    assert Z.shape == (2, 2)
    assert H.shape == (2, 2)
    assert np.isfinite(dnorm)

    def raise_linalg_error(_):
        raise np.linalg.LinAlgError("forced")

    monkeypatch.setattr(np.linalg, "pinv", raise_linalg_error)
    Z, H, dnorm = model._seminmf(
        np.ones((2, 2)),
        2,
        z0=np.ones((2, 2)),
        h0=np.ones((2, 2)),
        max_iter=1,
        update_z=True,
    )
    assert Z.shape == (2, 2)
    assert H.shape == (2, 2)
    assert np.isfinite(dnorm)
    monkeypatch.undo()

    H_list, dnorm1 = model._gd_h(
        X=np.zeros((2, 2)),
        Z=[np.ones((2, 2))],
        H=[np.ones((2, 2))],
        c=-np.ones((2, 2)),
        layer_idx=0,
        g_inv=lambda x: x,
        dnorm=0,
        E=np.ones((2, 2)),
        Hc_view=np.ones((2, 2)),
    )
    assert len(H_list) == 1
    assert np.isfinite(dnorm1)

    for nonlinearity, values in [
        ("square", np.array([[1., 4.]])),
        ("sigmoid", np.array([[0.25, 0.75]])),
        ("softplus", np.array([[1., 2.]])),
    ]:
        g, g_inv, g_inv_diff = estimator._nonlinear_functions(nonlinearity)
        assert np.isfinite(g(values)).all()
        assert np.isfinite(g_inv(values)).all()
        assert np.isfinite(g_inv_diff(values)).all()


if __name__ == "__main__":
    pytest.main()
