import numpy as np
import pandas as pd
import pytest

from imml.statistics.pid import pid


@pytest.fixture
def sample_data():
    X = np.random.default_rng(42).random((30, 10))
    X = pd.DataFrame(X)
    X1, X2, X3 = X.iloc[:, :3].copy(), X.iloc[:, 3:5].copy(), X.iloc[:, 5:10].copy()
    Xs_pandas, Xs_numpy = [X1, X2, X3], [X1.values, X2.values, X3.values]
    y_numpy = np.random.default_rng(42).choice(2, size=len(Xs_pandas[0]))
    y_pandas = pd.Series(y_numpy)
    return Xs_pandas, y_pandas, Xs_numpy, y_numpy

def test_default_params(sample_data):
    for (Xs,y) in [sample_data[:2], sample_data[2:]]:
        stats_list = pid(Xs=Xs, y=y, random_state=42)
        assert isinstance(stats_list, list)
        assert len(stats_list) == 3
        for stats in stats_list:
            assert set(stats.keys()) == {"Redundancy", "Uniqueness1", "Uniqueness2", "Synergy", "Information"}

def test_pid_two_modalities_basic(sample_data):
    for (Xs,y) in [sample_data[:2], sample_data[2:]]:
        Xs = Xs[:2]
        stats = pid(Xs=Xs, y=y, random_state=42)
        assert isinstance(stats, dict)
        for key in ["Redundancy", "Uniqueness1", "Uniqueness2", "Synergy", "Information"]:
            assert key in stats
            assert np.isfinite(stats[key])

def test_invalid_params(sample_data):
    Xs,y = sample_data[:2]
    with pytest.raises(ValueError, match="Invalid n_clusters"):
        pid(Xs=Xs, y=y, n_clusters=[5, 8, 10, 9])
    with pytest.raises(ValueError, match="Invalid n_components"):
        pid(Xs=Xs, y=y, n_components=[5, 8, 10, 9])
    with pytest.raises(ValueError, match="Invalid normalize"):
        pid(Xs=Xs, y=y, normalize=[5, 8, 10, 9])
    with pytest.raises(ValueError, match="Invalid return_index"):
        pid(Xs=Xs, y=y, return_index=[5, 8, 10, 9])
    with pytest.raises(ValueError, match="Invalid Xs"):
        pid(Xs=[Xs[0]], y=y, return_index=[5, 8, 10, 9])

def test_pid_two_modalities_normalized_and_indices_and_params_as_lists(sample_data):
    for (Xs,y) in [sample_data[:2], sample_data[2:]]:
        Xs = Xs[:2]
        n_clusters = [2, 2]
        n_components = [None, 0.95]
        stats, idxs = pid(
            Xs=Xs,
            y=y,
            n_clusters=n_clusters,
            n_components=n_components,
            random_state=0,
            normalize=True,
            return_index=True,
        )
        assert isinstance(stats, dict)
        assert idxs == [(0, 1)]
        del stats["Information"]
        total = sum(stats.values())
        assert np.isfinite(total)
        assert pytest.approx(1.0, rel=1e-6, abs=1e-6) == total

def test_pid_three_modalities_list_output_and_order(sample_data):
    for (Xs,y) in [sample_data[:2], sample_data[2:]]:
        stats_list = pid(Xs=Xs, y=y, n_clusters=2, n_components=None, random_state=0, normalize=True)
        assert isinstance(stats_list, list)
        assert len(stats_list) == 3
        for stats in stats_list:
            assert set(stats.keys()) == {"Redundancy", "Uniqueness1", "Uniqueness2", "Synergy", "Information"}
            del stats["Information"]
            assert pytest.approx(1.0, rel=1e-6, abs=1e-6) == sum(stats.values())

if __name__ == "__main__":
    pytest.main()
