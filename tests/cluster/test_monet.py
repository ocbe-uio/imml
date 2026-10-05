from string import ascii_lowercase
import pytest
import numpy as np
import pandas as pd

from imml.ampute import Amputer
from imml.cluster import MONET
from tests._sample_data import make_sample_data

sample_data = make_sample_data(n_samples=15, n_features=10, splits=(3, 5))


@pytest.mark.parametrize("Xs", sample_data)
def test_default_params(Xs):
    model = MONET()
    n_samples = len(Xs[0])
    labels = model.fit_predict(Xs)
    assert labels is not None
    assert len(labels) == n_samples
    assert len(np.unique(labels[~np.isnan(labels)])) == model.n_clusters_
    assert np.nanmin(labels) == 0
    assert np.nanmax(labels) == (model.n_clusters_ - 1)
    assert model.total_weight_ > 0


def test_custom_parameters():
    for Xs in sample_data:
        n_samples = len(Xs[0])
        try:
            init_modules = {'module1': list(Xs[0].index[:10]),
                            'module2': list(Xs[0].index[10:]),
                            }
            model = MONET(random_state=42, percentile_remove_edge=20, init_modules=init_modules, verbose=True)
        except AttributeError:
            model = MONET(random_state=42, percentile_remove_edge=10, verbose=True, min_mod_size=1)
        labels = model.fit_predict(Xs)
        assert labels is not None
        assert len(labels) == n_samples
        assert len(np.unique(labels[~np.isnan(labels)])) == model.n_clusters_
        assert np.nanmin(labels) == 0
        assert np.nanmax(labels) == (model.n_clusters_ - 1)
        assert model.total_weight_ > 0

def test_invalid_params():
    estimator = MONET
    with pytest.raises(ValueError, match="Invalid similarity_mode."):
        estimator(similarity_mode='invalid')

def test_fit_predict():
    model = MONET(random_state=42, verbose=True)
    for Xs in sample_data:
        n_samples = len(Xs[0])
        labels = model.fit_predict(Xs)
        assert labels is not None
        assert len(labels) == n_samples
        assert len(np.unique(labels[~np.isnan(labels)])) == model.n_clusters_
        assert np.nanmin(labels) == 0
        assert np.nanmax(labels) == (model.n_clusters_ - 1)
        assert model.total_weight_ > 0

def test_missing_values_handling():
    model = MONET(random_state=42, verbose=True)
    for Xs in sample_data:
        Xs = Amputer(p= 0.3, random_state=42).fit_transform(Xs)
        labels = model.fit_predict(Xs)
        assert labels is not None
        assert len(np.unique(labels[~np.isnan(labels)])) == model.n_clusters_
        assert np.nanmin(labels) == 0
        assert np.nanmax(labels) == (model.n_clusters_ - 1)
        assert model.total_weight_ > 0


if __name__ == "__main__":
    pytest.main()
