from string import ascii_lowercase
import pytest
import numpy as np
import pandas as pd

from imml.ampute import Amputer
from imml.cluster import SUMO
from tests._sample_data import make_sample_data

sample_data = make_sample_data(n_samples=20, n_features=10, splits=(3, 5))


@pytest.mark.parametrize("Xs", sample_data)
def test_default_params(Xs):
    model = SUMO(random_state=42, n_clusters=5)
    n_samples = len(Xs[0])
    labels = model.fit_predict(Xs)
    assert labels is not None
    assert len(labels) == n_samples
    assert len(np.unique(labels)) == model.n_clusters
    assert min(labels) == 0
    assert max(labels) == (model.n_clusters - 1)
    assert not np.isnan(labels).any()
    assert model.embedding_.shape == (n_samples, model.n_clusters)
    assert len(model.similarity_) == len(Xs)
    assert model.similarity_["0"].shape == (n_samples, n_samples)
    assert len(model.cophenet_list_) == model.rep
    assert len(model.pac_list_) == model.rep


def test_custom_parameters():
    n_clusters = 3
    model = SUMO(n_clusters=n_clusters, random_state=42, verbose=True)
    for Xs in sample_data:
        n_samples = len(Xs[0])
        labels = model.fit_predict(Xs)
        assert labels is not None
        assert len(labels) == n_samples
        assert len(np.unique(labels)) == n_clusters
        assert min(labels) == 0
        assert max(labels) == (n_clusters - 1)
        assert not np.isnan(labels).any()
        assert model.embedding_.shape == (n_samples, n_clusters)
        assert len(model.similarity_) == len(Xs)
        assert model.similarity_["0"].shape == (n_samples, n_samples)
        assert len(model.cophenet_list_) == model.rep
        assert len(model.pac_list_) == model.rep

def test_invalid_params():
    estimator = SUMO
    with pytest.raises(ValueError, match="Invalid n_clusters."):
        estimator(n_clusters='invalid')
    with pytest.raises(ValueError, match="Invalid n_clusters."):
        estimator(n_clusters=0)
    with pytest.raises(ValueError, match="Invalid method. Expected one of"):
        estimator(method='invalid')
    with pytest.raises(ValueError, match="Invalid method. Expected one of"):
        estimator(method=['invalid'])
    with pytest.raises(ValueError, match="Incorrect repetitions. It must be repetitions"):
        estimator(repetitions=0)
    with pytest.raises(ValueError, match="Incorrect rep."):
        estimator(rep=0)
    with pytest.raises(ValueError, match="Incorrect n_jobs."):
        estimator(n_jobs=0)
    with pytest.raises(ValueError, match="Incorrect subsample."):
        estimator(subsample=0.8)
    with pytest.raises(ValueError, match="number of similarity methods does not correspond."):
        estimator(method=["euclidean"] * 5).fit(sample_data[0])
    with pytest.raises(ValueError, match="Incorrect h_init."):
        estimator(h_init=-1).fit(sample_data[0])

def test_fit_predict():
    n_clusters = 3
    model = SUMO(n_clusters=n_clusters, random_state=42)
    for Xs in sample_data:
        n_samples = len(Xs[0])
        labels = model.fit_predict(Xs)
        assert labels is not None
        assert len(labels) == n_samples
        assert len(np.unique(labels)) == n_clusters
        assert min(labels) == 0
        assert max(labels) == (n_clusters - 1)
        assert not np.isnan(labels).any()
        assert model.embedding_.shape == (n_samples, n_clusters)
        assert len(model.similarity_) == len(Xs)
        assert model.similarity_["0"].shape == (n_samples, n_samples)
        assert len(model.cophenet_list_) == model.rep
        assert len(model.pac_list_) == model.rep

def test_missing_values_handling():
    n_clusters = 2
    model = SUMO(n_clusters=n_clusters, random_state=42)
    for Xs in sample_data:
        Xs = Amputer(p= 0.3, random_state=42).fit_transform(Xs)
        n_samples = len(Xs[0])
        labels = model.fit_predict(Xs)
        assert labels is not None
        assert len(labels) == n_samples
        assert len(np.unique(labels)) == n_clusters
        assert min(labels) == 0
        assert max(labels) == (n_clusters - 1)
        assert not np.isnan(labels).any()
        assert model.embedding_.shape == (n_samples, n_clusters)
        assert len(model.similarity_) == len(Xs)
        assert model.similarity_["0"].shape == (n_samples, n_samples)
        assert len(model.cophenet_list_) == model.rep
        assert len(model.pac_list_) == model.rep

if __name__ == "__main__":
    pytest.main()
