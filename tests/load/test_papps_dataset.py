import pytest
torch = pytest.importorskip("torch")
pytest.importorskip("lightning")
import importlib
import sys
from unittest.mock import patch
import numpy as np
import pandas as pd

from imml.load import PAPPSDataset
from imml.utils import as_tensor_X
from tests._sample_data import make_sample_supervised_data

n_samples = 15
n_features = 10
splits = (3, 5)
sample_data = make_sample_supervised_data(n_samples=n_samples, n_features=n_features, splits=splits)


def test_deepmodule_not_installed():
    Xs,y = sample_data[0]
    PAPPSDataset(Xs=Xs, y=y)
    with patch.dict(sys.modules, {"torch": None}):
        import imml as imml_mock
        import imml.load.papps_dataset as module_mock
        importlib.reload(imml_mock)
        importlib.reload(module_mock)
        with pytest.raises(ImportError, match="Module 'deep' needs to be installed."):
            PAPPSDataset(Xs=Xs, y=y)
    importlib.reload(imml_mock)
    importlib.reload(module_mock)


@pytest.mark.parametrize("Xs", sample_data)
def test_default_params(Xs):
    Xs,y = Xs
    dataset = PAPPSDataset(Xs=Xs, y=y)
    assert len(dataset) == len(Xs[0])
    assert hasattr(dataset, 'Xs')
    assert len(dataset.Xs) == len(Xs)
    assert hasattr(dataset, 'y')
    assert len(dataset.y) == len(y)
    assert dataset.class_distribution.tolist() == pytest.approx([6, 9])
    assert not dataset.missing_mod_indicator.any()


@pytest.mark.parametrize("Xs", sample_data)
def test_getitem(Xs):
    Xs,y = Xs
    dataset = PAPPSDataset(Xs=Xs, y=y)
    sample = dataset[0]
    assert isinstance(sample, tuple)
    assert len(sample) == 3
    for s, X in zip(sample[0], Xs):
        assert s.shape == (X.shape[1],)
    assert torch.isnan(torch.as_tensor(sample[1])) == torch.isnan(as_tensor_X(y)[0])
    assert torch.equal(sample[2], torch.tensor([6, 9]))


@pytest.mark.parametrize("Xs", sample_data)
def test_infers_missing_views_and_unlabeled_targets(Xs):
    Xs,y = Xs
    if isinstance(Xs[0], pd.DataFrame):
        Xs[0] = Xs[0].copy()
        Xs[1] = Xs[1].copy()
        Xs[0].iloc[0, :] = np.nan
        Xs[1].iloc[1, :] = np.nan
        y = y.astype(float)
        y.iloc[2] = np.nan
    elif isinstance(Xs[0], np.ndarray):
        Xs[0] = Xs[0].copy()
        Xs[1] = Xs[1].copy()
        Xs[0][0, :] = np.nan
        Xs[1][1, :] = np.nan
        y = y.astype(float)
        y[2] = np.nan
    elif isinstance(Xs[0], torch.Tensor):
        Xs[0] = Xs[0].clone()
        Xs[1] = Xs[1].clone()
        Xs[0][0, :] = float("nan")
        Xs[1][1, :] = float("nan")
        y = y.float()
        y[2] = float("nan")

    dataset = PAPPSDataset(Xs=Xs, y=y)
    assert dataset.missing_mod_indicator[0].tolist() == [True, False, False]
    assert dataset.missing_mod_indicator[1].tolist() == [False, True, False]
    assert dataset.y_indicator[2].item() is False
    assert torch.isnan(dataset[2][1])
    assert torch.isnan(dataset[0][0][0]).all()


def test_missing_and_invalid_labels():
    Xs = [np.ones((4, 2)), np.ones((4, 3))]

    with pytest.raises(ValueError, match="y. It cannot be None"):
        PAPPSDataset(Xs, None)
    with pytest.raises(ValueError, match="At least one training label"):
        PAPPSDataset(Xs, [np.nan] * 4)
    with pytest.raises(ValueError, match="At least one training label"):
        PAPPSDataset(Xs, [np.nan] * 4, class_distribution=[2, 2])
    with pytest.raises(ValueError, match="non-negative integer"):
        PAPPSDataset(Xs, [0, 0.5, 1, 1])
    with pytest.raises(ValueError, match="non-negative integer"):
        PAPPSDataset(Xs, [0, -1, 1, 1], class_distribution=[2, 2])
    with pytest.raises(ValueError, match="smaller than the number"):
        PAPPSDataset(Xs, [0, 1, 2, 1], class_distribution=[2, 2])
    with pytest.raises(ValueError, match="same number of samples"):
        PAPPSDataset(Xs, [0, 1, 0])


@pytest.mark.parametrize(
    "class_distribution",
    ([4], [1, 1, 1], [-1, 5], [1.5, 2.5], [float("nan"), 4]),
)
def test_invalid_class_distribution(class_distribution):
    Xs = [np.ones((4, 2)), np.ones((4, 3))]
    with pytest.raises(ValueError, match="class_distribution"):
        PAPPSDataset(Xs, [0, 0, 1, 1], class_distribution=class_distribution)


def test_class_distribution_must_match_dataset_size():
    Xs = [np.ones((4, 2)), np.ones((4, 3))]
    with pytest.raises(ValueError, match="sum to the number of samples"):
        PAPPSDataset(Xs, [0, 0, 1, 1], class_distribution=[1, 2])


if __name__ == "__main__":
    pytest.main()
