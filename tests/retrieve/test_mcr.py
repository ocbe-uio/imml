import pytest
torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
import importlib
import os
import shutil
import sys
from unittest.mock import patch
import numpy as np
import pytest
import pandas as pd

from imml.retrieve import MCR

estimator = MCR


@pytest.fixture
def sample_data():
    images = ["docs/figures/graph.png", "docs/figures/logo_imml.png"]
    texts = ["This is the graphical abstract of iMML.", "This is the logo of iMML."]
    Xs = [images, texts]
    y = pd.Series([0, 1])
    return Xs, y


def test_deepmodule_not_installed(sample_data):
    estimator(modalities=["image", "text"])
    with patch.dict(sys.modules, {"torch": None}):
        import imml.retrieve.mcr as module_mock
        importlib.reload(module_mock)
        with pytest.raises(ImportError, match="Module 'deep' needs to be installed."):
            estimator(modalities=["image", "text"])
    importlib.reload(module_mock)


def test_default_params(sample_data):
    Xs, y = sample_data
    model = estimator(modalities=["image", "text"], n_neighbors=1)
    model.fit(Xs, y)
    assert hasattr(model, 'memory_bank_')
    assert isinstance(model.memory_bank_, pd.DataFrame)
    assert len(model.memory_bank_) == len(y)
    predictions = model.predict(Xs)
    assert isinstance(predictions, dict)
    assert "image" in predictions
    assert "text" in predictions


def test_invalid_params(sample_data):
    with pytest.raises(ValueError, match="Invalid modalities."):
        estimator(modalities=None)
    with pytest.raises(ValueError, match="Invalid modalities."):
        estimator(modalities=["image"])
    with pytest.raises(ValueError, match="Invalid modalities."):
        estimator(modalities=["other", "image"])
    with pytest.raises(ValueError, match="Invalid batch_size."):
        estimator(modalities=["image", "text"], batch_size=None)
    with pytest.raises(ValueError, match="Invalid batch_size."):
        estimator(modalities=["image", "text"], batch_size=-1)
    with pytest.raises(ValueError, match="Invalid n_neighbors."):
        estimator(modalities=["image", "text"], n_neighbors=None)
    with pytest.raises(ValueError, match="Invalid n_neighbors."):
        estimator(modalities=["image", "text"], n_neighbors=-1)
    with pytest.raises(ValueError, match="Invalid device."):
        estimator(modalities=["image", "text"], device=123)
    with pytest.raises(ValueError, match="Invalid generate_cap."):
        estimator(modalities=["image", "text"], generate_cap="True")
    with pytest.raises(ValueError, match="Invalid prompt_path."):
        estimator(modalities=["image", "text"], generate_cap=True, prompt_path=1)
    with pytest.raises(ValueError, match="Invalid prompt_path."):
        estimator(modalities=["image", "text"], generate_cap=True, prompt_path="other")
    with pytest.raises(ValueError, match="Invalid prompt_path."):
        estimator(modalities=["image", "text"], generate_cap=True)
    with pytest.raises(ValueError, match="Invalid max_text_len."):
        estimator(modalities=["image", "text"], max_text_len=None)
    with pytest.raises(ValueError, match="Invalid max_text_len."):
        estimator(modalities=["image", "text"], max_text_len=-1)
    with pytest.raises(ValueError, match="Invalid max_image_len."):
        estimator(modalities=["image", "text"], max_image_len=None)
    with pytest.raises(ValueError, match="Invalid max_image_len."):
        estimator(modalities=["image", "text"], max_image_len=-1)
    with pytest.raises(ValueError, match="Invalid save_memory_bank."):
        estimator(modalities=["image", "text"], save_memory_bank=-1)

    Xs, y = sample_data
    with pytest.raises(ValueError, match="Invalid Xs."):
        estimator(modalities=["image", "text"], n_neighbors=1).fit(1, y)
    with pytest.raises(ValueError, match="Invalid Xs."):
        estimator(modalities=["image", "text"], n_neighbors=1).fit([Xs[0][:1], Xs[1]], y)
    with pytest.raises(ValueError, match="Invalid Xs."):
        estimator(modalities=["image", "text"], n_neighbors=1).fit([[1]], y)
    with pytest.raises(ValueError, match="Invalid Xs."):
        estimator(modalities=["image", "text"], n_neighbors=1).fit([[1], []], y)
    with pytest.raises(ValueError, match="Invalid y."):
        estimator(modalities=["image", "text"], n_neighbors=1).fit(Xs, None)
    with pytest.raises(ValueError, match="Invalid y."):
        estimator(modalities=["image", "text"], n_neighbors=1).fit(Xs, [1])
    with pytest.raises(ValueError, match="Invalid n_neighbors."):
        estimator(modalities=["image", "text"], n_neighbors=1).fit_predict(Xs, y, n_neighbors=-1)


def test_fit_methods(sample_data, tmp_path):
    Xs, y = sample_data
    model = estimator(modalities=["image", "text"], n_neighbors=1)
    result = model.fit(Xs, y)
    assert result is model
    assert hasattr(model, 'memory_bank_')
    assert isinstance(model.memory_bank_, pd.DataFrame)
    assert len(model.memory_bank_) == len(y)
    model = estimator(modalities=["image", "text"], save_memory_bank=False)
    result = model.fit(Xs, y)
    assert isinstance(result, pd.DataFrame)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(y)
    model = estimator(modalities=["image", "text"], n_neighbors=1, generate_cap=True, prompt_path=str(tmp_path))
    transformed = model.fit_transform(Xs, y)
    assert isinstance(transformed, pd.DataFrame)
    transformed = model.fit_transform(Xs, y, n_neighbors=1)
    assert isinstance(transformed, pd.DataFrame)
    model = estimator(modalities=["image", "text"], n_neighbors=1)
    predictions = model.fit_predict(Xs, y)
    assert isinstance(predictions, dict)
    assert "image" in predictions
    assert "text" in predictions
    predictions = model.fit_predict(Xs, y, n_neighbors=1)
    assert isinstance(predictions, dict)
    shutil.rmtree(tmp_path, ignore_errors=True)
    assert not os.path.exists(tmp_path)


def test_missing_values_handling(sample_data, tmp_path):
    Xs, y = sample_data
    model = estimator(modalities=["image", "text"], n_neighbors=1, generate_cap=True, prompt_path=str(tmp_path))
    model.fit(Xs, y)
    assert hasattr(model, 'memory_bank_')
    assert isinstance(model.memory_bank_, pd.DataFrame)
    assert len(model.memory_bank_) == len(y)

    # Test with one missing value
    Xs_with_missing = [Xs[0].copy(), Xs[1].copy()]
    Xs_with_missing[0][0] = np.nan
    predictions = model.predict(Xs_with_missing)
    assert isinstance(predictions, dict)
    assert "image" in predictions
    assert "text" in predictions

    # Test transform with missing values
    transformed = model.transform(Xs_with_missing, y)
    assert isinstance(transformed, pd.DataFrame)
    assert "observed_image" in transformed.columns
    assert "observed_text" in transformed.columns
    assert (transformed["observed_image"] == pd.notna(Xs_with_missing[0]).astype(int)).all()
    assert (transformed["observed_text"] == pd.notna(Xs_with_missing[1]).astype(int)).all()

    # Test with multiple missing values
    Xs_with_more_missing = [Xs[0].copy(), Xs[1].copy()]
    Xs_with_more_missing[0][0] = np.nan
    Xs_with_more_missing[1][1] = np.nan
    predictions = model.predict(Xs_with_more_missing)
    assert isinstance(predictions, dict)

    # Test transform with multiple missing values
    transformed = model.transform(Xs_with_more_missing, y)
    assert isinstance(transformed, pd.DataFrame)
    assert (transformed["observed_image"] == pd.notna(Xs_with_more_missing[0]).astype(int)).all()
    assert (transformed["observed_text"] == pd.notna(Xs_with_more_missing[1]).astype(int)).all()

    # Clean up
    shutil.rmtree(tmp_path, ignore_errors=True)
    assert not os.path.exists(tmp_path)


if __name__ == "__main__":
    pytest.main()
