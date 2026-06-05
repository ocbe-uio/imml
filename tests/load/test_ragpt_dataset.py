import shutil
import tempfile
from copy import deepcopy

import pytest
torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
import importlib
import os
import sys
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from imml.load import RAGPTDataset

modalities=["image", "text"]
n_neighbors=1


@pytest.fixture
def sample_data(tmp_path):
    Xs = [
        pd.DataFrame(["docs/figures/graph.png", "docs/figures/logo_imml.png"]),
        pd.DataFrame(["This is the graphical abstract of iMML.", "This is the logo of iMML."]),
    ]
    y = pd.Series([0, 1])
    return Xs, y


def test_deepmodule_not_installed(sample_data, tmp_path):
    Xs, y = sample_data
    RAGPTDataset(Xs=Xs, y=y, Xs_bank=Xs, y_bank=y, modalities=modalities,
                 prompt_path=str(tmp_path), n_neighbors=n_neighbors)
    with patch.dict(sys.modules, {"torch": None}):
        import imml as imml_mock
        import imml.load.ragpt_dataset as module_mock
        importlib.reload(imml_mock)
        importlib.reload(module_mock)
        with pytest.raises(ImportError, match="Module 'deep' needs to be installed."):
            RAGPTDataset(Xs=Xs, y=y, Xs_bank=Xs, y_bank=y, modalities=modalities,
                         prompt_path=str(tmp_path), n_neighbors=n_neighbors)
    importlib.reload(imml_mock)
    importlib.reload(module_mock)
    shutil.rmtree(tmp_path, ignore_errors=True)
    assert not os.path.exists(tmp_path)


def test_default_params(sample_data, tmp_path):
    Xs, y = sample_data
    data = RAGPTDataset(Xs=Xs, y=y, Xs_bank=Xs, y_bank=y, modalities=modalities,
                        prompt_path=str(tmp_path), n_neighbors=n_neighbors)
    assert hasattr(data, 'mcr_')
    assert hasattr(data, 'img_path_list_')
    assert hasattr(data, 'input_ids_list_')
    assert hasattr(data, 'attention_mask_list_')
    assert hasattr(data, 'token_type_ids_list_')
    assert hasattr(data, 'label_list_')
    assert hasattr(data, 'prompt_image_path_')
    assert hasattr(data, 'prompt_text_path_')
    assert hasattr(data, 'i2i_r_l_list_')
    assert hasattr(data, 't2t_r_l_list_')
    assert hasattr(data, 'observed_image_')
    assert hasattr(data, 'observed_text_')
    assert len(y) == len(data)
    sample = data[0]
    assert isinstance(sample, dict)
    assert 'image' in sample
    assert 'input_ids' in sample
    assert 'attention_mask' in sample
    assert 'token_type_ids' in sample
    assert 'label' in sample
    assert 'r_t_list' in sample
    assert 'r_i_list' in sample
    assert 'r_l_list' in sample
    assert 'observed_text' in sample
    assert 'observed_image' in sample
    shutil.rmtree(tmp_path, ignore_errors=True)
    assert not os.path.exists(tmp_path)


def test_invalid_params(sample_data, tmp_path):
    Xs, y = sample_data
    with pytest.raises(ValueError, match="Invalid modalities."):
        RAGPTDataset(Xs=Xs, y=y, modalities=None, n_neighbors=n_neighbors,
                     Xs_bank=Xs, y_bank=y, prompt_path=str(tmp_path))
    with pytest.raises(ValueError, match="Invalid batch_size."):
        RAGPTDataset(Xs=Xs, y=y, modalities=modalities, batch_size=None, n_neighbors=n_neighbors,
                     Xs_bank=Xs, y_bank=y, prompt_path=str(tmp_path))
    with pytest.raises(ValueError, match="Invalid batch_size."):
        RAGPTDataset(Xs=Xs, y=y, modalities=modalities, batch_size=-1,
                     Xs_bank=Xs, y_bank=y, prompt_path=str(tmp_path))
    with pytest.raises(ValueError, match="Invalid n_neighbors."):
        RAGPTDataset(Xs=Xs, y=y, modalities=modalities, n_neighbors=None,
                     Xs_bank=Xs, y_bank=y, prompt_path=str(tmp_path))
    with pytest.raises(ValueError, match="Invalid n_neighbors."):
        RAGPTDataset(Xs=Xs, y=y, modalities=modalities, n_neighbors=-1,
                     Xs_bank=Xs, y_bank=y, prompt_path=str(tmp_path))
    with pytest.raises(ValueError, match="Invalid device."):
        RAGPTDataset(Xs=Xs, y=y, modalities=modalities, device=123, n_neighbors=n_neighbors,
                     Xs_bank=Xs, y_bank=y, prompt_path=str(tmp_path))
    with pytest.raises(ValueError, match="Invalid prompt_path."):
        RAGPTDataset(Xs=Xs, y=y, modalities=modalities, prompt_path=1, n_neighbors=n_neighbors,
                     Xs_bank=Xs, y_bank=y)
    with pytest.raises(ValueError, match="Invalid prompt_path."):
        RAGPTDataset(Xs=Xs, y=y, modalities=modalities, prompt_path="other", n_neighbors=n_neighbors,
                     Xs_bank=Xs, y_bank=y)
    with pytest.raises(TypeError, match="Invalid prompt_path."):
        RAGPTDataset(Xs=Xs, y=y, modalities=modalities, n_neighbors=n_neighbors,
                     Xs_bank=Xs, y_bank=y)
    with pytest.raises(ValueError, match="Invalid max_text_len."):
        RAGPTDataset(Xs=Xs, y=y, modalities=modalities, max_text_len=None, n_neighbors=n_neighbors,
                     Xs_bank=Xs, y_bank=y, prompt_path=str(tmp_path))
    with pytest.raises(ValueError, match="Invalid max_text_len."):
        RAGPTDataset(Xs=Xs, y=y, modalities=modalities, max_text_len=-1, n_neighbors=n_neighbors,
                     Xs_bank=Xs, y_bank=y, prompt_path=str(tmp_path))
    with pytest.raises(ValueError, match="Invalid max_image_len."):
        RAGPTDataset(Xs=Xs, y=y, modalities=modalities, max_image_len=None, n_neighbors=n_neighbors,
                     Xs_bank=Xs, y_bank=y, prompt_path=str(tmp_path))
    with pytest.raises(ValueError, match="Invalid max_image_len."):
        RAGPTDataset(Xs=Xs, y=y, modalities=modalities, max_image_len=-1, n_neighbors=n_neighbors,
                     Xs_bank=Xs, y_bank=y, prompt_path=str(tmp_path))
    shutil.rmtree(tmp_path, ignore_errors=True)
    assert not os.path.exists(tmp_path)


def test_getitem_with_both_modalities(sample_data, tmp_path):
    Xs, y = sample_data
    data = RAGPTDataset(Xs=Xs, y=y, Xs_bank=Xs, y_bank=y, modalities=modalities,
                        prompt_path=str(tmp_path), n_neighbors=n_neighbors)
    sample = data[0]
    assert sample['observed_text'] == 1
    assert sample['observed_image'] == 1
    assert len(sample['r_t_list']) > 0
    assert len(sample['r_i_list']) > 0

    shutil.rmtree(tmp_path, ignore_errors=True)
    assert not os.path.exists(tmp_path)


def test_getitem_with_missing_text(sample_data, tmp_path):
    Xs, y = sample_data
    Xs_bank = deepcopy(Xs)
    Xs[1][0][0] = np.nan
    data = RAGPTDataset(Xs=Xs, y=y, Xs_bank=Xs_bank, y_bank=y, modalities=modalities,
                        prompt_path=str(tmp_path), n_neighbors=n_neighbors)
    sample = data[0]
    assert sample['observed_text'] == 0
    assert sample['observed_image'] == 1
    assert len(sample['r_t_list']) > 0
    assert len(sample['r_i_list']) > 0

    shutil.rmtree(tmp_path, ignore_errors=True)
    assert not os.path.exists(tmp_path)


def test_getitem_with_missing_image(sample_data, tmp_path):
    Xs, y = sample_data
    Xs_bank = deepcopy(Xs)
    Xs[0][0][0] = np.nan
    data = RAGPTDataset(Xs=Xs, y=y, Xs_bank=Xs_bank, y_bank=y, modalities=modalities,
                        prompt_path=str(tmp_path), n_neighbors=n_neighbors)
    sample = data[0]
    assert sample['observed_text'] == 1
    assert sample['observed_image'] == 0
    assert len(sample['r_t_list']) > 0
    assert len(sample['r_i_list']) > 0

    shutil.rmtree(tmp_path, ignore_errors=True)
    assert not os.path.exists(tmp_path)


def test_example(sample_data, tmp_path):
    from imml.load import RAGPTDataset
    Xs = [
        pd.DataFrame(["docs/figures/graph.png", "docs/figures/logo_imml.png"]),
        pd.DataFrame(["This is the graphical abstract of iMML.", "This is the logo of iMML."]),
    ]
    y = [0, 1]
    modalities = ["image", "text"]
    tmp_path = tempfile.mkdtemp()
    train_data = RAGPTDataset(Xs=Xs, y=y, Xs_bank=Xs, y_bank=y, modalities=modalities,
                              n_neighbors=1, prompt_path=str(tmp_path))
    shutil.rmtree(tmp_path, ignore_errors=True)
    assert not os.path.exists(tmp_path)


if __name__ == "__main__":
    pytest.main()