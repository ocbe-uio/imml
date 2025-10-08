import pytest
torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
from transformers import BertTokenizer
import importlib
import os
import sys
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from PIL import Image

from imml.load import RAGPTDataset, RAGPTCollator


@pytest.fixture
def sample_database(tmp_path):
    n_samples = 5
    prompt_dir = os.path.join(tmp_path, "prompts")
    image_dir = os.path.join(prompt_dir, "image")
    text_dir = os.path.join(prompt_dir, "text")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(text_dir, exist_ok=True)
    prompt_image_paths = []
    prompt_text_paths = []
    for i in range(n_samples):
        for j in range(3):
            img_path = os.path.join(image_dir, f"sample_{i}_{j}.npy")
            txt_path = os.path.join(text_dir, "sample_{i}_{j}.npy")
            np.save(img_path, np.random.random((768,)))
            np.save(txt_path, np.random.random((768,)))
            prompt_image_paths.append([str(img_path)])
            prompt_text_paths.append([str(txt_path)])
    
    database = pd.DataFrame({
        'img_path': [os.path.join(tmp_path, "image", f"image_{i}.jpg") for i in range(n_samples)],
        'text': [f"Sample text {i}" for i in range(n_samples)],
        'label': np.random.randint(0, 2, n_samples),
        'i2i_id_list': [[i] for i in range(n_samples)],
        't2t_id_list': [[i] for i in range(n_samples)],
        'prompt_image_path': prompt_image_paths[:n_samples],
        'prompt_text_path': prompt_text_paths[:n_samples],
        'i2i_label_list': [[0] for _ in range(n_samples)],
        't2t_label_list': [[1] for _ in range(n_samples)],
        'observed_image': [1 for _ in range(n_samples)],
        'observed_text': [1 for _ in range(n_samples)]
    })
    
    return database, prompt_dir


def test_deepmodule_not_installed(sample_database):
    database, _ = sample_database
    RAGPTDataset(database=database)
    with patch.dict(sys.modules, {"torch": None}):
        import imml.load.ragpt_dataset as module_mock
        importlib.reload(module_mock)
        with pytest.raises(ImportError, match="Module 'deep' needs to be installed."):
            RAGPTDataset(database=database)
        with pytest.raises(ImportError, match="Module 'deep' needs to be installed."):
            RAGPTCollator()
    importlib.reload(module_mock)


@patch('imml.load.ragpt_dataset.Image.open')
def test_default_params(mock_image_open, sample_database):
    mock_img = MagicMock()
    mock_img.convert.return_value = mock_img
    mock_image_open.return_value = mock_img
    database, _ = sample_database
    dataset = RAGPTDataset(database=database)
    assert dataset.max_text_len == 128
    assert len(dataset) == len(database)
    sample = dataset[0]
    assert isinstance(sample, dict)
    assert 'image' in sample
    assert 'text' in sample
    assert 'label' in sample
    assert 'r_t_list' in sample
    assert 'r_i_list' in sample
    assert 'r_l_list' in sample
    assert 'observed_text' in sample
    assert 'observed_image' in sample


def test_invalid_params():
    with pytest.raises(ValueError, match="Invalid database."):
        RAGPTDataset(database="not_a_dataframe")
    with pytest.raises(ValueError, match="Invalid database. It is missing required columns"):
        RAGPTDataset(database=pd.DataFrame())
    with pytest.raises(ValueError, match="Invalid max_text_len."):
        RAGPTDataset(database=pd.DataFrame({
            'img_path': [], 'text': [], 'label': [], 'i2i_id_list': [], 't2t_id_list': [],
            'prompt_image_path': [], 'prompt_text_path': [], 'i2i_label_list': [],
            't2t_label_list': [], 'observed_image': [], 'observed_text': []
        }), max_text_len=-1)
    with pytest.raises(ValueError, match="Invalid max_text_len."):
        RAGPTDataset(database=pd.DataFrame({
            'img_path': [], 'text': [], 'label': [], 'i2i_id_list': [], 't2t_id_list': [],
            'prompt_image_path': [], 'prompt_text_path': [], 'i2i_label_list': [],
            't2t_label_list': [], 'observed_image': [], 'observed_text': []
        }), max_text_len=None)
    with pytest.raises(ValueError, match="Invalid max_text_len."):
        RAGPTCollator(max_text_len=None)


@patch('imml.load.ragpt_dataset.Image.open')
@patch('imml.load.ragpt_dataset.np.load')
def test_getitem_with_both_modalities(mock_np_load, mock_image_open, sample_database):
    mock_img = MagicMock()
    mock_img.convert.return_value = mock_img
    mock_image_open.return_value = mock_img
    mock_np_load.return_value = np.random.random((768,))
    database, _ = sample_database
    dataset = RAGPTDataset(database=database)
    sample = dataset[0]
    assert sample['observed_text'] == 1
    assert sample['observed_image'] == 1
    assert len(sample['r_t_list']) > 0
    assert len(sample['r_i_list']) > 0


@patch('imml.load.ragpt_dataset.Image.open')
@patch('imml.load.ragpt_dataset.np.load')
def test_getitem_with_missing_text(mock_np_load, mock_image_open, sample_database):
    mock_img = MagicMock()
    mock_img.convert.return_value = mock_img
    mock_image_open.return_value = mock_img
    mock_np_load.return_value = np.random.random((768,))

    database, _ = sample_database
    database.loc[0, 'observed_text'] = 0
    dataset = RAGPTDataset(database=database)

    sample = dataset[0]
    assert sample['observed_text'] == 0
    assert sample['observed_image'] == 1
    assert len(sample['r_t_list']) > 0
    assert len(sample['r_i_list']) > 0


@patch('imml.load.ragpt_dataset.Image.open')
@patch('imml.load.ragpt_dataset.np.load')
def test_getitem_with_missing_image(mock_np_load, mock_image_open, sample_database):
    mock_img = MagicMock()
    mock_img.convert.return_value = mock_img
    mock_image_open.return_value = mock_img
    mock_np_load.return_value = np.random.random((768,))

    database, _ = sample_database
    database.loc[0, 'observed_image'] = 0
    dataset = RAGPTDataset(database=database)

    sample = dataset[0]
    assert sample['observed_text'] == 1
    assert sample['observed_image'] == 0
    assert len(sample['r_t_list']) > 0
    assert len(sample['r_i_list']) > 0


@patch('imml.load.ragpt_dataset.Image.open')
def test_getitem_with_both_modalities_missing(mock_image_open, sample_database):
    mock_img = MagicMock()
    mock_img.convert.return_value = mock_img
    mock_image_open.return_value = mock_img

    database, _ = sample_database
    database.loc[0, 'observed_text'] = 0
    database.loc[0, 'observed_image'] = 0
    dataset = RAGPTDataset(database=database)

    with pytest.raises(ValueError, match="No available modalities for item"):
        dataset[0]


@patch('transformers.BertTokenizer.from_pretrained')
@patch('imml.classify._ragpt.vilt.ViltImageProcessor.from_pretrained')
def test_collator_default_params(mock_vilt_processor, mock_bert_tokenizer):
    mock_bert_tokenizer.return_value = MagicMock()
    mock_vilt_processor.return_value = MagicMock()
    collator = RAGPTCollator()
    assert collator.max_text_len == 128
    assert hasattr(collator, 'tokenizer')
    assert hasattr(collator, 'image_processor')


def test_collator_invalid_params():
    with pytest.raises(ValueError, match="Invalid tokenizer."):
        RAGPTCollator(tokenizer="not_a_tokenizer")
    with pytest.raises(ValueError, match="Invalid image_processor."):
        RAGPTCollator(image_processor="not_an_image_processor")
    with pytest.raises(ValueError, match="Invalid max_text_len."):
        RAGPTCollator(max_text_len=-1)


@patch('transformers.BertTokenizer.from_pretrained')
@patch('imml.classify._ragpt.vilt.ViltImageProcessor.from_pretrained')
@patch('imml.classify._ragpt.resize_image')
def test_collator_call(mock_resize_image, mock_vilt_processor, mock_bert_tokenizer):
    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {
        'input_ids': [[1, 2, 3], [4, 5, 6]],
        'attention_mask': [[1, 1, 1], [1, 1, 1]],
        'token_type_ids': [[0, 0, 0], [0, 0, 0]]
    }
    mock_bert_tokenizer.return_value = mock_tokenizer
    mock_processor = MagicMock()
    mock_processor.return_value = {
        'pixel_values': torch.ones((2, 3, 224, 224)),
        'pixel_mask': torch.ones((2, 224, 224))
    }
    mock_vilt_processor.return_value = mock_processor
    mock_resize_image.side_effect = lambda x: x
    collator = RAGPTCollator()
    batch = [
        {
            'image': Image.new('RGB', (224, 224)),
            'text': 'Sample text 1',
            'label': 0,
            'r_t_list': [[1.0] * 768],
            'r_i_list': [[2.0] * 768],
            'r_l_list': [0],
            'observed_text': 1,
            'observed_image': 1
        },
        {
            'image': Image.new('RGB', (224, 224)),
            'text': 'Sample text 2',
            'label': 1,
            'r_t_list': [[3.0] * 768],
            'r_i_list': [[4.0] * 768],
            'r_l_list': [1],
            'observed_text': 1,
            'observed_image': 1
        }
    ]

    result = collator(batch)
    assert isinstance(result, dict)
    assert 'input_ids' in result
    assert 'pixel_values' in result
    assert 'pixel_mask' in result
    assert 'token_type_ids' in result
    assert 'attention_mask' in result
    assert 'label' in result
    assert 'r_t_list' in result
    assert 'r_i_list' in result
    assert 'r_l_list' in result
    assert 'observed_image' in result
    assert 'observed_text' in result


if __name__ == "__main__":
    pytest.main()