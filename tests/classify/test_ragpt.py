import pytest
torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
L = pytest.importorskip("lightning")
import copy
import importlib
import os
import shutil
import sys
import tempfile
from unittest.mock import patch
from torch.utils.data import DataLoader

from imml.classify import RAGPT
from imml.retrieve import MCR
from imml.load import RAGPTCollator, RAGPTDataset

estimator = RAGPT


@pytest.fixture
def sample_data():
    images = ["docs/figures/graph.png", "docs/figures/logo_imml.png"]
    texts = ["This is the graphical abstract of iMML.", "This is the logo of iMML."]
    Xs = [images, texts]
    y = [0, 1]
    tmp_path = tempfile.mkdtemp()
    model = MCR(modalities=["image", "text"], n_neighbors=1, generate_cap=True, prompt_path=str(tmp_path))
    database = model.fit_transform(Xs, y)
    train_data = RAGPTDataset(database)
    loader = DataLoader(train_data, collate_fn=RAGPTCollator(), batch_size=len(train_data))
    batch = next(iter(loader))
    return batch, tmp_path


def test_deepmodule_not_installed():
    estimator()
    with patch.dict(sys.modules, {"torch": None}):
        import imml.classify.ragpt as module_mock
        importlib.reload(module_mock)
        with pytest.raises(ImportError, match="Module 'deep' needs to be installed."):
            estimator()
    importlib.reload(module_mock)


def test_default_params(sample_data):
    model = estimator()
    assert hasattr(model, 'model')
    assert hasattr(model, 'loss')
    assert hasattr(model, 'learning_rate')
    assert hasattr(model, 'weight_decay')
    batch, tmp_path = sample_data
    with torch.no_grad():
        loss = model.training_step(batch)
    assert isinstance(loss, torch.Tensor)
    shutil.rmtree(tmp_path, ignore_errors=True)
    assert not os.path.exists(tmp_path)


def test_invalid_params():
    with pytest.raises(ValueError, match="Invalid max_text_len."):
        estimator(max_text_len=None)
    with pytest.raises(ValueError, match="Invalid max_text_len."):
        estimator(max_text_len=-1)

    with pytest.raises(ValueError, match="Invalid max_image_len."):
        estimator(max_image_len=None)
    with pytest.raises(ValueError, match="Invalid max_image_len."):
        estimator(max_image_len=-1)

    # Test invalid prompt_position
    with pytest.raises(ValueError, match="Invalid prompt_position."):
        estimator(prompt_position=None)
    with pytest.raises(ValueError, match="Invalid prompt_position."):
        estimator(prompt_position=-1)

    # Test invalid prompt_length
    with pytest.raises(ValueError, match="Invalid prompt_length."):
        estimator(prompt_length=None)
    with pytest.raises(ValueError, match="Invalid prompt_length."):
        estimator(prompt_length=-1)

    # Test invalid dropout_rate
    with pytest.raises(ValueError, match="Invalid dropout_rate."):
        estimator(dropout_rate=None)
    with pytest.raises(ValueError, match="Invalid dropout_rate."):
        estimator(dropout_rate=-1)
    with pytest.raises(ValueError, match="Invalid dropout_rate."):
        estimator(dropout_rate=2)

    # Test invalid hidden_dim
    with pytest.raises(ValueError, match="Invalid hidden_dim."):
        estimator(hidden_dim=None)
    with pytest.raises(ValueError, match="Invalid hidden_dim."):
        estimator(hidden_dim=-1)

    # Test invalid cls_num
    with pytest.raises(ValueError, match="Invalid cls_num."):
        estimator(cls_num=None)
    with pytest.raises(ValueError, match="Invalid cls_num."):
        estimator(cls_num=-1)

    # Test invalid loss
    with pytest.raises(ValueError, match="Invalid loss."):
        estimator(loss="not_callable")

    # Test invalid learning_rate
    with pytest.raises(ValueError, match="Invalid learning_rate."):
        estimator(learning_rate=None)
    with pytest.raises(ValueError, match="Invalid learning_rate."):
        estimator(learning_rate=-1)

    # Test invalid weight_decay
    with pytest.raises(ValueError, match="Invalid weight_decay."):
        estimator(weight_decay=None)
    with pytest.raises(ValueError, match="Invalid weight_decay."):
        estimator(weight_decay=-1)


def test_lightning_methods(sample_data):
    # Create model
    model = estimator()

    # Test training_step
    batch, tmp_path = sample_data
    loss = model.training_step(copy.deepcopy(batch))
    assert isinstance(loss, torch.Tensor)

    # Test validation_step
    loss = model.validation_step(copy.deepcopy(batch))
    assert isinstance(loss, torch.Tensor)

    # Test test_step
    loss = model.test_step(copy.deepcopy(batch))
    assert isinstance(loss, torch.Tensor)

    # Test predict_step
    preds = model.predict_step(copy.deepcopy(batch))
    assert isinstance(preds, torch.Tensor)

    # Test configure_optimizers
    optimizer = model.configure_optimizers()
    assert isinstance(optimizer, torch.optim.Optimizer)
    shutil.rmtree(tmp_path, ignore_errors=True)
    assert not os.path.exists(tmp_path)


def test_missing_values_handling(sample_data):
    # Create model
    model = estimator()

    # Create batch with missing values
    batch, tmp_path = sample_data
    batch['observed_image'] = [0, 1]  # First image is missing
    batch['observed_text'] = [1, 0]   # Second text is missing

    # Test forward pass with missing values
    with torch.no_grad():
        loss = model.training_step(copy.deepcopy(batch))
    assert isinstance(loss, torch.Tensor)
    shutil.rmtree(tmp_path, ignore_errors=True)
    assert not os.path.exists(tmp_path)


if __name__ == "__main__":
    pytest.main()
