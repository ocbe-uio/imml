# License: BSD-3-Clause

from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

try:
    import lightning.pytorch as pl
    from transformers import BertTokenizer
    import torch
    from ..classify._ragpt.core_tools import resize_image
    from ..classify._ragpt.vilt import ViltImageProcessor
    deepmodule_installed = True
except ImportError:
    deepmodule_installed = False
    deepmodule_error = "Module 'deep' needs to be installed. See https://imml.readthedocs.io/stable/main/installation.html#optional-dependencies"

TorchDatasetBase = torch.utils.data.Dataset if deepmodule_installed else object
BertTokenizer = BertTokenizer if deepmodule_installed else object
ViltImageProcessor = ViltImageProcessor if deepmodule_installed else object


class RAGPTDataset(TorchDatasetBase):
    r"""
    This class provides a `torch.utils.data.Dataset` implementation for handling multi-modal datasets with `RAGPT`. If 
    it is used with `torch.utils.data.DataLoader`, the `collate_fn` argument of the `DataLoader` constructor should be  
    `RAGPTCollator`.

    Parameters
    ----------
    database : pd.DataFrame (n_samples, 14)
        A database with the retrieval-augmented prompts created by MCR. It must contain the following columns:
            - item_id: Unique identifier for each sample.
            - img_path: Path to the image file.
            - text: Textual content of the sample.
            - label: Label of the sample.
            - observed_image: Indicator of whether the image was observed.
            - observed_text: Indicator of whether the text was observed.
            - i2i_id_list: List of ids of the retrieved items for the image-to-image modality.
            - i2i_sims_list: List of similarities of the retrieved items for the image-to-image modality.
            - i2i_label_list: List of labels of the retrieved items for the image-to-image modality.
            - prompt_image_path: Path to the generated image prompt. Only if generate_cap is True.
            - t2t_id_list: List of ids of the retrieved items for the text-to-text modality.
            - t2t_sims_list: List of similarities of the retrieved items for the text-to-text modality.
            - t2t_label_list: List of labels of the retrieved items for the text-to-text modality.
            - prompt_text_path: Path to the generated text prompt. Only if generate_cap is True.

    max_text_len : int, default=128
        Maximum token length for text inputs (used during prompt generation).

    Returns
    -------
    sample : dict
        Dictionary with the following keys for one sample:
            - image: Image of the sample.
            - text: Textual content of the sample.
            - label: Label of the sample.
            - r_t_list: List with retrieved textual content for the sample.
            - r_i_list: List with retrieved image content for the sample.
            - r_l_list: List with retrieved labels for the sample.
            - observed_text: True if the text is observed, False otherwise.
            - observed_image: True if the image is observed, False otherwise.

    Example
    --------
    >>> from torch.utils.data import DataLoader
    >>> from imml.load import RAGPTDataset
    >>> from imml.retrieve import MCR
    >>> images = ["docs/figures/graph.png", "docs/figures/logo_imml.png",
                  "docs/figures/graph.png", "docs/figures/logo_imml.png"]
    >>> texts = ["This is the graphical abstract of iMML.", "This is the logo of iMML.",
                 "This is the graphical abstract of iMML.", "This is the logo of iMML."]
    >>> Xs = [images, texts]
    >>> y = [0, 1, 0, 1]
    >>> modalities = ["image", "text"]
    >>> estimator = MCR(modalities=modalities)
    >>> database = estimator.fit_transform(Xs=Xs, y=y)
    >>> train_data = RAGPTDataset(database=database)
    >>> train_dataloader = DataLoader(train_data, collate_fn=RAGPTCollator())
    """


    def __init__(self, database: pd.DataFrame, max_text_len: int = 128):
        if not deepmodule_installed:
            raise ImportError(deepmodule_error)

        if not isinstance(database, pd.DataFrame):
            raise ValueError(f"Invalid database. It must be a pandas DataFrame. A {type(database)} was passed.")
        required_columns = ['img_path', 'text', 'label', 'i2i_id_list', 't2t_id_list', 
                           'prompt_image_path', 'prompt_text_path', 'i2i_label_list', 
                           't2t_label_list', 'observed_image', 'observed_text']
        missing_columns = [col for col in required_columns if col not in database.columns]
        if missing_columns:
            raise ValueError(f"Invalid database. It is missing required columns: {missing_columns}")
        if not isinstance(max_text_len, int):
            raise ValueError(f"Invalid max_text_len. It must be an integer. A {type(max_text_len)} was passed.")
        if max_text_len <= 0:
            raise ValueError(f"Invalid max_text_len. It must be positive. {max_text_len} was passed.")

        super().__init__()

        self.max_text_len = max_text_len
        self.img_path_list = database['img_path'].tolist()
        self.text_list = database['text'].tolist()
        self.label_list = database['label'].tolist()
        self.i2i_list = database['i2i_id_list'].tolist()
        self.t2t_list = database['t2t_id_list'].tolist()
        self.prompt_image_path = database['prompt_image_path'].tolist()
        self.prompt_text_path = database['prompt_text_path'].tolist()
        self.i2i_r_l_list_list = database['i2i_label_list'].tolist()
        self.t2t_r_l_list_list = database['t2t_label_list'].tolist()
        self.observed_image = database['observed_image'].tolist()
        self.observed_text = database['observed_text'].tolist()


    def __getitem__(self, index):
        text = self.text_list[index]
        image = self.img_path_list[index]
        image = Image.open(image) if pd.notna(image) else Image.new("RGBA", (256, 256), (0, 0, 0))
        image = image.convert("RGB")
        label = self.label_list[index]
        observed_text = self.observed_text[index]
        observed_image = self.observed_image[index]
        prompt_image_path = self.prompt_image_path[index]
        prompt_text_path = self.prompt_text_path[index]
        r_i_list = []
        r_t_list = []

        if (observed_text == 0) and (observed_image == 1):
            text = "I love deep learning" * 1024
            r_l_list = self.i2i_r_l_list_list[index]
            for i in range(len(prompt_image_path)):
                base = prompt_image_path[i]
                r_i_list.append(np.load(base).tolist())
                base= Path(*[("text" if p == "image" else p) for p in Path(base).parts])
                r_t_list.append(np.load(base).tolist())

        elif (observed_text == 1) and (observed_image == 0):
            r_l_list = self.t2t_r_l_list_list[index]
            for i in range(len(prompt_text_path)):
                base = prompt_text_path[i]
                r_t_list.append(np.load(base).tolist())
                base= Path(*[("image" if p == "text" else p) for p in Path(base).parts])
                r_i_list.append(np.load(base).tolist())

        elif (observed_text == 1) and (observed_image == 1):
            r_l_list = self.i2i_r_l_list_list[index]
            for prompt_image,prompt_text in zip(prompt_image_path, prompt_text_path):
                r_i_list.append(np.load(prompt_image).tolist())
                r_t_list.append(np.load(prompt_text).tolist())
        else:
            raise ValueError(f"No available modalities for item: {index}")

        return {
            "image": image,
            "text": text,
            "label": label,
            "r_t_list": r_t_list,
            "r_i_list": r_i_list,
            "r_l_list": r_l_list,
            "observed_text": observed_text,
            "observed_image": observed_image
        }


    def __len__(self):
        return len(self.label_list)


class RAGPTCollator():


    def __init__(self, tokenizer = None, image_processor = None,
                 max_text_len: int = 128):
        if not deepmodule_installed:
            raise ImportError(deepmodule_error)

        if tokenizer is not None and not isinstance(tokenizer, BertTokenizer):
            raise ValueError(f"Invalid tokenizer. It must be a BertTokenizer. A {type(tokenizer)} was passed.")
        if image_processor is not None and not isinstance(image_processor, ViltImageProcessor):
            raise ValueError(f"Invalid image_processor. It must be a ViltImageProcessor. A {type(image_processor)} was passed.")
        if not isinstance(max_text_len, int):
            raise ValueError(f"Invalid max_text_len. It must be an integer. A {type(max_text_len)} was passed.")
        if max_text_len <= 0:
            raise ValueError(f"Invalid max_text_len. It must be positive. {max_text_len} was passed.")

        if tokenizer is None:
            tokenizer = BertTokenizer.from_pretrained('dandelin/vilt-b32-mlm', do_lower_case=True)
        if image_processor is None:
            image_processor = ViltImageProcessor.from_pretrained('dandelin/vilt-b32-mlm')

        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_text_len = max_text_len


    def __call__(self, batch):
        text = [item['text'] for item in batch]
        image = [item['image'] for item in batch]
        label = [item['label'] for item in batch]
        r_t_list = [item['r_t_list'] for item in batch]
        r_i_list = [item['r_i_list'] for item in batch]
        observed_text = [item['observed_text'] for item in batch]
        observed_image = [item['observed_image'] for item in batch]
        r_l_list = [item['r_l_list'] for item in batch]
        text_encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_special_tokens_mask=True,
        )
        input_ids = text_encoding['input_ids']
        attention_mask = text_encoding['attention_mask']
        token_type_ids = text_encoding['token_type_ids']
        image = [resize_image(img) for img in image]
        image_encoding = self.image_processor(image, return_tensors="pt")
        pixel_values = image_encoding["pixel_values"]
        pixel_mask = image_encoding["pixel_mask"]
        input_ids = torch.tensor(input_ids,dtype=torch.int64)
        token_type_ids = torch.tensor(token_type_ids,dtype=torch.int64)
        attention_mask = torch.tensor(attention_mask,dtype=torch.int64)
        label = torch.tensor(label,dtype=torch.float)
        r_l_list = torch.tensor(r_l_list,dtype=torch.long)
        r_t_list = torch.tensor(r_t_list,dtype=torch.float)
        r_i_list = torch.tensor(r_i_list,dtype=torch.float)
        return {
            "input_ids": torch.tensor(input_ids,dtype=torch.int64),
            "pixel_values": pixel_values,
            "pixel_mask": pixel_mask,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "label": label,
            "r_t_list": r_t_list,
            "r_i_list": r_i_list,
            "r_l_list": r_l_list,
            "observed_image": torch.tensor(observed_image,dtype=torch.int64),
            "observed_text": torch.tensor(observed_text,dtype=torch.int64)
        }
