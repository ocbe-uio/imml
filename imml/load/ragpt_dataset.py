# License: BSD-3-Clause
import os
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

from .. import deepmodule_installed, deepmodule_error, Dataset, Module, ViltModel, BertTokenizer, ViltImageProcessor
from ..utils import check_Xs_y

if deepmodule_installed:
    from torch.nn import functional as F
    from transformers import AutoModel, AutoProcessor
    import torch


class RAGPTDataset(Dataset):
    r"""
    This class provides a `torch.utils.data.Dataset` implementation for handling multi-modal datasets with `RAGPT`. If 
    it is used with `torch.utils.data.DataLoader`, the `collate_fn` argument of the `DataLoader` constructor should be  
    `RAGPTCollator`.

    Parameters
    ----------
    batch_size : int, default=64
        Batch size used for encoding inputs during memory bank creation and inference.
    n_neighbors : int, default=20
        Number of neighbors to retrieve per sample during prediction.
    device : str, default="cpu"
        Device to use for model inference, typically "cpu" or "cuda".
    modalities : list of str, default=None
        Names of the modalities. Options are "text" and "image".
    pretrained_model : transformers.PreTrainedModel, default=None
        A pretrained HuggingFace model used for encoding multimodal inputs (e.g., CLIP model).
        If None, defaults to "openai/clip-vit-large-patch14-336".
    processor : transformers.ProcessorMixin, default=None
        HuggingFace processor corresponding to the pretrained model. Used to preprocess image/text inputs.
        If None, defaults to processor for "openai/clip-vit-large-patch14-336".
    prompt_path : str, default=None
        Path to save or load the generated prompts when `generate_cap` is True.
    pretrained_vilt : transformers.PreTrainedModel, default=None
        Pretrained model used for vision-language prompt generation. If None, defaults to
        ViltModel.from_pretrained('dandelin/vilt-b32-mlm').
    tokenizer : transformers.BertTokenizer, default=None
        Tokenizer used for text processing. If None, defaults to
        BertTokenizer.from_pretrained('dandelin/vilt-b32-mlm', do_lower_case=True).
    image_processor : transformers.ViltImageProcessor, default=None
        Image processor used with the ViLT model for image preprocessing. If None, defaults to
        ViltImageProcessor.from_pretrained('dandelin/vilt-b32-mlm').
    max_text_len : int, default=40
        Maximum token length for text inputs (used during prompt generation). Must not exceed the
        max_position_embeddings of the ViLT model (default: 40 for 'dandelin/vilt-b32-mlm').
    max_image_len : int, default=145
        Maximum token length for image inputs (used during prompt generation).

    Attributes
    ----------
    - mcr_: Multi-Channel Retriever (MCR) model for image-text retrieval.
    - input_ids_list_: Unique identifier for each sample.
    - img_path_list_: Path to the image files.
    - attention_mask_list_: Attention masks for texts.
    - token_type_ids_list_: Token ids for texts.
    - i2i_r_l_list_: List of labels of the retrieved items for the image-to-image modality.
    - t2t_r_l_list_: List of labels of the retrieved items for the text-to-text modality.
    - label_list_: Label of the sample.
    - prompt_image_path_: Path to the generated image prompt.
    - prompt_text_path_: Path to the generated text prompt.
    - observed_image_: Indicator of whether the image was observed.
    - observed_text_: Indicator of whether the text was observed.

    See Also
    --------
    :class:`~imml.classify.RAGPT`

    Example
    --------
    >>> from torch.utils.data import DataLoader
    >>> from imml.load import RAGPTDataset
    >>> Xs = [
            pd.DataFrame(["docs/figures/graph.png", "docs/figures/logo_imml.png"]),
            pd.DataFrame(["This is the graphical abstract of iMML.", "This is the logo of iMML."]),
        ]
    >>> y = [0, 1, 0, 1]
    >>> modalities = ["image", "text"]
    >>> tmp_path = tempfile.mkdtemp()
    >>> train_data = RAGPTDataset(Xs=Xs, y=y, Xs_bank=Xs, y_bank=y, modalities=modalities,
                                  n_neighbors=1, prompt_path=str(tmp_path))
    """


    def __init__(self, Xs: list, y, mcr = None, Xs_bank: list = None, y_bank = None,
                 batch_size: int = 64, n_neighbors: int = 20, device: str = "cpu", modalities: list = None,
                 pretrained_model = None, processor = None, prompt_path: str = None,
                 pretrained_vilt = None, tokenizer = None, image_processor = None,
                 max_text_len: int = 40, max_image_len: int = 145):
        if not deepmodule_installed:
            raise ImportError(deepmodule_error)

        if not isinstance(batch_size, int):
            raise ValueError(f"Invalid batch_size. It must be a integer. A {type(batch_size)} was passed.")
        if batch_size <= 0:
            raise ValueError(f"Invalid batch_size. It must be positive. A {type(batch_size)} was passed.")
        if not isinstance(n_neighbors, int):
            raise ValueError(f"Invalid n_neighbors. It must be a integer. A {type(n_neighbors)} was passed.")
        if n_neighbors <= 0:
            raise ValueError(f"Invalid n_neighbors. It must be positive. A {type(n_neighbors)} was passed.")
        if not isinstance(modalities, list):
            raise ValueError(f"Invalid modalities. It must be a list. A {type(modalities)} was passed.")
        if any([not isinstance(mod, str) for mod in modalities]):
            raise ValueError(f"Invalid modalities. It must be a list of str. A list of {[type(mod) for mod in modalities]} was passed.")
        if not isinstance(device, str):
            raise ValueError(f"Invalid device. It must be a string. A {type(device)} was passed.")
        if isinstance(prompt_path, int):
            raise ValueError("Invalid prompt_path. prompt_path must be a path.")
        try:
            if not os.path.exists(prompt_path):
                raise ValueError("Invalid prompt_path. prompt_path must exit.")
        except TypeError as e:
            raise TypeError(f"Invalid prompt_path. {e}")
        if not isinstance(max_text_len, int):
            raise ValueError(f"Invalid max_text_len. It must be a integer. A {type(max_text_len)} was passed.")
        if max_text_len <= 0:
            raise ValueError(f"Invalid max_text_len. It must be positive. {max_text_len} was passed.")
        if not isinstance(max_image_len, int):
            raise ValueError(f"Invalid max_image_len. It must be a integer. A {type(max_image_len)} was passed.")
        if max_image_len <= 0:
            raise ValueError(f"Invalid max_image_len. It must be positive. {max_image_len} was passed.")
        if mcr is None and (Xs_bank is None):
            raise ValueError("Invalid Xs_bank. Xs_bank must be provided if mcr is None.")
        if mcr is None and (y_bank is None):
            raise ValueError("Invalid y_bank. y_bank must be provided if mcr is None.")

        super().__init__()

        if mcr is None:
            mcr = _MCR(batch_size=batch_size, n_neighbors=n_neighbors, device=device, modalities=modalities,
                       pretrained_model=pretrained_model, processor=processor, prompt_path=prompt_path,
                       pretrained_vilt=pretrained_vilt, tokenizer=tokenizer, image_processor=image_processor,
                       max_text_len=max_text_len, max_image_len=max_image_len)

            mcr.fit(Xs=Xs_bank, y=y_bank)

        database = mcr.transform(Xs=Xs, y=y)

        database.loc[(1 - database['observed_text']).astype(bool), 'text'] = "I love deep learning" * 1024
        input_ids_list = []
        attention_mask_list = []
        token_type_ids_list = []
        for start in range(0, len(database), batch_size):
            chunk = database.iloc[start:start + batch_size]
            text_encoding = mcr.tokenizer(
                chunk['text'].to_list(),
                padding="max_length",
                truncation=True,
                max_length=max_text_len,
                return_special_tokens_mask=True,
            )
            input_ids = text_encoding['input_ids']
            attention_mask = text_encoding['attention_mask']
            token_type_ids = text_encoding['token_type_ids']

            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            token_type_ids_list.append(token_type_ids)
        input_ids_list = np.concatenate(input_ids_list, axis=0)
        attention_mask_list = np.concatenate(attention_mask_list, axis=0)
        token_type_ids_list = np.concatenate(token_type_ids_list, axis=0)

        self.mcr_ = mcr
        self.img_path_list_ = database['img_path'].tolist()
        self.input_ids_list_ = input_ids_list
        self.attention_mask_list_ = attention_mask_list
        self.token_type_ids_list_ = token_type_ids_list
        self.label_list_ = database['label'].tolist()
        self.prompt_image_path_ = database['prompt_image_path'].tolist()
        self.prompt_text_path_ = database['prompt_text_path'].tolist()
        self.i2i_r_l_list_ = database['i2i_label_list'].tolist()
        self.t2t_r_l_list_ = database['t2t_label_list'].tolist()
        self.observed_image_ = database['observed_image'].tolist()
        self.observed_text_ = database['observed_text'].tolist()


    def __getitem__(self, index):
        image = self.img_path_list_[index]
        image = Image.open(image) if pd.notna(image) else Image.new("RGBA", (256, 256), (0, 0, 0))
        image = image.convert("RGB")
        image = self.mcr_._resize_image(image)
        label = self.label_list_[index]
        input_ids = self.input_ids_list_[index]
        attention_mask = self.attention_mask_list_[index]
        token_type_ids = self.token_type_ids_list_[index]
        observed_text = self.observed_text_[index]
        observed_image = self.observed_image_[index]
        prompt_image_path = self.prompt_image_path_[index]
        prompt_text_path = self.prompt_text_path_[index]
        r_i_list = []
        r_t_list = []

        if (observed_text == 0) and (observed_image == 1):
            r_l_list = self.i2i_r_l_list_[index]
            for i in range(len(prompt_image_path)):
                base = prompt_image_path[i]
                r_i_list.append(np.load(base).tolist())
                base= Path(*[("text" if p == "image" else p) for p in Path(base).parts])
                r_t_list.append(np.load(base).tolist())

        elif (observed_text == 1) and (observed_image == 0):
            r_l_list = self.t2t_r_l_list_[index]
            for i in range(len(prompt_text_path)):
                base = prompt_text_path[i]
                r_t_list.append(np.load(base).tolist())
                base= Path(*[("image" if p == "text" else p) for p in Path(base).parts])
                r_i_list.append(np.load(base).tolist())

        elif (observed_text == 1) and (observed_image == 1):
            r_l_list = self.i2i_r_l_list_[index]
            for prompt_image,prompt_text in zip(prompt_image_path, prompt_text_path):
                r_i_list.append(np.load(prompt_image).tolist())
                r_t_list.append(np.load(prompt_text).tolist())
        else:
            raise ValueError(f"No available modalities for item: {index}")

        output = {
            "image": torch.from_numpy(np.array(image)),
            "input_ids": torch.from_numpy(input_ids),
            "attention_mask": torch.from_numpy(attention_mask),
            "token_type_ids": torch.from_numpy(token_type_ids),
            "label": torch.tensor(label),
            "r_t_list": torch.tensor(r_t_list),
            "r_i_list": torch.tensor(r_i_list),
            "r_l_list": torch.tensor(r_l_list),
            "observed_text": torch.tensor(observed_text),
            "observed_image": torch.tensor(observed_image)
        }
        return output


    def __len__(self):
        return len(self.label_list_)


class _MCR(Module):
    r"""
    Multi-Channel Retriever (MCR). [#ragptpaper]_ [#ragptcode]_

    MCR is a multimodal retrieval framework that enables similarity-based matching within modalities,
    even under missing modality settings. It builds a memory bank of multimodal embeddings and supports
    retrieval-augmented prompt generation for tasks like classification or generation.

    Parameters
    ----------
    batch_size : int, default=64
        Batch size used for encoding inputs during memory bank creation and inference.
    n_neighbors : int, default=20
        Number of neighbors to retrieve per sample during prediction.
    device : str, default="cpu"
        Device to use for model inference, typically "cpu" or "cuda".
    modalities : list of str, default=None
        Names of the modalities. Options are "text" and "image".
    pretrained_model : transformers.PreTrainedModel, default=None
        A pretrained HuggingFace model used for encoding multimodal inputs (e.g., CLIP model).
        If None, defaults to "openai/clip-vit-large-patch14-336".
    processor : transformers.ProcessorMixin, default=None
        HuggingFace processor corresponding to the pretrained model. Used to preprocess image/text inputs.
        If None, defaults to processor for "openai/clip-vit-large-patch14-336".
    generate_cap : bool, default=False
        Whether to generate retrieval-based prompts.
    prompt_path : str, default=None
        Path to save or load the generated prompts when `generate_cap` is True.
    pretrained_vilt : transformers.PreTrainedModel, default=None
        Pretrained model used for vision-language prompt generation. If None, defaults to
        ViltModel.from_pretrained('dandelin/vilt-b32-mlm').
    tokenizer : transformers.BertTokenizer, default=None
        Tokenizer used for text processing. If None, defaults to
        BertTokenizer.from_pretrained('dandelin/vilt-b32-mlm', do_lower_case=True).
    image_processor : transformers.ViltImageProcessor, default=None
        Image processor used with the ViLT model for image preprocessing. If None, defaults to
        ViltImageProcessor.from_pretrained('dandelin/vilt-b32-mlm').
    max_text_len : int, default=40
        Maximum token length for text inputs (used during prompt generation). Must not exceed the
        max_position_embeddings of the ViLT model (default: 40 for 'dandelin/vilt-b32-mlm').
    max_image_len : int, default=145
        Maximum token length for image inputs (used during prompt generation).
    save_memory_bank : bool, default=True
        Whether to save the memory bank of embeddings after fitting as an attribute. If False, the memory bank is
        returned as output during fit.

    Attributes
    ----------
    memory_bank_ : pd.DataFrame (n_samples, 6)
        DataFrame storing encoded modality representations for retrieval. Only if save_memory_bank is True. The columns
        are:
        - item_id: Unique identifier for each sample.
        - img_path: Path to the image file.
        - text: Textual content of the sample.
        - q_i: Image embedding.
        - q_t: Text embedding.
        - label: Label of the sample.
        - prompt_image_path: Path to the generated image prompt. Only if generate_cap is True.
        - prompt_text_path: Path to the generated text prompt. Only if generate_cap is True.

    References
    ----------
    .. [#ragptpaper] Lang, J., Z. Cheng, T. Zhong, and F. Zhou. “Retrieval-Augmented Dynamic Prompt Tuning for
                     Incomplete Multimodal Learning”. Proceedings of the AAAI Conference on Artificial Intelligence,
                     vol. 39, no. 17, Apr. 2025, pp. 18035-43, doi:10.1609/aaai.v39i17.33984.
    .. [#ragptcode] https://github.com/Jian-Lang/RAGPT/

    See Also
    --------
    :class:`~imml.classify.RAGPT`
    :class:`~imml.load.RAGPTDataset`

    Example
    --------
    >>> import pandas as pd
    >>> Xs = [
            pd.DataFrame(["docs/figures/graph.png", "docs/figures/logo_imml.png",
                          "docs/figures/graph.png", "docs/figures/logo_imml.png"]),
            pd.DataFrame(["This is the graphical abstract of iMML.", "This is the logo of iMML.",
                          "This is the graphical abstract of iMML.", "This is the logo of iMML."]),
    >>> ]
    >>> y = [0, 1, 0, 1]
    >>> modalities = ["image", "text"]
    >>> estimator = MCR(modalities=modalities, n_neighbors=1)
    >>> estimator.fit(Xs=Xs, y=y)
    >>> memory_bank = estimator.memory_bank_
    >>> estimator.predict(Xs=Xs)
    """


    def __init__(self, batch_size: int = 64, n_neighbors: int = 20, device: str = "cpu",
                 modalities: list = None, pretrained_model = None, processor = None,
                 prompt_path: str = None, pretrained_vilt = None,
                 tokenizer = None, image_processor = None,
                 max_text_len: int = 40, max_image_len: int = 145):

        if not deepmodule_installed:
            raise ImportError(deepmodule_error)

        super().__init__()

        if pretrained_model is None:
            pretrained_model = AutoModel.from_pretrained("openai/clip-vit-large-patch14-336").to(device)
        if processor is None:
            processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
        if pretrained_vilt is None:
            pretrained_vilt = ViltModel.from_pretrained('dandelin/vilt-b32-mlm')
        embedding_layer = pretrained_vilt.embeddings
        if tokenizer is None:
            tokenizer = BertTokenizer.from_pretrained('dandelin/vilt-b32-mlm',
                                                      do_lower_case=True)
        if image_processor is None:
            image_processor = ViltImageProcessor.from_pretrained('dandelin/vilt-b32-mlm')

        self.modalities = modalities
        self.batch_size = batch_size
        self.n_neighbors = n_neighbors
        self.device = device
        self.pretrained_model = pretrained_model
        self.processor = processor
        self.prompt_path = prompt_path
        self.pretrained_vilt = pretrained_vilt
        self.embedding_layer = embedding_layer
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_text_len = max_text_len
        self.max_image_len = max_image_len

        for param in self.embedding_layer.parameters():
            param.requires_grad = False



    def fit(self, Xs: list, y):
        r"""
        Fit the transformer to the input data.

        Parameters
        ----------
        Xs : list of array-likes objects
            - Xs length: 2
            - Xs[i] shape: (n_samples_i, 1)

            A list with images and texts.
        y : array-like of shape (n_samples,)
            Target vector relative to X.

        Returns
        -------
        self : Fitted retriever (or memory_bank if save_memory_bank=False)
        """
        Xs = check_Xs_y(Xs=Xs, y=y, supervised=True, modalities=self.modalities, mod_types=["image", "text"])

        Xs = self._convert_to_1dlist(Xs=Xs)
        q_i_list, q_t_list = self._encode_img_text(Xs=Xs)
        memory_bank = pd.DataFrame({
            'item_id': list(range(len(Xs[0]))),
            'img_path': Xs[self.modalities.index("image")],
            'text': Xs[self.modalities.index("text")],
            'q_i': q_i_list,
            'q_t': q_t_list,
            'label': y,
        })
        memory_bank = memory_bank.reset_index(drop=True)
        self._cap(memory_bank=memory_bank)

        if isinstance(y, pd.Series):
            memory_bank["item_id"] = y.index
            memory_bank.index = y.index
        self.memory_bank_ = memory_bank
        return self


    def transform(self, Xs: list, y = None):
        r"""
        Generate retrieval-augmented prompts.

        Parameters
        ----------
        Xs : list of array-likes objects
            - Xs length: 2
            - Xs[i] shape: (n_samples_i, 1)

            A list with images and texts.
        y : array-like of shape (n_samples,)
            Target vector relative to X.
        memory_bank : pd.DataFrame (n_samples, 10)
            Memory bank generated during fit. If None, the memory bank stored in the estimator is used.
        n_neighbors : int, default=None
            Number of neighbors to retrieve per sample during prediction. If None, the value stored in the estimator
            is used.

        Returns
        -------
        database : pd.DataFrame (n_samples, 14)
            A database with the retrieval-augmented prompts. It contains the following columns:
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

        """
        Xs = check_Xs_y(Xs=Xs, y=y, supervised=True, modalities=self.modalities, mod_types=["image", "text"])
        Xs = self._convert_to_1dlist(Xs=Xs)
        output = self.predict(Xs=Xs)
        database = pd.DataFrame({
            'item_id': list(range(len(Xs[0]))),
            'img_path': Xs[self.modalities.index("image")],
            'text': Xs[self.modalities.index("text")],
            'label': y,
        })
        if isinstance(Xs[0], pd.DataFrame):
            database["item_id"] = Xs[0].index
            database.index = Xs[0].index
        elif isinstance(y, pd.Series):
            database["item_id"] = y.index
            database.index = y.index
        observed_mod_indicator = pd.DataFrame({f"observed_{mod}": pd.notna(X).tolist()
                                               for X, mod in zip(Xs, self.modalities)}, index=database.index)
        database = pd.concat([database, observed_mod_indicator.astype(int)], axis=1)
        for mod in self.modalities:
            key = mod[0]
            key = f"{key}2{key}"
            for obj in list(output[mod].keys()):
                final_key = f"{key}_{obj}_list"
                database[final_key] = output[mod][obj]
            database[f"prompt_{mod}_path"] = [self.memory_bank_.loc[id, f"prompt_{mod}_path"].to_list() for id in output[mod]["id"]]
        return database


    def predict(self, Xs: list):
        r"""
        Retrieve the most similar instances.

        Parameters
        ----------
        Xs : list of array-likes objects
            - Xs length: 2
            - Xs[i] shape: (n_samples_i, 1)

            A list with images and texts.
        memory_bank : pd.DataFrame (n_samples, 10)
            Memory bank generated during fit. If None, the memory bank stored in the estimator is used.
        n_neighbors : int, default=None
            Number of neighbors to retrieve per sample during prediction. If None, the value stored in the estimator
            is used,

        Returns
        -------
        pred :  Dictionary with the ids, similarities and labels of the retrieved items for each modality.
        """
        memory_bank = self.memory_bank_
        Xs = self._convert_to_1dlist(Xs=Xs)
        q_i, q_t = self._encode_img_text(Xs=Xs)
        r_v_i = memory_bank['q_i'].tolist()
        r_v_t = memory_bank['q_t'].tolist()
        memory_bank_id = memory_bank['item_id'].tolist()
        memory_bank_label = memory_bank['label'].tolist()

        q_i = torch.tensor(q_i).squeeze(1).to(self.device)
        q_t = torch.tensor(q_t).squeeze(1).to(self.device)
        r_v_i = torch.tensor(r_v_i).squeeze(1).to(self.device)
        r_v_t = torch.tensor(r_v_t).squeeze(1).to(self.device)

        t2t_id_list, t2t_sims_list, t2t_label_list = self._compute_similarity_in_batches(q_t ,r_v_t,
                                                                                         memory_bank_id,
                                                                                         memory_bank_label)
        i2i_id_list, i2i_sims_list, i2i_label_list = self._compute_similarity_in_batches(q_i ,r_v_i,
                                                                                         memory_bank_id,
                                                                                         memory_bank_label)
        output = {
            "image": {
                "id": i2i_id_list,
                "sims": i2i_sims_list,
                "label": i2i_label_list,
            },
            "text": {
                "id": t2t_id_list,
                "sims": t2t_sims_list,
                "label": t2t_label_list,
            },
        }

        return output


    def _cap(self, memory_bank: pd.DataFrame = None):
        prompt_path = self.prompt_path

        for start in range(0, memory_bank.shape[0], self.batch_size):
            chunk = memory_bank.iloc[start:start + self.batch_size]
            texts = chunk['text']
            ids = chunk['img_path']

            text_encodings = self.tokenizer(
                texts.tolist(),
                padding="max_length",
                truncation=True,
                max_length=self.max_text_len,
                return_tensors="pt",
            )
            input_ids = text_encodings['input_ids']
            attention_mask = text_encodings['attention_mask']
            token_type_ids = text_encodings['token_type_ids']

            images = []
            for image_path in ids:
                image = Image.open(image_path).convert("RGB") \
                    if pd.notna(image_path) else Image.new("RGB", (256, 256), (0, 0, 0))
                image = self._resize_image(image)
                images.append(image)

            encoding_image_processor = self.image_processor(images, return_tensors="pt")
            pixel_values = encoding_image_processor["pixel_values"]
            pixel_mask = encoding_image_processor["pixel_mask"]

            emb = self._encode(input_ids, pixel_values, pixel_mask, token_type_ids, attention_mask)
            text_emb = emb[:, :self.max_text_len]
            image_emb = emb[:, self.max_text_len:]

            os.makedirs(os.path.join(prompt_path, "text"), exist_ok=True)
            os.makedirs(os.path.join(prompt_path, "image"), exist_ok=True)

            for i, id in enumerate(ids):
                idx = chunk.iloc[i].name
                file_name = os.path.basename(id).split(".")[0]
                file_path = os.path.join(prompt_path, "image", f"{file_name}.npy")
                memory_bank.loc[idx, "prompt_image_path"] = file_path
                np.save(file_path, image_emb[i].detach().numpy())
                file_path = os.path.join(prompt_path, "text", f"{file_name}.npy")
                memory_bank.loc[idx, "prompt_text_path"] = file_path
                np.save(file_path, text_emb[i].detach().numpy())


    def _encode_img_text(self, Xs: list):
        r"""
        Fit the transformer to the input data.

        Parameters
        ----------
        Xs : list of array-likes objects
            - Xs length: n_mods
            - Xs[i] shape: (n_samples, n_features_i)

            A list of different modalities.
        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self :  Fitted estimator.
        """

        q_i_list = []
        q_t_list = []
        for X,mod in zip(Xs, self.modalities):
            if mod == "image":
                batch_size = self.batch_size
                for i in range(0, len(X), batch_size):
                    batch_images = X[i: i + batch_size]
                    batch_images = [Image.open(img_name)
                                  if pd.notna(img_name) else Image.new("RGBA", (256, 256), (0, 0, 0))
                                  for img_name in batch_images]
                    batch_outputs = self._encode_image(batch_images)
                    q_i_list.extend(batch_outputs.cpu().tolist())
                q_i_list = [q_i if pd.notna(img) else [torch.nan for q in q_i] for img,q_i in zip(X, q_i_list)]
            elif mod == "text":
                q_t_list = [self._encode_text(text).tolist()
                            if pd.notna(text) else self._encode_text("").tolist()
                            for text in X]
                q_t_list = [q_t if pd.notna(text) else [torch.nan for q in q_t] for text,q_t in zip(X, q_t_list)]
            else:
                raise ValueError(f"Unknown modality type: {mod}")

        return q_i_list, q_t_list


    def _compute_similarity_in_batches(self, query_vectors, memory_bank, memory_bank_id, memory_bank_label):
        n_neighbors = self.n_neighbors
        r_id_list = []
        sims_list = []
        r_label_list = []
        for i in range(0, len(query_vectors), self.batch_size):
            batch = query_vectors[i: i +self.batch_size].unsqueeze(1)
            similarity = F.cosine_similarity(batch, memory_bank.unsqueeze(0), dim=-1)
            sim_scores, top_k_id = torch.topk(similarity, k=n_neighbors+1, dim=-1)
            for j in range(batch.size(0)):
                id_index = i + j
                if torch.isnan(sim_scores[j]).all():
                    retrieved_ids = []
                    sim_score = []
                    retrieved_labels = []
                else:
                    id = memory_bank_id[id_index] if id_index < len(memory_bank_id) else None
                    retrieved_ids = [memory_bank_id[idx] for idx in top_k_id[j].tolist() if memory_bank_id[idx] != id]
                    retrieved_labels = [memory_bank_label[idx] for idx in top_k_id[j].tolist() if memory_bank_id[idx] != id]
                    sim_score = sim_scores[j ,1:].tolist()
                    if len(retrieved_ids) > n_neighbors:
                        retrieved_ids = retrieved_ids[:n_neighbors]
                        sim_score = sim_score[:n_neighbors]
                        retrieved_labels = retrieved_labels[:n_neighbors]
                r_id_list.append(retrieved_ids)
                sims_list.append(sim_score)
                r_label_list.append(retrieved_labels)
        return  r_id_list ,sims_list, r_label_list


    def _encode_text(self, text):
        inputs = self.processor(text=text, return_tensors="pt", padding=True ,truncation=True)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        text_features = self.pretrained_model.text_model.embeddings(input_ids, attention_mask)
        text_features = self.pretrained_model.text_model.encoder(text_features).last_hidden_state
        text_features = self.pretrained_model.text_model.final_layer_norm(text_features)
        text_features = (self.pretrained_model.text_projection(text_features))
        return text_features[0, -1, :]


    def _encode_image(self, images):
        with torch.no_grad():
            processed_images = self.processor(images=images, return_tensors="pt").to(self.device)
            image_features = self.pretrained_model.vision_model.embeddings(processed_images['pixel_values'])
            image_features = self.pretrained_model.vision_model.pre_layrnorm(image_features)
            image_features = self.pretrained_model.vision_model.encoder(image_features).last_hidden_state
            image_features = self.pretrained_model.vision_model.post_layernorm(image_features)
            image_features = self.pretrained_model.visual_projection(image_features)
            return image_features[:, 0, :]


    def _encode(self, input_ids, pixel_values, pixel_mask, token_type_ids, attention_mask, image_token_type_idx=1):
        embedding, attention_mask = self.embedding_layer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=None,
            image_embeds=None,
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            image_token_type_idx=image_token_type_idx
        )
        return embedding


    def _resize_image(self, img, size=(384, 384)):
        return img.resize(size, Image.BILINEAR)


    def _convert_to_1dlist(self, Xs):
        if isinstance(Xs[0], pd.DataFrame):
            Xs = [X.squeeze().to_list() for X in Xs]
        elif isinstance(Xs[0], np.ndarray):
            Xs = [X.flatten().tolist() for X in Xs]
        elif isinstance(Xs[0], torch.Tensor):
            Xs = [X.numpy().flatten().tolist() for X in Xs]
        return Xs
