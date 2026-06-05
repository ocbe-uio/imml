# License: BSD-3-Clause

from ._ragpt import MMG, CAP
from .. import deepmodule_installed, deepmodule_error, LightningModule, Module, ViltModel, ViltImageProcessor

if deepmodule_installed:
    from torch import nn
    import torch


class RAGPT(LightningModule):
    r"""

    Retrieval-AuGmented dynamic Prompt Tuning (RAGPT). [#ragptpaper]_ [#ragptcode]_

    RAGPT is designed for incomplete vision-language learning, where one modality may be missing at
    inference or training time. It combines three core modules to address this challenge: 1) Multi-Channel Retriever,
    which retrieves semantically similar instances from a training database, per modality; 2) Missing Modality
    Generator, which fills in missing modality data using context from retrieved neighbors; and 3) Context-Aware
    Prompter, which generates dynamic prompts based on context to improve downstream classification in
    multimodal transformers.

    This class provides training, validation, testing, and prediction logic compatible with the
    `Lightning Trainer <https://lightning.ai/docs/pytorch/stable/common/trainer.html>`_.

    Parameters
    ----------
    vilt : transformers.ViltModel, optional
        Pretrained model used for joint vision-language encoding. If None, defaults to
        ViltModel.from_pretrained('dandelin/vilt-b32-mlm').
    image_processor : transformers.ViltImageProcessor, default=None
        Image processor used with the ViLT model for image preprocessing. If None, defaults to
        ViltImageProcessor.from_pretrained('dandelin/vilt-b32-mlm').
    max_text_len : int, default=40
        Maximum token length for text inputs (used during prompt generation). Must not exceed the
        max_position_embeddings of the ViLT model (default: 40 for 'dandelin/vilt-b32-mlm').
    max_image_len : int, default=145
        Maximum number of image patches/tokens processed by the vision encoder.
    prompt_position : int, default=0
        Index position at which to insert dynamic prompts in the transformer input sequence.
    prompt_length : int, default=1
        Number of prompt tokens to insert for dynamic prompt tuning.
    dropout_rate : float, default=0.2
        Dropout probability.
    hidden_dim : int, default=768
        Hidden dimension size.
    output_dim : int, default=2
        Number of classes in your response variable. Typically 2 for binary classification.
    loss_fn : callable, default=None
        Loss function. If None, defaults to `nn.BCEWithLogitsLoss()` if output_dim == <=2, else `nn.CrossEntropyLoss()`.
    learning_rate : float, default=1e-3
        Learning rate for the optimizer.
    weight_decay : float, default=2e-2
        Weight decay used by the optimizer.

    References
    ----------
    .. [#ragptpaper] Lang, J., Z. Cheng, T. Zhong, and F. Zhou. “Retrieval-Augmented Dynamic Prompt Tuning for
                     Incomplete Multimodal Learning”. Proceedings of the AAAI Conference on Artificial Intelligence,
                     vol. 39, no. 17, Apr. 2025, pp. 18035-43, doi:10.1609/aaai.v39i17.33984.
    .. [#ragptcode] https://github.com/Jian-Lang/RAGPT/

    See Also
    --------
    :class:`~imml.load.RAGPTDataset`

    Example
    --------
    >>> from imml.load import RAGPTDataset
    >>> from imml.classify import RAGPT
    >>> from lightning import Trainer
    >>> from torch.utils.data import DataLoader
    >>> images = ["docs/figures/graph.png", "docs/figures/logo_imml.png",
                  "docs/figures/graph.png", "docs/figures/logo_imml.png"]
    >>> texts = ["This is the graphical abstract of iMML.", "This is the logo of iMML.",
                 "This is the graphical abstract of iMML.", "This is the logo of iMML."]
    >>> Xs = [
            pd.DataFrame(images),
            pd.DataFrame(texts),
        ]
    >>> y = [0, 1, 0, 1]
    >>> modalities = ["image", "text"]
    >>> tmp_path = tempfile.mkdtemp()
    >>> train_data = RAGPTDataset(Xs=Xs, y=y, Xs_bank=Xs, y_bank=y, modalities=modalities,
                                  n_neighbors=1, prompt_path=str(tmp_path))
    >>> train_dataloader = DataLoader(train_data, batch_size=len(train_data))
    >>> trainer = Trainer(max_epochs=2, logger=False, enable_checkpointing=False)
    >>> estimator = RAGPT()
    >>> trainer.fit(estimator, train_dataloader)
    >>> trainer.predict(estimator, train_dataloader)
    """


    def __init__(self, max_text_len: int = 40, max_image_len: int = 145,
                 vilt: ViltModel = None, image_processor : ViltImageProcessor = None,
                 prompt_position: int = 0, prompt_length: int = 1,
                 dropout_rate: float = 0.2, hidden_dim: int = 768, output_dim: int = 2,
                 loss_fn: callable = None, learning_rate: float = 1e-3, weight_decay: float = 2e-2):

        if not deepmodule_installed:
            raise ImportError(deepmodule_error)

        if (image_processor is not None) and (not isinstance(image_processor, ViltImageProcessor)):
            raise ValueError(f"Invalid image_processor. It must be a ViltImageProcessor. A {type(image_processor)} was passed.")
        if not isinstance(max_text_len, int):
            raise ValueError(f"Invalid max_text_len. It must be an integer. A {type(max_text_len)} was passed.")
        if max_text_len <= 0:
            raise ValueError(f"Invalid max_text_len. It must be positive. {max_text_len} was passed.")
        if not isinstance(max_image_len, int):
            raise ValueError(f"Invalid max_image_len. It must be an integer. A {type(max_image_len)} was passed.")
        if max_image_len <= 0:
            raise ValueError(f"Invalid max_image_len. It must be positive. {max_image_len} was passed.")
        if not isinstance(prompt_position, int):
            raise ValueError(f"Invalid prompt_position. It must be an integer. A {type(prompt_position)} was passed.")
        if prompt_position < 0:
            raise ValueError(f"Invalid prompt_position. It must be non-negative. {prompt_position} was passed.")
        if not isinstance(prompt_length, int):
            raise ValueError(f"Invalid prompt_length. It must be an integer. A {type(prompt_length)} was passed.")
        if prompt_length <= 0:
            raise ValueError(f"Invalid prompt_length. It must be positive. {prompt_length} was passed.")
        if not isinstance(dropout_rate, float):
            raise ValueError(f"Invalid dropout_rate. It must be a float. A {type(dropout_rate)} was passed.")
        if dropout_rate < 0 or dropout_rate > 1:
            raise ValueError(f"Invalid dropout_rate. It must be between 0 and 1. {dropout_rate} was passed.")
        if not isinstance(hidden_dim, int):
            raise ValueError(f"Invalid hidden_dim. It must be an integer. A {type(hidden_dim)} was passed.")
        if hidden_dim <= 0:
            raise ValueError(f"Invalid hidden_dim. It must be positive. {hidden_dim} was passed.")
        if not isinstance(output_dim, int):
            raise ValueError(f"Invalid output_dim. It must be an integer. A {type(output_dim)} was passed.")
        if output_dim <= 0:
            raise ValueError(f"Invalid output_dim. It must be positive. {output_dim} was passed.")
        if loss_fn is not None and not callable(loss_fn):
            raise ValueError(f"Invalid loss_fn. It must be callable. A {type(loss_fn)} was passed.")
        if not isinstance(learning_rate, float):
            raise ValueError(f"Invalid learning_rate. It must be a float. A {type(learning_rate)} was passed.")
        if learning_rate <= 0:
            raise ValueError(f"Invalid learning_rate. It must be positive. {learning_rate} was passed.")
        if not isinstance(weight_decay, float):
            raise ValueError(f"Invalid weight_decay. It must be a float. A {type(weight_decay)} was passed.")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay. It must be non-negative. {weight_decay} was passed.")

        super().__init__()

        if image_processor is None:
            image_processor = ViltImageProcessor.from_pretrained('dandelin/vilt-b32-mlm')

        self.model = RAGPTModule(vilt=vilt, max_text_len=max_text_len, max_image_len=max_image_len,
                                 prompt_position=prompt_position, prompt_length=prompt_length,
                                 image_processor=image_processor,
                                 dropout_rate=dropout_rate, hidden_dim=hidden_dim, output_dim=output_dim)
        if loss_fn is None:
            loss_fn = nn.BCEWithLogitsLoss() if output_dim == 1 else nn.CrossEntropyLoss()
        self.loss_fn = loss_fn
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        if output_dim == 1:
            self.get_probs = nn.Sigmoid()
        else:
            self.get_probs = nn.Softmax(dim=-1)


    def training_step(self, batch, batch_idx=None):
        r"""
        Method required for training using `Lightning Trainer <https://lightning.ai/docs/pytorch/stable/common/trainer.html>`_.
        """
        batch = self.model.collect(batch)
        labels = batch.pop('label')
        preds = self.model(**batch)
        loss = self.loss_fn(preds, labels)
        return loss


    def validation_step(self, batch, batch_idx=None):
        r"""
        Method required for validating using `Lightning Trainer <https://lightning.ai/docs/pytorch/stable/common/trainer.html>`_.
        """
        batch = self.model.collect(batch)
        labels = batch.pop('label')
        preds = self.model(**batch)
        loss = self.loss_fn(preds, labels)
        return loss


    def test_step(self, batch, batch_idx=None):
        r"""
        Method required for testing using `Lightning Trainer <https://lightning.ai/docs/pytorch/stable/common/trainer.html>`_.
        """
        batch = self.model.collect(batch)
        labels = batch.pop('label')
        preds = self.model(**batch)
        loss = self.loss_fn(preds, labels)
        return loss


    def predict_step(self, batch, batch_idx=None):
        r"""
        Method required for predicting using `Lightning Trainer <https://lightning.ai/docs/pytorch/stable/common/trainer.html>`_.
        """
        batch = self.model.collect(batch)
        _ = batch.pop('label')
        preds = self.model(**batch)
        preds = self.get_probs(preds)
        return preds


    def configure_optimizers(self):
        r"""
        Method required for training using `Lightning Trainer <https://lightning.ai/docs/pytorch/stable/common/trainer.html>`_.
        """
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer


class RAGPTModule(Module):
    def __init__(self, vilt: ViltModel = None, max_text_len: int = 40, max_image_len: int = 145,
                 prompt_position: int = 0, prompt_length: int = 1, dropout_rate: float = 0.2,
                 hidden_dim: int = 768, output_dim: int = 1, image_processor = None):

        if not deepmodule_installed:
            raise ImportError(deepmodule_error)

        super().__init__()

        if vilt is None:
            vilt = ViltModel.from_pretrained('dandelin/vilt-b32-mlm')

        self.max_text_len = max_text_len
        self.vilt = vilt  # Keep reference to vilt for get_extended_attention_mask
        self.image_processor = image_processor
        self.embedding_layer = vilt.embeddings
        self.encoder_layer = vilt.encoder.layer
        self.layernorm = vilt.layernorm
        self.prompt_length = prompt_length
        self.prompt_position = prompt_position
        self.hs = hidden_dim

        self.freeze()
        self.pooler = vilt.pooler
        self.MMG_t = MMG(n = max_text_len, d=hidden_dim, dropout_rate=dropout_rate)
        self.MMG_i = MMG(n = max_image_len, d=hidden_dim, dropout_rate=dropout_rate)
        self.dynamic_prompt = CAP(prompt_length=prompt_length)
        cls_num = 2 if output_dim <= 2 else output_dim
        self.label_enhanced = nn.Parameter(torch.randn(cls_num, hidden_dim))
        self.classifier = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim * 2),
                nn.LayerNorm(hidden_dim * 2),
                nn.GELU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
        self.classifier.apply(self.init_weights)

    def freeze(self):
        for param in self.embedding_layer.parameters():
            param.requires_grad = False
        for param in self.encoder_layer.parameters():
            param.requires_grad = False
        for param in self.layernorm.parameters():
            param.requires_grad = False

    def forward(self,
                input_ids,
                pixel_values,
                pixel_mask,
                token_type_ids,
                attention_mask,
                r_t_list,
                r_i_list,
                r_l_list,
                observed_image = None,
                observed_text = None,
                image_token_type_idx=1):
        embedding, base_attention_mask = self.embedding_layer(input_ids=input_ids,
                                                         attention_mask=attention_mask,
                                                         token_type_ids=token_type_ids,
                                                         inputs_embeds=None,
                                                         image_embeds=None,
                                                         pixel_values=pixel_values,
                                                         pixel_mask=pixel_mask,
                                                         image_token_type_idx=image_token_type_idx)
        text_emb = embedding[:, :self.max_text_len, :]
        image_emb = embedding[:, self.max_text_len:, :]

        recovered_t = self.MMG_t(r_t_list)
        recovered_i = self.MMG_i(r_i_list)
        t_observed_mask = torch.as_tensor(observed_text, dtype=torch.float32).to(pixel_values.device)
        i_observed_mask = torch.as_tensor(observed_image, dtype=torch.float32).to(pixel_values.device)
        observed_mask_t = t_observed_mask.view(-1, 1, 1).expand(-1, self.max_text_len, self.hs)
        observed_mask_i = i_observed_mask.view(-1, 1, 1).expand(-1, 145, self.hs)
        text_emb = text_emb * observed_mask_t + recovered_t * (1 - observed_mask_t)
        image_emb = image_emb * observed_mask_i + recovered_i * (1 - observed_mask_i)

        t_prompt, i_prompt = self.dynamic_prompt(r_i=r_i_list, r_t=r_t_list, T=text_emb, V=image_emb)

        label_emb = self.label_enhanced[r_l_list]
        label_cls = self.label_enhanced
        label_emb = torch.mean(label_emb, dim=1)
        label_emb = label_emb.view(-1, 1, self.hs)

        output = torch.cat([text_emb, image_emb], dim=1)
        N = embedding.shape[0]
        current_attention_mask = base_attention_mask
        extended_attention_mask = self.vilt.get_extended_attention_mask(
            current_attention_mask, output.shape[:2]
        )
        for i, layer_module in enumerate(self.encoder_layer):
            if i == self.prompt_position:
                output = torch.cat([label_emb, t_prompt, i_prompt, output], dim=1)
                prompt_mask = torch.ones(N, 1+self.prompt_length*2,
                                        dtype=current_attention_mask.dtype,
                                        device=pixel_values.device)
                current_attention_mask = torch.cat([prompt_mask, current_attention_mask], dim=1)
                extended_attention_mask = self.vilt.get_extended_attention_mask(
                    current_attention_mask, output.shape[:2]
                )
            layer_outputs = layer_module(output, attention_mask=extended_attention_mask)
            output = layer_outputs[0]
        output = self.layernorm(output)
        output = self.pooler(output)
        output = torch.cat([output,label_emb.squeeze(1)],dim=1)
        output = self.classifier(output)
        label_cls = label_cls.repeat(N, 1,1)
        label_cls = label_cls.transpose(-1,-2)
        output = output.unsqueeze(1)
        output = torch.matmul(output, label_cls)
        output = output.squeeze(1)
        return output


    @staticmethod
    def init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


    def collect(self, batch):
        image_encoding = self.image_processor(batch["image"], return_tensors="pt")
        batch["pixel_values"] = image_encoding["pixel_values"]
        batch["pixel_mask"] = image_encoding["pixel_mask"]
        batch = {key:value for key,value in batch.items() if key != "image"}
        return batch
