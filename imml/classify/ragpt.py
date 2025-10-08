# License: BSD-3-Clause

from ._ragpt import MMG, CAP
from ._ragpt.vilt import ViltModel

try:
    from torch import nn
    import torch
    import torch.nn.functional as F
    import lightning as L
    deepmodule_installed = True
except ImportError:
    deepmodule_installed = False
    deepmodule_error = "Module 'deep' needs to be installed. See https://imml.readthedocs.io/stable/main/installation.html#optional-dependencies"

LightningModuleBase = L.LightningModule if deepmodule_installed else object
ViltModel = ViltModel if deepmodule_installed else object
nnModuleBase = nn.Module if deepmodule_installed else object


class RAGPT(LightningModuleBase):
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
    max_text_len : int, default=128
        Maximum number of tokens for text inputs.
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
    cls_num : int, default=2
        Number of target classes for classification tasks.
    loss : callable, optional
        Loss function. If None, defaults to `F.cross_entropy`.
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

    Example
    --------
    >>> from imml.retrieve import MCR
    >>> from imml.load import RAGPTDataset, RAGPTCollator
    >>> from imml.classify import RAGPT
    >>> from lightning import Trainer
    >>> from torch.utils.data import DataLoader
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
    >>> train_dataloader = DataLoader(train_data, collate_fn=RAGPTCollator)
    >>> trainer = Trainer(max_epochs=2, logger=False, enable_checkpointing=False)
    >>> estimator = RAGPT()
    >>> trainer.fit(estimator, train_dataloader)
    >>> trainer.predict(estimator, train_dataloader)
    """


    def __init__(self, vilt: ViltModel = None, max_text_len: int = 128, max_image_len: int = 145,
                 prompt_position: int = 0, prompt_length: int = 1, dropout_rate: float = 0.2, hidden_dim: int = 768,
                 cls_num: int = 2, loss: callable = None, learning_rate: float = 1e-3,
                 weight_decay: float = 2e-2):

        if not deepmodule_installed:
            raise ImportError(deepmodule_error)

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
        if not isinstance(cls_num, int):
            raise ValueError(f"Invalid cls_num. It must be an integer. A {type(cls_num)} was passed.")
        if cls_num <= 0:
            raise ValueError(f"Invalid cls_num. It must be positive. {cls_num} was passed.")
        if loss is not None and not callable(loss):
            raise ValueError(f"Invalid loss. It must be callable. A {type(loss)} was passed.")
        if not isinstance(learning_rate, float):
            raise ValueError(f"Invalid learning_rate. It must be a float. A {type(learning_rate)} was passed.")
        if learning_rate <= 0:
            raise ValueError(f"Invalid learning_rate. It must be positive. {learning_rate} was passed.")
        if not isinstance(weight_decay, float):
            raise ValueError(f"Invalid weight_decay. It must be a float. A {type(weight_decay)} was passed.")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay. It must be non-negative. {weight_decay} was passed.")

        super().__init__()

        self.model = RAGPTModule(vilt=vilt, max_text_len=max_text_len, max_image_len=max_image_len,
                                prompt_position=prompt_position, prompt_length=prompt_length,
                                dropout_rate=dropout_rate, hidden_dim=hidden_dim, cls_num=cls_num)
        if loss is None:
            loss = F.cross_entropy
        self.loss = loss
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay


    def training_step(self, batch, batch_idx=None):
        r"""
        Method required for training using `Lightning Trainer <https://lightning.ai/docs/pytorch/stable/common/trainer.html>`_.
        """
        labels = batch.pop('label').long()
        preds = self.model(**batch)
        loss = self.loss(preds, labels)
        return loss


    def validation_step(self, batch, batch_idx=None):
        r"""
        Method required for validating using `Lightning Trainer <https://lightning.ai/docs/pytorch/stable/common/trainer.html>`_.
        """
        labels = batch.pop('label').long()
        preds = self.model(**batch)
        loss = self.loss(preds, labels)
        return loss


    def test_step(self, batch, batch_idx=None):
        r"""
        Method required for testing using `Lightning Trainer <https://lightning.ai/docs/pytorch/stable/common/trainer.html>`_.
        """
        labels = batch.pop('label').long()
        preds = self.model(**batch)
        loss = self.loss(preds, labels)
        return loss


    def predict_step(self, batch, batch_idx=None):
        r"""
        Method required for predicting using `Lightning Trainer <https://lightning.ai/docs/pytorch/stable/common/trainer.html>`_.
        """
        _ = batch.pop('label').long()
        preds = self.model(**batch)
        return preds


    def configure_optimizers(self):
        r"""
        Method required for training using `Lightning Trainer <https://lightning.ai/docs/pytorch/stable/common/trainer.html>`_.
        """
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer


class RAGPTModule(nnModuleBase):
    def __init__(self, vilt: ViltModel = None, max_text_len: int = 128, max_image_len: int = 145,
                 prompt_position: int = 0, prompt_length: int = 1, dropout_rate: float = 0.2,
                 hidden_dim: int = 768, cls_num: int = 2):

        if not deepmodule_installed:
            raise ImportError(deepmodule_error)

        super().__init__()

        if vilt is None:
            vilt = ViltModel.from_pretrained('dandelin/vilt-b32-mlm')

        self.max_text_len = max_text_len
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
        embedding, attention_mask = self.embedding_layer(input_ids=input_ids,
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
        t_observed_mask = torch.tensor(observed_text).to(pixel_values.device)
        i_observed_mask = torch.tensor(observed_image).to(pixel_values.device)
        observed_mask_t = t_observed_mask.view(-1, 1, 1).expand(-1,self.max_text_len, self.hs)
        observed_mask_i = i_observed_mask.view(-1, 1, 1).expand(-1, 145, self.hs)
        text_emb = text_emb * observed_mask_t + recovered_t * (~observed_mask_t)
        image_emb = image_emb * observed_mask_i + recovered_i * (~observed_mask_i)

        t_prompt,i_prompt = self.dynamic_prompt(r_i=r_i_list, r_t=r_t_list, T=text_emb, V=image_emb)
        t_prompt = torch.mean(t_prompt, dim=1)
        i_prompt = torch.mean(i_prompt, dim=1)

        label_emb = self.label_enhanced[r_l_list]
        label_cls = self.label_enhanced
        label_emb = torch.mean(label_emb, dim=1)
        label_emb = label_emb.view(-1, 1, self.hs)

        output = torch.cat([text_emb, image_emb], dim=1)
        for i, layer_module in enumerate(self.encoder_layer):
            if i == self.prompt_position:
                output = torch.cat([label_emb,t_prompt.unsqueeze(1),i_prompt.unsqueeze(1),output], dim=1)
                N = embedding.shape[0]
                attention_mask = torch.cat([torch.ones(N,1+self.prompt_length*2).to(pixel_values.device), attention_mask],
                                           dim=1)
                layer_outputs = layer_module(output, attention_mask=attention_mask)
                output = layer_outputs[0]
            else:
                layer_outputs = layer_module(output, attention_mask=attention_mask)
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
