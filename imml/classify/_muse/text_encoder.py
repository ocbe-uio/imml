# License: BSD-3-Clause

try:
    import torch
    import torch.nn as nn
    from transformers import AutoModel, AutoTokenizer
    deepmodule_installed = True
except ImportError:
    deepmodule_installed = False
    deepmodule_error = "Module 'deep' needs to be installed. See https://imml.readthedocs.io/stable/main/installation.html#optional-dependencies"

nnModuleBase = nn.Module if deepmodule_installed else object


class TextEncoder(nnModuleBase):
    def __init__(self, bert_type="emilyalsentzer/Bio_ClinicalBERT", device="cpu") -> None:
        super().__init__()
        self.bert_type = bert_type
        self.model = AutoModel.from_pretrained(self.bert_type, output_hidden_states=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_type)
        self.device = device

    def forward(self, text):
        text_tokenized = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256)
        text_tokenized = text_tokenized.to(self.device)
        embeddings = self.model(**text_tokenized).pooler_output
        return embeddings
