# License: BSD-3-Clause

from ... import deepmodule_installed, Module

if deepmodule_installed:
    from transformers import AutoModel, AutoTokenizer


class TextEncoder(Module):
    def __init__(self, bert_type="emilyalsentzer/Bio_ClinicalBERT") -> None:
        super().__init__()
        self.bert_type = bert_type
        self.model = AutoModel.from_pretrained(self.bert_type, output_hidden_states=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_type)

    def forward(self, text):
        text_tokenized = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256)
        embeddings = self.model(**text_tokenized).pooler_output
        return embeddings
