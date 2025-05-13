import torch
import torch.nn as nn
from transformers import BertForMaskedLM


class BertWord(nn.Module):
    def __init__(self, args):
        super(BertWord, self).__init__()
        output_dim = args.word_vocab_size
        self.bert_model = BertForMaskedLM.from_pretrained("./tokenizer/tibetan_bert")
        self.bertmodule_outdim = self.bert_model.config.hidden_size

        self.bert_dropout = torch.nn.Dropout(0.2)
        # self.dropout = nn.Dropout(p=0.4)
        self.dense = nn.Linear(self.bertmodule_outdim, output_dim)
        
    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        bert_encodings = self.bert_model(input_ids, token_type_ids, attention_mask)
        bert_encodings = self.bert_dropout(bert_encodings)
        logits = self.dense(bert_encodings)
        return logits