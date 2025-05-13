import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig


class DownStreamer(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DownStreamer, self).__init__()
        self.fc1 = nn.Linear(output_size, hidden_size)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        out = self.fc2(x)
        return out
    
    
class TiSpell_RoBERTa(nn.Module):
    def __init__(self, model_name, tokenizer):
        super(TiSpell_RoBERTa, self).__init__()
        self.roberta = AutoModelForCausalLM.from_pretrained(model_name, output_attentions=True)
        self.vocab_size = len(tokenizer)
        self.roberta.resize_token_embeddings(self.vocab_size)
        self.config = self.roberta.config
        self.character_corrector = DownStreamer(self.vocab_size, self.vocab_size)
        self.syllable_corrector = DownStreamer(self.vocab_size, self.vocab_size)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.roberta(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_states = outputs.logits
        logit_c = self.character_corrector(hidden_states)
        logit_s = self.syllable_corrector(hidden_states)
        logit = logit_c + logit_s
        return logit, logit_c
    
class TiSpell_RoBERTa_wo_Res(nn.Module):
    def __init__(self, model_name, tokenizer):
        super(TiSpell_RoBERTa_wo_Res, self).__init__()
        self.roberta = AutoModelForCausalLM.from_pretrained(model_name, output_attentions=True)
        self.vocab_size = len(tokenizer)
        self.roberta.resize_token_embeddings(self.vocab_size)
        self.config = self.roberta.config
        self.character_corrector = DownStreamer(self.vocab_size, self.vocab_size)
        self.syllable_corrector = DownStreamer(self.vocab_size, self.vocab_size)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.roberta(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_states = outputs.logits
        logit_c = self.character_corrector(hidden_states)
        logit_s = self.syllable_corrector(hidden_states)
        logit = logit_s
        return logit, logit_c
    
class TiSpell_RoBERTa_wo_Pretrain(nn.Module):
    def __init__(self, model_name, tokenizer):
        super(TiSpell_RoBERTa_wo_Pretrain, self).__init__()
        # self.roberta = AutoModelForCausalLM.from_pretrained(model_name, output_attentions=True)
        config = AutoConfig.from_pretrained(model_name)
        self.roberta = AutoModelForCausalLM.from_config(config)
        self.vocab_size = len(tokenizer)
        self.roberta.resize_token_embeddings(self.vocab_size)
        self.config = self.roberta.config
        self.character_corrector = DownStreamer(self.vocab_size, self.vocab_size)
        self.syllable_corrector = DownStreamer(self.vocab_size, self.vocab_size)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.roberta(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_states = outputs.logits
        logit_c = self.character_corrector(hidden_states)
        logit_s = self.syllable_corrector(hidden_states)
        logit = logit_c + logit_s
        return logit, logit_c
    
class TiSpell_RoBERTa_FC1(nn.Module):
    def __init__(self, model_name, tokenizer):
        super(TiSpell_RoBERTa_FC1, self).__init__()
        self.roberta = AutoModelForCausalLM.from_pretrained(model_name)
        self.roberta.resize_token_embeddings(len(tokenizer))
        self.vocab_size = len(tokenizer)
        self.character_corrector = nn.Linear(self.vocab_size, self.vocab_size)
        self.syllable_corrector = nn.Linear(self.vocab_size, self.vocab_size)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.roberta(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_states = outputs.logits
        logit_c = self.character_corrector(hidden_states)
        logit_s = self.syllable_corrector(hidden_states)
        logit = logit_c + logit_s
        return logit, logit_c
    