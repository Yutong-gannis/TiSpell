import torch
import torch.nn as nn
from .bert import BertModel, BertEncoder, BertPooler, BertConfig

class SCBertWord(nn.Module):
    def __init__(self, args):
        super(SCBertWord, self).__init__()
        output_dim = args.word_vocab_size
        config = BertConfig(
            vocab_size=3 * args.char_vocab_size,
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            num_attention_heads=args.num_attention_heads,
            intermediate_size=args.intermediate_size,
            max_position_embeddings=args.max_position_embeddings,
            hidden_dropout_prob=args.dropout_prob,
            attention_probs_dropout_prob=args.dropout_prob,
            type_vocab_size=2,
            layer_norm_eps=1e-12,
            initializer_range=0.02
        )
        self.embedding = nn.Linear(3 * args.char_vocab_size, args.hidden_size)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.bert_dropout = torch.nn.Dropout(0.2)
        self.bert_model = BertModel(config=config)
        self.bertmodule_outdim = self.bert_model.config.hidden_size

        # self.dropout = nn.Dropout(p=0.4)
        self.dense = nn.Linear(self.bertmodule_outdim, output_dim)
        
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, output_all_encoded_layers=False):
        #bert_encodings, _ = self.bert_model(input_ids, token_type_ids, attention_mask, output_all_encoded_layers)
        #bert_encodings = self.bert_dropout(bert_encodings)
        #logits = self.dense(bert_encodings)
        #return logits
        #input_ids = self.embedding(input_ids)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        #embedding_output = self.embeddings(input_ids, token_type_ids)
        embedding_output = self.embedding(input_ids)
        encoded_layers = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers)
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        
        bert_encodings = self.bert_dropout(encoded_layers)
        logits = self.dense(bert_encodings)
        return logits