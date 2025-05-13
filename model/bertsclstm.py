import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.rnn import pad_sequence
from .bert import BertModel


class BertSCLSTM(nn.Module):
    def __init__(self, config, screp_dim, padding_idx, output_dim, early_concat=True,
                 bert_pretrained_name_or_path=None, freeze_bert=False):
        super(BertSCLSTM, self).__init__()

        self.bert_dropout = torch.nn.Dropout(0.2)
        self.bert_model = BertModel(config=config)
        self.bertmodule_outdim = self.bert_model.config.hidden_size
        self.early_concat = early_concat  # if True, (bert+sc)->lstm->linear, else ((sc->lstm)+bert)->linear
        if freeze_bert:
            # Uncomment to freeze BERT layers
            for param in self.bert_model.parameters():
                param.requires_grad = False

        # lstm module
        # expected  input dim: [BS,max_nwords,*] and batch_lengths as [BS] for pack_padded_sequence
        bidirectional, hidden_size, nlayers = True, 512, 2
        if self.early_concat:
            self.lstmmodule_indim = screp_dim + self.bertmodule_outdim
            self.lstmmodule = nn.LSTM(self.lstmmodule_indim, hidden_size, nlayers,
                                      batch_first=True, dropout=0.3, bidirectional=bidirectional)
            self.lstmmodule_outdim = hidden_size * 2 if bidirectional else hidden_size
            self.encodings_outdim = self.lstmmodule_outdim
        else:
            self.lstmmodule_indim = screp_dim
            self.lstmmodule = nn.LSTM(self.lstmmodule_indim, hidden_size, nlayers,
                                      batch_first=True, dropout=0.3, bidirectional=bidirectional)
            self.lstmmodule_outdim = hidden_size * 2 if bidirectional else hidden_size
            self.encodings_outdim = self.lstmmodule_outdim + self.bertmodule_outdim

        # output module
        assert output_dim > 0
        self.dropout = nn.Dropout(p=0.4)
        self.dense = nn.Linear(self.encodings_outdim, output_dim)

        # loss
        # See https://pytorch.org/docs/stable/nn.html#crossentropyloss
        self.criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=padding_idx)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def get_merged_encodings(self, bert_seq_encodings, seq_splits, mode='avg'):
        bert_seq_encodings = bert_seq_encodings[:sum(seq_splits) + 2, :]  # 2 for [CLS] and [SEP]
        bert_seq_encodings = bert_seq_encodings[1:-1, :]
        # a tuple of tensors
        split_encoding = torch.split(bert_seq_encodings, seq_splits, dim=0)
        batched_encodings = pad_sequence(split_encoding, batch_first=True, padding_value=0)
        if mode == 'avg':
            seq_splits = torch.tensor(seq_splits).reshape(-1, 1).to(self.device)
            out = torch.div(torch.sum(batched_encodings, dim=1), seq_splits)
        elif mode == "add":
            out = torch.sum(batched_encodings, dim=1)
        else:
            raise Exception("Not Implemented")
        return out

    def forward(self,
                batch_screps: list[pad_sequence],
                batch_lengths: torch.tensor,
                batch_bert_dict: "{'input_ids':torch.tensor, 'attention_mask':torch.tensor, 'token_type_ids':torch.tensor}",
                batch_splits: list[list[int]],
                aux_word_embs: torch.tensor = None,
                targets: torch.tensor = None,
                topk=1):

        if aux_word_embs is not None:
            raise Exception("dimensions of aux_word_embs not used in __init__()")

        # cnn
        batch_size = len(batch_screps)
        batch_screps = pad_sequence(batch_screps, batch_first=True, padding_value=0)

        # bert
        # BS X max_nsubwords x self.bertmodule_outdim
        bert_encodings = self.bert_model(**batch_bert_dict, return_dict=False)[0]
        bert_encodings = self.bert_dropout(bert_encodings)
        # BS X max_nwords x self.bertmodule_outdim
        bert_merged_encodings = pad_sequence(
            [self.get_merged_encodings(bert_seq_encodings, seq_splits, mode='avg') \
             for bert_seq_encodings, seq_splits in zip(bert_encodings, batch_splits)],
            batch_first=True,
            padding_value=0
        )

        if self.early_concat:

            # concat aux_embs
            # if not None, the expected dim for aux_word_embs: [BS,max_nwords,*]
            intermediate_encodings = torch.cat((batch_screps, bert_merged_encodings), dim=2)
            if aux_word_embs is not None:
                intermediate_encodings = torch.cat((intermediate_encodings, aux_word_embs), dim=2)

            # lstm
            # dim: [BS,max_nwords,*]->[BS,max_nwords,self.lstmmodule_outdim]
            intermediate_encodings = pack_padded_sequence(intermediate_encodings, batch_lengths,
                                                          batch_first=True, enforce_sorted=False)
            lstm_encodings, (last_hidden_states, last_cell_states) = self.lstmmodule(intermediate_encodings)
            lstm_encodings, _ = pad_packed_sequence(lstm_encodings, batch_first=True, padding_value=0)

            # out
            final_encodings = lstm_encodings

        else:

            # concat aux_embs
            # if not None, the expected dim for aux_word_embs: [BS,max_nwords,*]
            intermediate_encodings = batch_screps
            if aux_word_embs is not None:
                intermediate_encodings = torch.cat((intermediate_encodings, aux_word_embs), dim=2)

                # lstm
            # dim: [BS,max_nwords,*]->[BS,max_nwords,self.lstmmodule_outdim]
            intermediate_encodings = pack_padded_sequence(intermediate_encodings, batch_lengths,
                                                          batch_first=True, enforce_sorted=False)
            lstm_encodings, (last_hidden_states, last_cell_states) = self.lstmmodule(intermediate_encodings)
            lstm_encodings, _ = pad_packed_sequence(lstm_encodings, batch_first=True, padding_value=0)

            # out
            final_encodings = torch.cat((lstm_encodings, bert_merged_encodings), dim=2)

        # dense
        # [BS,max_nwords,self.encodings_outdim]->[BS,max_nwords,output_dim]
        logits = self.dense(self.dropout(final_encodings))

        # loss
        if targets is not None:
            assert len(targets) == batch_size  # targets:[[BS,max_nwords]
            logits_permuted = logits.permute(0, 2, 1)  # logits: [BS,output_dim,max_nwords]
            loss = self.criterion(logits_permuted, targets)

        # eval preds
        if not self.training:
            probs = F.softmax(logits, dim=-1)  # [BS,max_nwords,output_dim]
            if topk > 1:
                topk_values, topk_inds = \
                    torch.topk(probs, topk, dim=-1, largest=True,
                               sorted=True)  # -> (Tensor, LongTensor) of [BS,max_nwords,topk]
            elif topk == 1:
                topk_inds = torch.argmax(probs, dim=-1)  # [BS,max_nwords]

            # Note that for those positions with padded_idx,
            #   the arg_max_prob above computes a index because 
            #   the bias term leads to non-uniform values in those positions

            return loss.cpu().detach().numpy(), topk_inds.cpu().detach().numpy()
        return loss