import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.rnn import pad_sequence
from typing import List


class CharLSTMModel(nn.Module):
    def __init__(self, nembs, embdim, hidden_size, num_layers, bidirectional, output_combination):
        super(CharLSTMModel, self).__init__()

        # Embeddings
        self.embeddings = nn.Embedding(nembs, embdim)
        # torch.nn.init.normal_(self.embeddings.weight.data, std=1.0)
        self.embeddings.weight.requires_grad = True

        # lstm module
        # expected input dim: [BS,max_nwords,*] and batch_lengths as [BS] for pack_padded_sequence
        self.lstmmodule = nn.LSTM(embdim, hidden_size, num_layers, batch_first=True, dropout=0.3,
                                  bidirectional=bidirectional)
        self.lstmmodule_outdim = hidden_size * 2 if bidirectional else hidden_size

        # output
        assert output_combination in ["end", "max", "mean"], print(
            'invalid output_combination; required one of {"end","max","mean"}')
        self.output_combination = output_combination

    def forward(self, batch_tensor, batch_lengths):

        batch_size = len(batch_tensor)

        # [BS, max_seq_len]->[BS, max_seq_len, emb_dim]
        embs = self.embeddings(batch_tensor)

        # lstm
        lstm_encodings, (_, _) = self.lstmmodule(embs)
        # [BS, max_seq_len, self.lstmmodule_outdim]->[BS, self.lstmmodule_outdim]
        if self.output_combination == "end":
            last_seq_idxs = torch.LongTensor([x - 1 for x in batch_lengths])
            source_encodings = lstm_encodings[range(lstm_encodings.shape[0]), last_seq_idxs, :]
        elif self.output_combination == "max":
            source_encodings, _ = torch.max(lstm_encodings, dim=1)
        elif self.output_combination == "mean":
            sum_ = torch.sum(lstm_encodings, dim=1)
            lens_ = batch_lengths.unsqueeze(dim=1).expand(batch_size, self.lstmmodule_outdim)
            assert sum_.size() == lens_.size()
            source_encodings = torch.div(sum_, lens_)
        else:
            raise NotImplementedError

        return source_encodings


class CharLSTMWordLSTMModel(nn.Module):
    def __init__(self, args):
        super(CharLSTMWordLSTMModel, self).__init__()
        nchars = args.char_vocab_size
        char_emb_dim = args.hidden_size
        output_dim = args.word_vocab_size
  
        # charlstm module
        # takes in a list[pad_sequence] with each pad_sequence of dim: [BS][nwords,max_nchars]
        # runs a for loop to obtain list[tensor] with each tensor of dim: [BS][nwords,charlstm_outputdim]
        # then use rnn.pad_sequence(.) to obtain the dim: [BS, max_nwords, charlstm_outputdim]
        hidden_size = args.hidden_size
        num_layers = 2
        bidirectional, output_combination = True, "max"
        self.charlstmmodule = CharLSTMModel(nchars, char_emb_dim, hidden_size, num_layers,
                                            bidirectional, output_combination)
        self.charlstmmodule_outdim = self.charlstmmodule.lstmmodule_outdim

        # lstm module
        # expected  input dim: [BS,max_nwords,*] and batch_lengths as [BS] for pack_padded_sequence
        hidden_size = args.hidden_size
        num_layers = 2
        bidirectional = True
        self.lstmmodule = nn.LSTM(self.charlstmmodule_outdim, hidden_size, num_layers,
                                  batch_first=True, dropout=0.3, bidirectional=bidirectional)
        self.lstmmodule_outdim = hidden_size * 2 if bidirectional else hidden_size

        # output module
        assert output_dim > 0
        self.dropout = nn.Dropout(p=0.4)
        self.dense = nn.Linear(self.lstmmodule_outdim, output_dim)

    def forward(self,
                batch_idxs: List[pad_sequence],
                batch_char_lengths: List[torch.tensor],
                batch_lengths: torch.tensor,
                aux_word_embs: torch.tensor = None):

        batch_size = len(batch_idxs)
        # print("************ stage 1")

        # charlstm
        charlstm_encodings = [self.charlstmmodule(pad_sequence_, lens) for pad_sequence_, lens in
                              zip(batch_idxs, batch_char_lengths)]
        charlstm_encodings = pad_sequence(charlstm_encodings, batch_first=True, padding_value=0)

        # concat aux_embs
        # if not None, the expected dim for aux_word_embs: [BS,max_nwords,*]
        intermediate_encodings = charlstm_encodings
        if aux_word_embs is not None:
            intermediate_encodings = torch.cat((intermediate_encodings, aux_word_embs), dim=2)

        # lstm
        # dim: [BS,max_nwords,*]->[BS,max_nwords,self.lstmmodule_outdim]
        intermediate_encodings = pack_padded_sequence(intermediate_encodings, batch_lengths,
                                                      batch_first=True, enforce_sorted=False)
        lstm_encodings, (last_hidden_states, last_cell_states) = self.lstmmodule(intermediate_encodings)
        lstm_encodings, _ = pad_packed_sequence(lstm_encodings, batch_first=True, padding_value=0)

        # dense
        # [BS,max_nwords,self.lstmmodule_outdim]->[BS,max_nwords,output_dim]
        logits = self.dense(self.dropout(lstm_encodings))

        return logits