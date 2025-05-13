import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.rnn import pad_sequence
from typing import List


class SCLSTM(nn.Module):
    def __init__(self, args):
        super(SCLSTM, self).__init__()
        # lstm module
        # expected  input dim: [BS,max_nwords,*] and batch_lengths as [BS] for pack_padded_sequence
        screp_dim = 3 * args.char_vocab_size
        hidden_size = args.hidden_size
        output_dim = args.word_vocab_size
        nlayers = 4
        bidirectional = True
        self.lstmmodule = nn.LSTM(screp_dim, hidden_size, nlayers,
                                  batch_first=True, dropout=0.4, bidirectional=bidirectional)  # 0.3 or 0.4
        self.lstmmodule_outdim = hidden_size * 2 if bidirectional else hidden_size

        # output module
        assert output_dim > 0
        self.dropout = nn.Dropout(p=0.5)  # 0.4 or 0.5
        self.dense = nn.Linear(self.lstmmodule_outdim, output_dim)

    def forward(self,
                batch_screps: List[pad_sequence],
                batch_lengths: torch.tensor,
                aux_word_embs: torch.tensor = None):

        # cnn
        batch_size = len(batch_screps)
        batch_screps = pad_sequence(batch_screps, batch_first=True, padding_value=0)

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

        # dense
        # [BS,max_nwords,self.lstmmodule_outdim]->[BS,max_nwords,output_dim]
        logits = self.dense(self.dropout(lstm_encodings))
        return logits