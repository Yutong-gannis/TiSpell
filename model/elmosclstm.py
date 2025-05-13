import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.rnn import pad_sequence


class ElmoSCLSTM(nn.Module):
    def __init__(self, screp_dim, padding_idx, output_dim, early_concat=True):
        super(ElmoSCLSTM, self).__init__()

        self.early_concat = early_concat  # if True, (elmo+sc)->lstm->linear, else ((sc->lstm)+elmo)->linear

        self.elmo = get_pretrained_elmo()
        self.elmomodule_outdim = 1024

        # lstm module
        # expected  input dim: [BS,max_nwords,*] and batch_lengths as [BS] for pack_padded_sequence
        bidirectional, hidden_size, nlayers = True, 512, 2
        if self.early_concat:
            self.lstmmodule_indim = screp_dim + self.elmomodule_outdim
            self.lstmmodule = nn.LSTM(self.lstmmodule_indim, hidden_size, nlayers,
                                      batch_first=True, dropout=0.3, bidirectional=bidirectional)
            self.lstmmodule_outdim = hidden_size * 2 if bidirectional else hidden_size
            self.encodings_outdim = self.lstmmodule_outdim
        else:
            self.lstmmodule_indim = screp_dim
            self.lstmmodule = nn.LSTM(self.lstmmodule_indim, hidden_size, nlayers,
                                      batch_first=True, dropout=0.3, bidirectional=bidirectional)
            self.lstmmodule_outdim = hidden_size * 2 if bidirectional else hidden_size
            self.encodings_outdim = self.lstmmodule_outdim + self.elmomodule_outdim

        # output module
        assert output_dim > 0
        self.dropout = nn.Dropout(p=0.4)
        self.dense = nn.Linear(self.encodings_outdim, output_dim)

        # loss
        # See https://pytorch.org/docs/stable/nn.html#crossentropyloss
        self.criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=padding_idx)

    def forward(self,
                batch_screps: List[pad_sequence],
                batch_lengths: torch.tensor,
                elmo_inp: torch.tensor,
                aux_word_embs: torch.tensor = None):

        if aux_word_embs is not None:
            raise Exception("dimensions of aux_word_embs not used in __init__()")

        # cnn
        batch_size = len(batch_screps)
        batch_screps = pad_sequence(batch_screps, batch_first=True, padding_value=0)

        # elmo
        elmo_encodings = self.elmo(elmo_inp)['elmo_representations'][0]  # BS X max_nwords x 1024

        if self.early_concat:

            # concat aux_embs
            # if not None, the expected dim for aux_word_embs: [BS,max_nwords,*]
            intermediate_encodings = torch.cat((batch_screps, elmo_encodings), dim=2)
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
            final_encodings = torch.cat((lstm_encodings, elmo_encodings), dim=2)

        # dense
        # [BS,max_nwords,self.encodings_outdim]->[BS,max_nwords,output_dim]
        logits = self.dense(self.dropout(final_encodings))