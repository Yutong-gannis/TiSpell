import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.rnn import pad_sequence
from typing import List


class CharCNNModel(nn.Module):
    def __init__(self, nembs, embdim, filterlens, nfilters):
        super(CharCNNModel, self).__init__()

        # Embeddings
        self.embeddings = nn.Embedding(nembs, embdim)
        # torch.nn.init.normal_(self.embeddings.weight.data, std=1.0)
        self.embeddings.weight.requires_grad = True

        # Unsqueeze [BS, MAXSEQ, EMDDIM] as [BS, 1, MAXSEQ, EMDDIM] and send as input
        self.convmodule = nn.ModuleList()
        for length, n in zip(filterlens, nfilters):
            self.convmodule.append(
                nn.Sequential(
                    nn.Conv2d(1, n, (length, embdim), padding=(length - 1, 0), dilation=1, bias=True,
                              padding_mode='zeros'),
                    nn.ReLU()
                )
            )
        # each conv outputs [BS, nfilters, MAXSEQ, 1]

    def forward(self, batch_tensor):
        batch_size = len(batch_tensor)

        # [BS, max_seq_len]->[BS, max_seq_len, emb_dim]
        embs = self.embeddings(batch_tensor)

        # [BS, max_seq_len, emb_dim]->[BS, 1, max_seq_len, emb_dim]
        embs_unsqueezed = torch.unsqueeze(embs, dim=1)

        # [BS, 1, max_seq_len, emb_dim]->[BS, out_channels, max_seq_len, 1]->[BS, out_channels, max_seq_len]
        conv_outputs = [conv(embs_unsqueezed).squeeze(3) for conv in self.convmodule]

        # [BS, out_channels, max_seq_len]->[BS, out_channels]
        maxpool_conv_outputs = [F.max_pool1d(out, out.size(2)).squeeze(2) for out in conv_outputs]

        # cat( [BS, out_channels] )->[BS, sum(nfilters)]
        source_encodings = torch.cat(maxpool_conv_outputs, dim=1)
        return source_encodings
    
    
class CharCNNWordLSTMModel(nn.Module):
    def __init__(self, args):
        super(CharCNNWordLSTMModel, self).__init__()
        nchars = args.char_vocab_size
        char_emb_dim = args.hidden_size
        output_dim = args.word_vocab_size

        # cnn module
        # takes in a list[pad_sequence] with each pad_sequence of dim: [BS][nwords,max_nchars]
        # runs a for loop to obtain list[tensor] with each tensor of dim: [BS][nwords,sum(nfilters)]
        # then use rnn.pad_sequence(.) to obtain the dim: [BS, max_nwords, sum(nfilters)]
        nfilters, filtersizes = [50, 100, 100, 100], [2, 3, 4, 5]
        self.cnnmodule = CharCNNModel(nchars, char_emb_dim, filtersizes, nfilters)
        self.cnnmodule_outdim = sum(nfilters)

        # lstm module
        # expected  input dim: [BS,max_nwords,*] and batch_lengths as [BS] for pack_padded_sequence
        bidirectional = True
        hidden_size = args.hidden_size
        nlayers = 2
        self.lstmmodule = nn.LSTM(self.cnnmodule_outdim, hidden_size, nlayers,
                                  batch_first=True, dropout=0.3, bidirectional=bidirectional)
        self.lstmmodule_outdim = hidden_size * 2 if bidirectional else hidden_size

        # output module
        assert output_dim > 0
        self.dropout = nn.Dropout(p=0.4)
        self.dense = nn.Linear(self.lstmmodule_outdim, output_dim)

    def forward(self,
                batch_idxs: List[pad_sequence],
                batch_lengths: torch.tensor,
                aux_word_embs: torch.tensor = None):

        batch_size = len(batch_idxs)

        # cnn
        cnn_encodings = [self.cnnmodule(pad_sequence_) for pad_sequence_ in batch_idxs]
        cnn_encodings = pad_sequence(cnn_encodings, batch_first=True, padding_value=0)

        # concat aux_embs
        # if not None, the expected dim for aux_word_embs: [BS,max_nwords,*]
        intermediate_encodings = cnn_encodings
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