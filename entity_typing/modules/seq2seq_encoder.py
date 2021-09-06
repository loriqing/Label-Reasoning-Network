#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Tuple

import torch
import torch.nn as nn
from allennlp.nn.util import sort_batch_by_length


class Seq2SeqEncoder(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 bidirectional: bool = True):
        super(Seq2SeqEncoder, self).__init__()
        self._hidden_size = hidden_size
        self._encoder = nn.LSTM(input_size=input_size,
                                hidden_size=hidden_size,
                                num_layers=1,
                                bidirectional=bidirectional,
                                batch_first=True,
                                bias=True)
        # direction * layers
        self._channels = (int(bidirectional) + 1) * self._encoder.num_layers

    def forward(self, text_embedding, mask: torch.BoolTensor) -> \
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # B: batch_size, H: hidden_dim, E: embedding_dim, D: direction * layers, S:max_sequence_len, L: label_nums
        # text_embedding B, S, E
        batch_size, max_seq_len, input_size = text_embedding.size()
        # B
        lengths = mask.sum(-1)
        # sorted_text_embedding B, S, E
        # sorted_sequence_lengths B
        # restoration_indices B
        # sorting_indices B
        sorted_text_embedding, sorted_sequence_lengths, restoration_indices, sorting_indices = sort_batch_by_length(
            text_embedding, lengths)
        pack_embedding = nn.utils.rnn.pack_padded_sequence(sorted_text_embedding, sorted_sequence_lengths,
                                                           batch_first=True)
        # D, B, H
        h0 = sorted_text_embedding.new_zeros(self._channels, batch_size, self._hidden_size)
        # D, B, H
        c0 = sorted_text_embedding.new_zeros(self._channels, batch_size, self._hidden_size)
        # h_last D, B, E
        # c_last D, B, E
        out_embedding, (h_last, c_last) = self._encoder(pack_embedding, (h0, c0))

        # encoded_hidden_embeddings shape B, S, E*2
        # out_len B
        encoded_hidden_embeddings, out_len = nn.utils.rnn.pad_packed_sequence(out_embedding, batch_first=True)

        encoded_hidden_embeddings = encoded_hidden_embeddings.index_select(0, restoration_indices)
        h_last = h_last.index_select(1, restoration_indices)
        c_last = c_last.index_select(1, restoration_indices)

        return encoded_hidden_embeddings, h_last, c_last, lengths
