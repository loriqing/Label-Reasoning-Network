#!/usr/bin/env python
# -*- coding: utf-8 -*-
import random, math
from typing import List, Dict

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
from allennlp.data import Vocabulary
from allennlp.modules import TextFieldEmbedder

from entity_typing.constant import EOS_SYMBOL, MAX_VAL


class Seq2SeqTransformerDecoder2(nn.Module):
    def __init__(self,
                 vocab: Vocabulary,
                 label_field_embedder: TextFieldEmbedder,
                 label_embedding_dim: int,
                 input_size: int,
                 decoder_hidden_size: int,
                 encoder_hidden_dim: int,
                 output_size: int,
                 dropout_rate: float,
                 teaching_forcing_rate: float,
                 decode_max_seq_len: int = 20,
                 label_namespace: List[str] = ['seq_label_vocab', 'seq_labels'],
                 ):
        super(Seq2SeqTransformerDecoder2, self).__init__()
        self._vocab = vocab
        self._label_embedding_dim = label_embedding_dim
        self._decoder_hidden_size = decoder_hidden_size
        decoder_layer = nn.TransformerDecoderLayer(d_model=encoder_hidden_dim, nhead=8)
        self._decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)
        self._mlp = nn.Sequential(
            nn.Linear(in_features=self._decoder_hidden_size,
                      out_features=self._decoder_hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features=self._decoder_hidden_size,
                      out_features=output_size),
        )
        self._label_namespace = label_namespace
        self._teaching_forcing_rate = teaching_forcing_rate
        self._label_field_embedder = label_field_embedder
        self._eos_label_id = vocab.get_token_index(EOS_SYMBOL, namespace=self._label_namespace[0])
        self._seq_label_num = output_size

        self._decode_max_seq_len = decode_max_seq_len
        self._dense_net = nn.Linear(in_features=label_embedding_dim,
                                    out_features=encoder_hidden_dim)
        self._input_mlp = nn.Linear(input_size, encoder_hidden_dim)

        self._position_emb = PositionalEncoding(d_model=encoder_hidden_dim, )
        self._dropout = nn.Dropout(dropout_rate)

    def get_memory_mask(self, text_lengths, target_len):
        # :shape text_lengths: B
        max_len = max(text_lengths).item()
        mask = text_lengths.new_zeros((target_len, max_len))
        for idx, i in enumerate(text_lengths):
            mask[idx, :i.item()] = 1
        return mask

    def _generate_subsequent_mask(self, tgt_sz, src_sz):
        mask = (torch.triu(torch.ones(src_sz, tgt_sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.to('cuda')

    def forward(self, encoded_hidden_embeddings: torch.Tensor,  hidden: torch.Tensor, cell: torch.Tensor,
                text_lengths: torch.Tensor, encoder_mask: torch.BoolTensor,
                gold_label_ids: torch.Tensor = None, gold_label_embeddings: torch.Tensor = None,
                seq_label_mask=None) -> torch.Tensor:
        """
        :shape encoded_hidden_embeddings: B, Se, He
        :shape hidden: De, B, He
        :shape cell: De, B, He
        :shape text_lengths: B
        :shape encoder_mask: B, Se
        :shape gold_label_ids: B, Sd, 1
        :shape gold_label_embeddings: B, Sd, Ed
        :return:
        """
        """
        B: batch_size, C: Classes  
        He: encoder_hidden_dim, Hd: decoder_hidden_dim 
        Ee: encoder_text_embedding_dim, Ed: decoder_label_embedding_dim 
        De: encoder's (direction * layers), Dd: decoder's (direction * layers)
        Se: encoder_text_sequence_len, Sd: decoder_label_sequence_len
        N: seq_label_num
        """
        batch_size = encoded_hidden_embeddings.size(0)
        encoder_hidden_dim = encoded_hidden_embeddings.size(-1)

        # B, 1, Ed
        input_label_embeddings = encoded_hidden_embeddings.new_zeros((batch_size, 1, self._label_embedding_dim))
        # B, 1, He
        input_encoder_hidden_embeddings = encoded_hidden_embeddings.new_zeros((batch_size, 1, encoder_hidden_dim))
        # B, 1, Ed + He
        input_embeddings = self._input_mlp(torch.cat((input_label_embeddings, input_encoder_hidden_embeddings), dim=-1))

        if self.training and gold_label_ids is not None:
            decode_max_seq_len = gold_label_ids.size(1)
        else:
            decode_max_seq_len = self._decode_max_seq_len

        if self.training:
            tgt_mask = self._generate_subsequent_mask(decode_max_seq_len, decode_max_seq_len)
            tgt_key_padding_mask = ~seq_label_mask
            memory_mask = None
            memory_key_padding_mask = ~encoder_mask
            repeat_hidden = encoded_hidden_embeddings[:, 0, :].unsqueeze(1).repeat(1, gold_label_embeddings.size(1), 1)
            repeat_hidden = self._dropout(repeat_hidden)  # DP2
            shift_gold_label_embeddings = torch.cat([input_label_embeddings, gold_label_embeddings[:, :-1, :]], dim=1)
            input_embeddings = self._input_mlp(torch.cat((shift_gold_label_embeddings, repeat_hidden), dim=-1))
            input_embeddings = self._position_emb(input_embeddings)
            # L, B, Hd
            outputs = self._decoder(tgt=input_embeddings.transpose(1, 0),
                                    memory=encoded_hidden_embeddings.transpose(1, 0),
                                    tgt_mask=tgt_mask, memory_mask=memory_mask,
                                    tgt_key_padding_mask=tgt_key_padding_mask,
                                    memory_key_padding_mask=memory_key_padding_mask)
            outputs = self._mlp(outputs).transpose(1, 0)
            # B, Sd
            next_input_ids = outputs.argmax(dim=-1)
        else:
            # B, Sd, L
            outputs = encoded_hidden_embeddings.new_tensor([])
            # B, 1, C
            logits_mask = input_label_embeddings.new_zeros((batch_size, 1, self._seq_label_num), dtype=torch.int64)

            # B, C, Ed
            all_label_embedding = self._label_field_embedder(
                {self._label_namespace[1]: {
                    'tokens': torch.LongTensor(range(self._seq_label_num)).to('cuda').repeat(batch_size, 1)}}
            )
            next_input_ids = []
            for step in range(decode_max_seq_len):

                tgt_mask = None
                tgt_key_padding_mask = None
                memory_mask = None
                memory_key_padding_mask = ~encoder_mask

                # L-1, B, Hd
                output = self._decoder(tgt=input_embeddings.transpose(1, 0), memory=encoded_hidden_embeddings.transpose(1, 0),
                                       tgt_mask=tgt_mask, memory_mask=memory_mask,
                                       tgt_key_padding_mask=tgt_key_padding_mask,
                                       memory_key_padding_mask=memory_key_padding_mask)

                output = self._mlp(output.transpose(1, 0).mean(dim=1).unsqueeze(1)) + logits_mask
                # B, Sd, L
                outputs = torch.cat((outputs, output), dim=1)

                # B, 1
                next_input_label_ids = output.argmax(dim=-1)

                pred_label_embeddings = self._label_field_embedder(
                    {self._label_namespace[1]: {'tokens': next_input_label_ids}})

                if step > 0:
                    input_label_embeddings = torch.cat([input_label_embeddings, pred_label_embeddings], dim=1)
                else:
                    input_label_embeddings = pred_label_embeddings
                next_input_label_ids = next_input_label_ids.squeeze(1)

                # (B, 1) * L
                next_input_ids.append(next_input_label_ids)

                # B, L
                select_idx = torch.stack(next_input_ids, dim=1)
                logits_mask = logits_mask.squeeze(1).scatter(1, select_idx, -1e7).unsqueeze(1)
                logits_mask[:, :, self._eos_label_id] = 0

                repeat_hidden = encoded_hidden_embeddings[:, 0, :].unsqueeze(1).repeat(1, len(next_input_ids), 1)
                repeat_hidden = self._dropout(repeat_hidden)  # DP2
                input_embeddings = self._input_mlp(torch.cat((input_label_embeddings, repeat_hidden), dim=-1))
                input_embeddings = self._position_emb(input_embeddings)

            # B, Sd
            next_input_ids = torch.stack(next_input_ids, dim=1)

        return outputs, next_input_ids


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


if __name__ == "__main__":
    src_sz = 5
    tgt_sz = 4
    mask = (torch.triu(torch.ones(src_sz, tgt_sz)) == 1).transpose(0, 1)
    print(mask)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    print(mask)