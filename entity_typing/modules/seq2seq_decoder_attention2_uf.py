#!/usr/bin/env python
# -*- coding: utf-8 -*-
import random
from typing import List, Dict

import torch
import torch.nn as nn
from allennlp.data import Vocabulary
from allennlp.modules import TextFieldEmbedder
from allennlp.modules.attention import BilinearAttention, AdditiveAttention, DotProductAttention, LinearAttention
from allennlp.nn import Activation

from entity_typing.constant import EOS_SYMBOL, UNK_SYMBOL, MAX_VAL
from entity_typing.modules.mlp_attention import SelfAttentiveSum


class Seq2SeqAttn2DecoderUF(nn.Module):
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
        super(Seq2SeqAttn2DecoderUF, self).__init__()
        self._vocab = vocab
        self._label_embedding_dim = label_embedding_dim
        self._decoder_hidden_size = decoder_hidden_size
        self._decoder = nn.LSTM(input_size=self._label_embedding_dim,
                                hidden_size=self._decoder_hidden_size,
                                num_layers=1,
                                batch_first=True,
                                bias=True)
        # output -> vocab
        self._mlp = nn.Sequential(
            nn.Linear(in_features=self._decoder_hidden_size + self._label_embedding_dim,
                      out_features=self._decoder_hidden_size + self._label_embedding_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features=self._decoder_hidden_size + self._label_embedding_dim,
                      out_features=output_size),
        )
        self._label_namespace = label_namespace
        self._teaching_forcing_rate = teaching_forcing_rate
        self._label_field_embedder = label_field_embedder
        self._eos_label_id = vocab.get_token_index(EOS_SYMBOL, namespace=self._label_namespace[0])
        self._unk_label_id = vocab.get_token_index(UNK_SYMBOL, namespace=self._label_namespace[0])
        self._seq_label_num = output_size
        self._decode_max_seq_len = decode_max_seq_len
        activation = Activation.by_name('relu')()
        # ATT1
        self._dense_net = nn.Linear(in_features=decoder_hidden_size,
                                    out_features=encoder_hidden_dim)
        self._dense_net2 = SelfAttentiveSum(output_dim=label_embedding_dim,
                                            hidden_dim=label_embedding_dim,
                                            activation=activation)

        self.ge_proj1 = nn.Linear(label_embedding_dim, label_embedding_dim)
        self.ge_proj2 = nn.Linear(label_embedding_dim, label_embedding_dim)
        self._dropout = nn.Dropout(dropout_rate)

    def forward(self, encoded_hidden_embeddings: torch.Tensor, hidden: torch.Tensor, cell: torch.Tensor,
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
        # 1, B, Hd
        h0 = hidden.permute(1, 0, 2).reshape(batch_size, -1).view(batch_size, 1, -1).permute(1, 0, 2)
        # 1, B, Hd
        c0 = cell.permute(1, 0, 2).reshape(batch_size, -1).view(batch_size, 1, -1).permute(1, 0, 2)

        # B, 1, Ed
        input_label_embeddings = encoded_hidden_embeddings.new_zeros((batch_size, 1, self._label_embedding_dim))

        # B, Sd, L
        outputs = encoded_hidden_embeddings.new_tensor([])
        # B, 1, C
        logits_mask = input_label_embeddings.new_zeros((batch_size, 1, self._seq_label_num), dtype=torch.int64)

        # B, N, Ed
        all_label_embedding = self._label_field_embedder(
            {self._label_namespace[1]: {
                'tokens': torch.LongTensor(range(self._seq_label_num)).to('cuda').repeat(batch_size, 1)}}
        )
        if self.training and gold_label_ids is not None:
            decode_max_seq_len = gold_label_ids.size(1)
        else:
            decode_max_seq_len = self._decode_max_seq_len

        next_input_ids = []
        last_label_embeddings = input_label_embeddings.new_zeros(input_label_embeddings.size())
        # last_label_embeddings = input_label_embeddings.new_tensor([])

        for step in range(decode_max_seq_len):
            # output shape: B, 1, Hd
            # h1 shape: 1, B, Hd
            # c1 shape: 1, B, Hd
            output, (h1, c1) = self._decoder(input_label_embeddings, (h0, c0))

            h0 = h1
            c0 = c1

            # B, He, 1  # dot attention
            query_tensors = self._dense_net(h0.transpose(1, 0)).transpose(1, 2)
            # B, Se, 1 = B, Se, He * B, He, 1
            weight_score = torch.matmul(encoded_hidden_embeddings, query_tensors)
            # B, Se, 1  # linear attention
            # weight_score = self._dense_net(input_label_embeddings.squeeze(1), encoded_hidden_embeddings)
            # weight_score = weight_score.unsqueeze(2)
            # B, Se, 1
            pad_value_mask = (encoder_mask.logical_not() * -MAX_VAL).unsqueeze(2)
            # B, Se, 1
            weight = (weight_score + pad_value_mask).softmax(1)
            topk1 = weight.topk(3, dim=1)
            # B, 1, He
            attn_encoder_hidden_embeddings = torch.matmul(encoded_hidden_embeddings.transpose(1, 2),
                                                          weight).transpose(1, 2)

            weighted_values, weight = self._dense_net2(last_label_embeddings)

            # B, Se, 1
            # weight = weight_score.softmax(1)
            topk2 = weight.topk(1, dim=1)
            # attn_last_label_embeddings = torch.matmul(last_label_embeddings.transpose(1, 2),
            #                                                weight).transpose(1, 2)
            attn_last_label_embeddings = weighted_values.unsqueeze(1)

            sentence_repre = self._dropout(encoded_hidden_embeddings[:, 0, :].unsqueeze(1))  # DP2
            m1 = sentence_repre + attn_encoder_hidden_embeddings  # AD1
            m2 = input_label_embeddings + attn_last_label_embeddings  # AD2
            # # B, 1, Ed + He
            m = torch.cat((m1, m2), dim=-1)

            output = self._mlp(m) + logits_mask
            # output = output.softmax(2)
            # B, Sd, L
            outputs = torch.cat((outputs, output), dim=1)

            # B, 1
            next_input_label_ids = output.argmax(dim=-1)

            # soft_label_embeddings = torch.matmul(output.softmax(2), all_label_embedding)
            # input_label_embeddings = soft_label_embeddings

            pred_label_embeddings = self._label_field_embedder(
                {self._label_namespace[1]: {'tokens': next_input_label_ids}})
            input_label_embeddings = pred_label_embeddings

            next_input_label_ids = next_input_label_ids.squeeze(1)

            next_input_ids.append(next_input_label_ids)
            if len(next_input_ids) == 1:
                last_label_embeddings = input_label_embeddings
            else:
                last_label_embeddings = torch.cat([last_label_embeddings, input_label_embeddings], dim=1)

            # B, L
            select_idx = torch.stack(next_input_ids, dim=1)
            logits_mask = logits_mask.squeeze(1).scatter(1, select_idx, -1e7).unsqueeze(1)
            logits_mask[:, :, self._eos_label_id] = 0

        next_input_ids = torch.stack(next_input_ids, dim=1)

        return outputs, next_input_ids
