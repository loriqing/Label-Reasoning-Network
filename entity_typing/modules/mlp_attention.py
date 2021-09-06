#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch.nn as nn
import torch

from allennlp.nn.util import masked_softmax

from entity_typing.constant import MAX_VAL


class SelfAttentiveSum(nn.Module):
    """
    Attention mechanism to get a weighted sum of RNN output sequence to a single RNN output dimension.
    """

    def __init__(self, output_dim, hidden_dim, activation):
        super(SelfAttentiveSum, self).__init__()
        self.key_maker = nn.Linear(output_dim, hidden_dim, bias=False)
        self.activation = activation
        self.hidden_dim = hidden_dim
        self.key_output = nn.Linear(hidden_dim, 1, bias=False)
        self.key_softmax = nn.Softmax(dim=1)

    def forward(self, input_embed, mask=None):
        '''
        :param input_embed: (batch_size, len, output_dim)
        :param mask: (batch_size, len)
        :return: ()
        '''
        # (batch_size * len, output_dim)
        input_embed_squeezed = input_embed.view(-1, input_embed.size()[2])
        # (batch_size * len, hidden_dim)
        k_d = self.key_maker(input_embed_squeezed)
        # (batch_size * len, hidden_dim)
        k_d = self.activation(k_d)
        if self.hidden_dim == 1:
            # (batch_size, len)
            k = k_d.view(input_embed.size()[0], -1)
        else:
            # (batch_size, len)
            k = self.key_output(k_d).view(input_embed.size()[0], -1)

        if mask is not None:
            # (batch_size, len, 1)
            weighted_keys = masked_softmax(k, mask).view(input_embed.size()[0], -1, 1)
        else:
            # (batch_size, len, 1)
            weighted_keys = self.key_softmax(k).view(input_embed.size()[0], -1, 1)

        # batch_size, len, output_dim -> batch_size, output_dim
        weighted_values = torch.sum(weighted_keys * input_embed, 1)

        return weighted_values, weighted_keys
