#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


class Seq2SeqLoss(nn.Module):

    # gold_ids shape: 1D
    def smooth_one_hot(self, labels_len: int, class_num: int, gold_ids: torch.Tensor):
        confidence = 1 - self._smoothing
        smooth_true_dist = gold_ids.new_ones(labels_len, class_num) * (self._smoothing / (class_num - 1))
        smooth_true_dist.scatter_(dim=1, index=gold_ids.unsqueeze(1), value=confidence)
        return smooth_true_dist

    def __init__(self, smoothing: float):
        super(Seq2SeqLoss, self).__init__()
        assert 0 <= smoothing < 1
        self._smoothing = smoothing
        self._epsilon = 1e-8

    def forward(self, outputs: torch.Tensor, gold_label_ids: torch.Tensor, mask: torch.BoolTensor,
                weight: torch.Tensor = None,
                reduce: bool = True):
        # B: batch_size, H: hidden_dim, E: embedding_dim, D: direction * layers, S:max_sequence_len, L: label_nums
        # outputs: B, S, L
        # gold_label_ids: B, S
        # mask: B, S
        batch_size, max_seq_len, class_num = outputs.size()
        if weight is None:
            weight = outputs.new_ones(class_num)

        labels_len = batch_size * max_seq_len
        # B * S, L
        output_dist = -outputs.reshape(labels_len, -1).log_softmax(dim=-1)
        # B * S, L
        smooth_true_dist = self.smooth_one_hot(labels_len, class_num, gold_label_ids.reshape(-1))

        if self._smoothing == 0:
            loss = nn.functional.cross_entropy(outputs.reshape(batch_size * max_seq_len, -1),
                                               gold_label_ids.reshape(-1),
                                               weight,
                                               reduction='none')
        else:
            # B * S
            loss = (weight * output_dist * smooth_true_dist).sum(dim=-1)

        # B * S
        seq_mask = mask.reshape(-1)
        loss = loss * seq_mask
        if reduce:
            loss = torch.sum(loss) / (torch.sum(seq_mask) + self._epsilon)
        return loss
