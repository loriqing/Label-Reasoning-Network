#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, MultiLabelSoftMarginLoss
from scipy.optimize import linear_sum_assignment
from entity_typing.constant import BITMATCH_ERROR


class BipartiteMatchingLoss(nn.Module):

    def __init__(self):
        super(BipartiteMatchingLoss, self).__init__()
        self._epsilon = 1e-8

    def hungary(self, task_matrix):
        b = task_matrix.clone().detach().cpu()
        return linear_sum_assignment(b)

    def forward(self, outputs: torch.Tensor, gold_label_ids: torch.Tensor, mask: torch.BoolTensor,
                weight: torch.Tensor = None,
                reduce: bool = True):
        # B: batch_size, H: hidden_dim, E: embedding_dim, D: direction * layers, S:max_sequence_len, L: label_nums
        # outputs: B, S, L
        # gold_label_ids: B, S
        # mask: B, S

        seq_mask = mask.reshape(-1)
        batch_size, max_seq_len, class_num = outputs.size()
        if weight is None:
            weight = outputs.new_ones(class_num)

        # check the no change loss
        check_loss = nn.functional.cross_entropy(outputs.reshape(batch_size * max_seq_len, -1),
                                                 gold_label_ids.reshape(-1), weight, reduction='none')
        check_loss = check_loss * seq_mask
        if reduce:
            check_loss = torch.sum(check_loss) / (torch.sum(seq_mask) + self._epsilon)

        match_gold_label_ids = gold_label_ids.new_zeros(size=gold_label_ids.size())

        for idx in range(batch_size):
            # n, 1
            gold_len = mask[idx].sum()
            gold_label = gold_label_ids[idx, :gold_len]
            # S, n
            probs = outputs[idx][:, gold_label]
            # n
            w = weight[gold_label]
            probs = -torch.log_softmax(probs, 1) * w.unsqueeze(0)
            probs = torch.where(torch.isinf(probs), torch.full_like(probs, 1e8), probs)
            try:
                row_ind, col_ind = self.hungary(probs)
            except Exception as e:
                print(e)
                print(outputs)
                print(gold_label_ids)
                print(mask)
                print(probs)
                exit(0)
            match_gold_label_ids[idx, :gold_len] = gold_label_ids[idx].gather(dim=0, index=torch.LongTensor(col_ind).to('cuda'))

        # print('match_gold_label_ids: ', match_gold_label_ids)
        loss = nn.functional.cross_entropy(outputs.reshape(batch_size * max_seq_len, -1),
                                           match_gold_label_ids.reshape(-1),
                                           weight,
                                           reduction='none')
        loss = loss * seq_mask
        if reduce:
            loss = torch.sum(loss) / (torch.sum(seq_mask) + self._epsilon)

        # if check_loss < loss:
        #     BITMATCH_ERROR += 1
        # assert check_loss >= loss, '{0}, {1}, {2}, {3}'.format(check_loss, loss, gold_label_ids, match_gold_label_ids)
        return loss


if __name__ == "__main__":
    import numpy as np
    # cost = np.array([[4, 1, 3], [2, 0, 5], [3, 2, 2]])
    # from scipy.optimize import linear_sum_assignment
    # row_ind, col_ind = linear_sum_assignment(cost)
    # print(col_ind)
    # print(cost[row_ind, col_ind].sum())

    loss_function = BipartiteMatchingLoss()
    task_matrix = np.random.randint(0, 100, size=(5, 5))
    task_matrix = torch.Tensor(task_matrix)
    min_cost, best_solution = loss_function.hungary(task_matrix)
    print(task_matrix)
    print(min_cost, best_solution)

    outputs = torch.Tensor(2,3,4)
    gold = torch.LongTensor([[2,1,3],[2,1,0]])
    mask = (gold > 0)
    loss = loss_function(outputs, gold, mask)
    print(loss)

