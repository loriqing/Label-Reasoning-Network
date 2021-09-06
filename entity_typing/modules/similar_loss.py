import torch.nn as nn
import torch


class SimilarLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        self.activation = nn.Tanh()

    def forward(self, similars: torch.Tensor, gold_labels: torch.Tensor, memory_mask: torch.Tensor) -> torch.Tensor:
        '''
        :param similar: (batch_size, memory_len)
        :param gold_labels: (batch_size, memory_len)
        :param memory_mask: (batch_size, memory_len)
        :return:
        '''
        log = -similars * gold_labels
        if memory_mask is not None:
            loss = log.masked_fill(~memory_mask, 0).mean(-1)
        if self.reduction == 'sum':
            loss = loss.sum(-1)
        if self.reduction == 'mean':
            loss = loss.mean(-1)
        return loss

        # (batch_size, memory_len)
        # log = - torch.log(self.activation(similars)) * gold_labels
        # loss = log.masked_fill(~memory_mask, 0).sum(-1)
        # if self.reduction == 'sum':
        #     loss = loss.sum(-1)
        # if self.reduction == 'mean':
        #     loss = loss.mean(-1)
        # return loss
