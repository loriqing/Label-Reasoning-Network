import torch
import torch.nn as nn
from typing import Dict

from torch.nn import BCEWithLogitsLoss, MultiLabelSoftMarginLoss
from entity_typing import utils


class MultiLayerLoss(nn.Module):
    def __init__(self, label_namespace=['labels']):
        super().__init__()
        self.label_namespace = label_namespace
        self.loss_function = BCEWithLogitsLoss(reduction='mean')

    def forward(self, pred_labels: Dict[str, torch.Tensor], gold_labels: Dict[str, torch.Tensor]):
        loss = None
        for (name1, pred_label), (name2, gold_label) in zip(pred_labels.items(), gold_labels.items()):
            assert name1 == name2, "{0} is not equal to {1}, in MultiLayerLoss calculate. ".format(name1, name2)
            if name1 not in self.label_namespace:
                continue
            if gold_label is not None:
                select_idx = utils.get_select_idx(gold_label)
                if select_idx is not None:
                    true = torch.index_select(gold_label, 0, select_idx)
                    pred = torch.index_select(pred_label, 0, select_idx)
                    if loss is None:
                        loss = self.loss_function(pred, true.float())
                    else:
                        loss += self.loss_function(pred, true.float())
        return loss
