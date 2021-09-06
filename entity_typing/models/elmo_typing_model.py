import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Dict, List

from allennlp.models import Model
from allennlp.data import Vocabulary
from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder
from allennlp.nn.util import get_text_field_mask

from entity_typing.multi_layer_measure import MultiLayerF1Measure
from entity_typing.modules.multi_layer_loss import MultiLayerLoss

logger = logging.getLogger(__name__)


@Model.register("elmo_typing_model")
class ElmoTypingModel(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 context_field_embedder: TextFieldEmbedder,  # bert
                 context_encoder: Seq2VecEncoder = None,
                 dropout_rate: float = 0.3,
                 type_dim: int = None,
                 label_namespace: List[str] = ['labels'],
                 show: bool = False,
                 evaluation: str = 'macro',
                 ) -> None:
        super().__init__(vocab)
        self._context_field_embedder = context_field_embedder
        self._context_encoder = context_encoder

        self._dropout = dropout_rate
        self._label_namespace = label_namespace

        self.label_representation = nn.ModuleList()
        self.idx2namespace = {}
        for idx, namespace in enumerate(self._label_namespace):
            v = self.vocab._token_to_index[namespace]
            type_num = self.vocab.get_vocab_size(namespace)
            self.idx2namespace[idx] = namespace
            self.label_representation.append(nn.Linear(in_features=type_dim, out_features=type_num, bias=False))

        self._measure = MultiLayerF1Measure(label_namespace=label_namespace, evaluation=evaluation)
        self.loss_function = MultiLayerLoss(label_namespace=label_namespace)
        self.sigm = nn.Sigmoid()

        self._dropout = nn.Dropout(dropout_rate)

        self._show = show

    def forward(self, context, labels=None, fine_labels=None, ultra_fine_labels=None) -> Dict:
        context_mask = get_text_field_mask(context)
        context_embedding = self._context_field_embedder(context)  #, type_ids=type_ids
        # (batch_size, context_dim)
        context_hidden, _ = self._context_encoder(context_embedding, context_mask)  # self-attention
        context_hidden = self._dropout(context_hidden)  # DP1
        context_representation = context_hidden

        pred_scores = {}
        measure_scores = {}
        for idx, representation in enumerate(self.label_representation):
            logits = representation(context_representation)
            pred_scores[self.idx2namespace[idx]] = logits
            measure_scores[self.idx2namespace[idx]] = self.sigm(logits)
        forward_result = {'logits': pred_scores, 'measures': measure_scores}

        gold_labels = {"labels": labels}
        if fine_labels is not None:
            gold_labels["fine_labels"] = fine_labels
        if ultra_fine_labels is not None:
            gold_labels["ultra_fine_labels"] = ultra_fine_labels

        loss = self.loss_function(pred_scores, gold_labels)

        if loss is not None:
            forward_result['loss'] = loss
        self._measure(measure_scores, gold_labels)
        return forward_result

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        ret = self._measure.get_metric(reset)
        # result_metrics = ret
        result_metrics = {
            "precision": ret["precision"],
            "recall": ret["recall"],
            "fscore": ret["fscore"],
        }
        for k, v in ret.items():
            if k in result_metrics:
                continue
            else:
                key = k[0] + "_" + k.split("_")[-1]
                result_metrics[key] = v
        return result_metrics

    def make_output_human_readable(self, output_dict: Dict[str, torch.Tensor]) -> Dict:
        # Take the logits from the forward pass, and compute the label
        # IDs for > 0.5 values
        scores = {key: logit.cpu().data.numpy() for key, logit in output_dict['measures'].items()}

        for key, score in scores.items():
            output_dict['pred_' + key] = []
            for instance in score:
                label_idx = np.where(instance > 0.5)[-1]
                output_dict['pred_' + key].append([self.vocab.get_token_from_index(x, namespace=key)
                                                       for x in label_idx])

        return output_dict

