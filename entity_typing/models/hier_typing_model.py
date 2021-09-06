import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List

from allennlp.models import Model
from allennlp.data import Vocabulary
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder, Seq2VecEncoder, Attention
from allennlp.nn.util import get_text_field_mask

from entity_typing.multi_layer_measure import MultiLayerF1Measure
from entity_typing import utils


@Model.register("hier_typing_model")
class HierTypingModel(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 context_field_embedder: TextFieldEmbedder,  # bert_word & pos + attention
                 mention_field_embedder: TextFieldEmbedder,  # char + CNN & word + attention(pooling)
                 mention_char_field_embedder: TextFieldEmbedder,
                 context_attention: Attention,
                 context_encoder: Seq2SeqEncoder = None,
                 mention_encoder: Seq2VecEncoder = None,
                 dropout_rate: float = 0.3,
                 type_dim: int = None,
                 label_namespace: List[str] = ['labels'],
                 show: bool = False,
                 ) -> None:
        super().__init__(vocab)
        self._context_field_embedder = context_field_embedder
        self._mention_field_embedder = mention_field_embedder
        self._mention_char_field_embedder = mention_char_field_embedder
        self._context_attention = context_attention
        self._context_encoder = context_encoder
        self._mention_encoder = mention_encoder

        self._dropout = dropout_rate
        self._label_namespace = label_namespace

        self.label_representation = nn.ModuleList()
        self.idx2namespace = {}
        for idx, namespace in enumerate(self._label_namespace):
            v = self.vocab._token_to_index[namespace]
            type_num = self.vocab.get_vocab_size(namespace)
            self.idx2namespace[idx] = namespace
            self.label_representation.append(nn.Linear(in_features=type_dim, out_features=type_num, bias=False))

        self._measure = MultiLayerF1Measure(label_namespace=label_namespace)
        self.loss_function = nn.MultiLabelSoftMarginLoss(reduction='mean')

        self._show = show

    def forward(self, context, mention, mention_char, labels=None, fine_labels=None, ultra_fine_labels=None) -> Dict:
        context_mask = get_text_field_mask(context)
        context_embedding = self._context_field_embedder(context)
        context_hidden = self._context_encoder(context_embedding)

        mention_embedding = self._mention_field_embedder(mention)
        mention_char_embedding = self._mention_char_field_embedder(mention_char)
        mention_output = torch.cat([mention_embedding.mean(dim=1), mention_char_embedding.mean(dim=1)], dim=1)

        # (batch_size, context_len)
        attn = self._context_attention(vector=mention_output, matrix=context_hidden, matrix_mask=context_mask)
        # (batch_size, context_dim)
        context_representation = (attn.unsqueeze(-1) * context_hidden).sum(dim=1)
        # (batch_size, context_dim + mention_dim)
        context_mention_representation = torch.cat([context_representation, mention_output], dim=1)
        pred_scores = {}
        for idx, representation in enumerate(self.label_representation):
            pred_scores[self.idx2namespace[idx]] = representation(context_mention_representation)
        forward_result = {'logits': pred_scores}

        loss = None
        all_pred = {}
        all_label = {}
        if labels is not None:
            select_idx = utils.get_select_idx(labels)
            if select_idx is not None:
                labels = torch.index_select(labels, 0, select_idx)
                forward_result['labels'] = labels
                pred = torch.index_select(pred_scores['labels'], 0, select_idx)
                loss = self.loss_function(pred, labels)
                all_pred['labels'] = pred
                all_label['labels'] = labels
        if fine_labels is not None:
            select_idx = utils.get_select_idx(fine_labels)
            if select_idx is not None:
                fine_labels = torch.index_select(fine_labels, 0, select_idx)
                forward_result['fine_labels'] = fine_labels
                pred = torch.index_select(pred_scores['fine_labels'], 0, select_idx)
                if loss is not None:
                    loss += self.loss_function(pred, fine_labels)
                else:
                    loss = self.loss_function(pred, fine_labels)
                all_pred['fine_labels'] = pred
                all_label['fine_labels'] = fine_labels
        if ultra_fine_labels is not None:
            select_idx = utils.get_select_idx(fine_labels)
            if select_idx is not None:
                ultra_fine_labels = torch.index_select(ultra_fine_labels, 0, select_idx)
                forward_result['ultra_fine_labels'] = ultra_fine_labels
                pred = torch.index_select(pred_scores['ultra_fine_labels'], 0, select_idx)
                if loss is not None:
                    loss += self.loss_function(pred, ultra_fine_labels)
                else:
                    loss = self.loss_function(pred, ultra_fine_labels)
                all_pred['ultra_fine_labels'] = pred
                all_label['ultra_fine_labels'] = ultra_fine_labels
        if loss is not None:
            forward_result['loss'] = loss
            self._measure(all_pred, all_label)

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
        # IDs for maximum values
        scores = {key: logit.cpu().data.numpy() for key, logit in output_dict['logits'].items()}
        # scores = {key: logit for key, logit in output_dict['logits'].items()}

        for key, score in scores.items():
            output_dict['pred_' + key] = []
            for instance in score:
                label_idx = np.where(instance > 0.5)[-1]
                output_dict['pred_' + key].append([self.vocab.get_token_from_index(x, namespace=key)
                                                       for x in label_idx])
                # output_dict['pred_label'][key].append([self.vocab.get_token_from_index(x, namespace=key)
                #                                   for x in label_idx])

        # output_dict['pred_label'] = output_dict['pred_label']

        return output_dict

