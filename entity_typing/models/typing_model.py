import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Dict, List

from allennlp.models import Model
from allennlp.data import Vocabulary
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder, Seq2VecEncoder, Attention
from allennlp.nn.util import get_text_field_mask

from entity_typing.multi_layer_measure import MultiLayerF1Measure
from entity_typing.modules.multi_layer_loss import MultiLayerLoss

logger = logging.getLogger(__name__)


@Model.register("typing_model")
class TypingModel(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 context_field_embedder: TextFieldEmbedder,  # bert_word & pos + attention
                 mention_field_embedder: TextFieldEmbedder,  # char + CNN & word + attention(pooling)
                 mention_char_field_embedder: TextFieldEmbedder,
                 context_attention: Attention,
                 mention_token_attention: Attention,
                 mention_attention: Attention,
                 context_encoder: Seq2SeqEncoder = None,
                 mention_encoder: Seq2VecEncoder = None,
                 mention_char_encoder: Seq2VecEncoder = None,
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
        self._mention_token_attention = mention_token_attention
        self._mention_attention = mention_attention
        self._context_encoder = context_encoder
        self._mention_encoder = mention_encoder
        self._mention_char_encoder = mention_char_encoder

        self._dropout = dropout_rate
        self._label_namespace = label_namespace

        self.mention_dropout = nn.Dropout(p=0.5)
        self.context_dropout = nn.Dropout(p=0.2)

        self.label_representation = nn.ModuleList()
        self.idx2namespace = {}
        for idx, namespace in enumerate(self._label_namespace):
            v = self.vocab._token_to_index[namespace]
            type_num = self.vocab.get_vocab_size(namespace)
            self.idx2namespace[idx] = namespace
            self.label_representation.append(nn.Linear(in_features=type_dim, out_features=type_num, bias=False))

        self._measure = MultiLayerF1Measure(label_namespace=label_namespace)
        self.loss_function = MultiLayerLoss(label_namespace=label_namespace)

        self.sigm = nn.Sigmoid()
        self._show = show

    def forward(self, context, mention, mention_char, labels=None, fine_labels=None, ultra_fine_labels=None) -> Dict:
        context_mask = get_text_field_mask(context)
        context_embedding = self.context_dropout(self._context_field_embedder(context))
        context_hidden = self._context_encoder(context_embedding, context_mask)

        mention_mask = get_text_field_mask(mention)
        mention_char_mask = get_text_field_mask(mention_char)
        mention_embedding = self.mention_dropout(self._mention_field_embedder(mention))
        mention_char_embedding = self.mention_dropout(self._mention_char_field_embedder(mention_char))
        mention_token_output = self._mention_encoder(mention_embedding, mention_mask)
        mention_char_output = self._mention_char_encoder(mention_char_embedding, mention_char_mask)

        mention_output, m_attn = self._mention_token_attention(mention_token_output, mention_mask)
        mention_output = torch.cat([mention_output, mention_char_output], dim=1)

        # (batch_size, context_len)
        attn = self._mention_attention(vector=mention_output, matrix=context_hidden, matrix_mask=context_mask)
        # (batch_size, context_dim)
        mention_output = (attn.unsqueeze(-1) * context_hidden).sum(dim=1)

        # # (batch_size, context_len)
        # attn = self._context_attention(vector=mention_output, matrix=context_hidden, matrix_mask=context_mask)
        # # (batch_size, context_dim)
        # context_representation = (attn.unsqueeze(-1) * context_hidden).sum(dim=1)

        # (batch_size, context_dim)
        context_representation, c_attn = self._context_attention(input_embed=context_hidden, mask=context_mask)

        # (batch_size, context_dim + mention_dim)
        context_mention_representation = torch.cat([context_representation, mention_output], dim=1)

        pred_scores = {}
        measure_scores = {}
        for idx, representation in enumerate(self.label_representation):
            logits = representation(context_mention_representation)
            pred_scores[self.idx2namespace[idx]] = logits
            measure_scores[self.idx2namespace[idx]] = logits
        forward_result = {'logits': pred_scores}

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
        scores = {key: logit.cpu().data.numpy() for key, logit in output_dict['logits'].items()}

        for key, score in scores.items():
            output_dict['pred_' + key] = []
            for instance in score:
                label_idx = np.where(instance > 0.5)[-1]
                output_dict['pred_' + key].append([self.vocab.get_token_from_index(x, namespace=key)
                                                       for x in label_idx])

        return output_dict

