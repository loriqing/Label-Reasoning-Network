import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Dict, List

from allennlp.models import Model
from allennlp.data import Vocabulary
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder, Seq2VecEncoder
from allennlp.nn.util import get_text_field_mask

from entity_typing.multi_layer_measure import MultiLayerF1Measure
from entity_typing.modules import BipartiteMatchingLoss, SimilarLoss, MultiLayerLoss
from entity_typing import utils
from entity_typing.modules import Seq2KVDecoder, Seq2KVAttnDecoder, Seq2KVAttn2Decoder, Seq2SeqLoss
from entity_typing.constant import NAME_SPACE_SEQ_LABEL, FEATURE_NAME_SEQ_LABEL, EOS_SYMBOL, PADDING_TOKEN, \
    FEATURE_NAME_KEY, NAME_SPACE_KEY, FEATURE_NAME_VALUE, NAME_SPACE_VALUE

logger = logging.getLogger(__name__)


@Model.register("seq_kv_typing_model")
class SeqKVTypingModel(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 context_field_embedder: TextFieldEmbedder,  # bert
                 label_field_embedder: TextFieldEmbedder,
                 key_field_embedder: TextFieldEmbedder,  # pretrained word embedding
                 value_field_embedder: TextFieldEmbedder,  # pretrained word embedding
                 context_vec_encoder: Seq2VecEncoder = None,
                 context_seq_encoder: Seq2SeqEncoder = None,
                 dropout_rate: float = 0.3,
                 label_namespace: List[str] = ['labels'],
                 show: bool = False,
                 decoder_dropout: float = 0.5,
                 teaching_forcing_rate: float = 1,
                 decode_max_seq_len: int = 20,
                 loss_type: str = 'cross_entropy',
                 decoder_type: str = 'lstm',
                 evaluation: str = 'macro',
                 loss_lambda: float = 1,
                 loss_lambda2: float = 0.1,
                 detach: bool = False,
                 similar_thred: float = 0,
                 sparce_rate: float = 0.8,
                 ) -> None:
        super().__init__(vocab)
        self._context_field_embedder = context_field_embedder
        self._label_field_embedder = label_field_embedder
        self._key_field_embedder = key_field_embedder
        self._value_field_embedder = value_field_embedder
        self._context_vec_encoder = context_vec_encoder
        self._context_seq_encoder = context_seq_encoder

        # self._dropout = dropout_rate
        self._label_namespace = label_namespace
        self.mapping_dic = self.calculate_mapping_dic(vocab, label_namespace)
        self.eos_label_id = vocab.get_token_index(EOS_SYMBOL, namespace=NAME_SPACE_SEQ_LABEL)

        if decoder_type == 'lstm':
            self._decoder = Seq2KVAttnDecoder(
                vocab=vocab,
                label_field_embedder=self._label_field_embedder,
                label_embedding_dim=self._label_field_embedder.get_output_dim(),
                input_size=self._label_field_embedder.get_output_dim() * 1 + self._context_seq_encoder.get_output_dim(),
                decoder_hidden_size=self._context_vec_encoder.get_output_dim(),
                encoder_hidden_dim=self._context_seq_encoder.get_output_dim(),
                output_size=vocab.get_vocab_size(NAME_SPACE_SEQ_LABEL),
                dropout_rate=decoder_dropout,
                teaching_forcing_rate=teaching_forcing_rate,
                decode_max_seq_len=decode_max_seq_len,
                memory_dim=self._key_field_embedder.get_output_dim(),
                label_namespace=[NAME_SPACE_SEQ_LABEL, FEATURE_NAME_SEQ_LABEL],
                detach=detach,
                sparce_rate=sparce_rate,
            )
        if decoder_type == 'lstm2':
            self._decoder = Seq2KVAttn2Decoder(
                vocab=vocab,
                label_field_embedder=self._label_field_embedder,
                label_embedding_dim=self._label_field_embedder.get_output_dim(),
                input_size=self._label_field_embedder.get_output_dim() * 1 + self._context_seq_encoder.get_output_dim(),
                decoder_hidden_size=self._context_vec_encoder.get_output_dim(),
                encoder_hidden_dim=self._context_seq_encoder.get_output_dim(),
                output_size=vocab.get_vocab_size(NAME_SPACE_SEQ_LABEL),
                dropout_rate=decoder_dropout,
                teaching_forcing_rate=teaching_forcing_rate,
                decode_max_seq_len=decode_max_seq_len,
                memory_dim=self._key_field_embedder.get_output_dim(),
                label_namespace=[NAME_SPACE_SEQ_LABEL, FEATURE_NAME_SEQ_LABEL],
                detach=detach,
                sparce_rate=sparce_rate,
            )
        if decoder_type == 'lstm0':
            self._decoder = Seq2KVDecoder(
                vocab=vocab,
                label_field_embedder=self._label_field_embedder,
                label_embedding_dim=self._label_field_embedder.get_output_dim(),
                input_size=self._label_field_embedder.get_output_dim() * 1 + self._context_seq_encoder.get_output_dim(),
                decoder_hidden_size=self._context_vec_encoder.get_output_dim(),
                encoder_hidden_dim=self._context_seq_encoder.get_output_dim(),
                output_size=vocab.get_vocab_size(NAME_SPACE_SEQ_LABEL),
                dropout_rate=decoder_dropout,
                teaching_forcing_rate=teaching_forcing_rate,
                decode_max_seq_len=decode_max_seq_len,
                memory_dim=self._key_field_embedder.get_output_dim(),
                label_namespace=[NAME_SPACE_SEQ_LABEL, FEATURE_NAME_SEQ_LABEL],
                detach=detach,
                sparce_rate=sparce_rate,
            )
        self._measure = MultiLayerF1Measure(label_namespace=label_namespace, evaluation=evaluation)
        if loss_type == 'cross_entropy':
            self.loss_function = Seq2SeqLoss(smoothing=0)
        if loss_type == 'match':
            self.loss_function = BipartiteMatchingLoss()
        self.memory_loss = SimilarLoss(reduction='mean')
        self._loss_lambda = loss_lambda
        self._loss_lambda2 = loss_lambda2
        self.similar_thred = similar_thred
        self._dropout = nn.Dropout(dropout_rate)

        self._show = show

    def calculate_mapping_dic(self, vocab, label_namespace):
        mapping_dic = {}
        start, end = 0, 0
        for namespace in label_namespace:
            mapping = vocab._index_to_token[namespace]
            num_tokens = len(mapping)
            start_index = 1 if mapping[0] == PADDING_TOKEN else 0
            start = end
            end += num_tokens - start_index
            mapping_dic[namespace] = [start + 1, end + 1]
        mapping_dic['add_token'] = [end + 1, len(vocab._index_to_token[NAME_SPACE_SEQ_LABEL])]

        return mapping_dic

    def calculate_mem2seq_dic(self, vocab):
        seq_vocab = vocab._token_to_index[NAME_SPACE_SEQ_LABEL]
        memory_vocab = vocab._token_to_index[NAME_SPACE_VALUE]
        mem2seq_dic = {}
        for word in memory_vocab:
            mem2seq_dic[memory_vocab[word]] = seq_vocab[word]
        return mem2seq_dic

    def forward(self, context,
                seq_labels: Dict[str, Dict[str, torch.Tensor]] = None,
                memory_key: Dict[str, Dict[str, torch.Tensor]] = None, key_labels=None,
                memory_value: Dict[str, Dict[str, torch.Tensor]] = None, value_labels=None,
                labels=None, fine_labels=None, ultra_fine_labels=None, value2label=None,
                ) -> Dict:

        if memory_key is not None:
            # (batch_size, memory_key_len)
            memory_key_mask = get_text_field_mask(memory_key)
            # (batch_size, memory_value_len)
            memory_value_mask = get_text_field_mask(memory_value)
            # (batch_size, memory_key_len, memory_emb)
            memory_key_embedding = self._key_field_embedder(memory_key)
            # (batch_size, memory_len)
            key_labels = key_labels.masked_fill(~memory_key_mask, 0)

            # (batch_size, memory_value_len, memory_emb)
            memory_value_embedding = self._value_field_embedder(memory_value)
        else:
            memory_key_mask = None
            memory_value_mask = None
            memory_key_embedding = None
            memory_value_embedding = None

        # (batch_size, context_len)
        context_mask = get_text_field_mask(context)
        # (batch_size, seq_label_len)
        seq_label_mask = get_text_field_mask(seq_labels)
        # (batch_size)
        context_lengths = context_mask.sum(-1)
        # (batch_size, context_len, context_emb)
        context_embedding = self._dropout(self._context_field_embedder(context))    # DP1
        # (batch_size, context_len, encoder_dim)
        context_hidden = self._context_seq_encoder(context_embedding, context_mask)
        init_represent = self._context_vec_encoder(context_embedding, context_mask)

        # (batch_size, seq_labels_len, label_dim)
        labels_embedding = self._label_field_embedder(seq_labels) if seq_labels is not None else None

        output, decode_ids, key_similars, value_similars = self._decoder(encoded_hidden_embeddings=context_hidden,
                                                   hidden=init_represent.unsqueeze(0),
                                                   cell=init_represent.unsqueeze(0),
                                                   text_lengths=context_lengths,
                                                   encoder_mask=context_mask,
                                                   memory_key_mask=memory_key_mask,
                                                   memory_value_mask=memory_value_mask,
                                                   memory_key_embedding=memory_key_embedding,
                                                   memory_value_embedding=memory_value_embedding,
                                                   gold_label_ids=seq_labels[FEATURE_NAME_SEQ_LABEL]['tokens'],
                                                   gold_label_embeddings=labels_embedding,
                                                   seq_label_mask=seq_label_mask,
                                                   )
        gold_label_len = seq_label_mask.size(1)
        output = output[:, :gold_label_len, :]
        assert memory_value[FEATURE_NAME_VALUE]['tokens'].size() == value2label.size()
        pred_scores, measure_scores = utils.convert_memory_output(output, decode_ids, seq_label_mask, self.similar_thred,
                                                                  value_similars, self.mapping_dic, self.eos_label_id,
                                                                  value2label, self.training, self._show)
        forward_result = {'logits': pred_scores, 'decode_ids': decode_ids, 'measures': measure_scores}

        gold_labels = {"labels": labels}
        if fine_labels is not None:
            gold_labels["fine_labels"] = fine_labels
        if ultra_fine_labels is not None:
            gold_labels["ultra_fine_labels"] = ultra_fine_labels

        # loss = self.loss_function(measure_scores, gold_labels)
        loss = self.loss_function(outputs=output, gold_label_ids=seq_labels[FEATURE_NAME_SEQ_LABEL]['tokens'],
                                  mask=seq_label_mask)
        key_loss = self.memory_loss(similars=key_similars, gold_labels=key_labels, memory_mask=memory_key_mask)
        value_loss = self.memory_loss(similars=value_similars, gold_labels=value_labels, memory_mask=memory_value_mask)
        loss = loss + self._loss_lambda * key_loss + self._loss_lambda2 * value_loss
        if loss is not None:
            forward_result['loss'] = loss
        measure_scores.pop('add_token')
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
        # IDs for maximum values

        decode_ids = output_dict['decode_ids'].cpu().data.numpy()

        output_dict['pred_seq_labels'] = []
        for decode_id in decode_ids:
            seq_list = []
            for x in decode_id:
                if x == self.eos_label_id:
                    break
                seq_list.append(self.vocab.get_token_from_index(x, namespace=NAME_SPACE_SEQ_LABEL))
            output_dict['pred_seq_labels'].append(seq_list)

        scores = {key: logit.cpu().data.numpy() for key, logit in output_dict['measures'].items()}

        for key, score in scores.items():
            output_dict['pred_' + key] = []
            for instance in score:
                # label_idx = np.where(instance > 0.5)[-1]
                label_idx = np.where(instance >= 1)[-1]
                output_dict['pred_' + key].append([self.vocab.get_token_from_index(x, namespace=key)
                                                       for x in label_idx])

        return output_dict
