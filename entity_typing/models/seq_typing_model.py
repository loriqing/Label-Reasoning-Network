import torch
import numpy as np
import logging
from typing import Dict, List

from allennlp.models import Model
from allennlp.data import Vocabulary
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder, Seq2VecEncoder, Attention
from allennlp.nn.util import get_text_field_mask

from entity_typing.multi_layer_measure import MultiLayerF1Measure
from entity_typing.modules.multi_layer_loss import MultiLayerLoss
from entity_typing.modules.bitpartite_matching_loss import BipartiteMatchingLoss
from entity_typing import utils
from entity_typing.modules import Seq2SeqDecoder, Seq2SeqLoss, Seq2SeqTransformerDecoder
from entity_typing.constant import NAME_SPACE_SEQ_LABEL, FEATURE_NAME_SEQ_LABEL, EOS_SYMBOL, PADDING_TOKEN

logger = logging.getLogger(__name__)


@Model.register("seq_typing_model")
class SeqTypingModel(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 context_field_embedder: TextFieldEmbedder,  # bert_word & pos + attention
                 mention_field_embedder: TextFieldEmbedder,  # char + CNN & word + attention(pooling)
                 mention_char_field_embedder: TextFieldEmbedder,
                 label_field_embedder: TextFieldEmbedder,
                 context_attention: Attention,
                 mention_token_attention: Attention,
                 mention_attention: Attention,
                 context_encoder: Seq2SeqEncoder = None,
                 mention_encoder: Seq2VecEncoder = None,
                 mention_char_encoder: Seq2VecEncoder = None,
                 dropout_rate: float = 0.3,
                 label_namespace: List[str] = ['labels'],
                 show: bool = False,
                 decoder_dropout: float = 0.5,
                 teaching_forcing_rate: float = 1,
                 decode_max_seq_len: int = 20,
                 smoothing: float = 0,
                 loss_type: str = 'cross_entropy',
                 decoder_type: str = 'lstm',
                 evaluation: str = 'macro',
                 ) -> None:
        super().__init__(vocab)
        self._context_field_embedder = context_field_embedder
        self._mention_field_embedder = mention_field_embedder
        self._mention_char_field_embedder = mention_char_field_embedder
        self._label_field_embedder = label_field_embedder
        self._context_attention = context_attention
        self._mention_token_attention = mention_token_attention
        self._mention_attention = mention_attention
        self._context_encoder = context_encoder
        self._mention_encoder = mention_encoder
        self._mention_char_encoder = mention_char_encoder

        self._dropout = dropout_rate
        self._label_namespace = label_namespace
        self.mapping_dic = self.calculate_mapping_dic(vocab, label_namespace)
        self.eos_label_id = vocab.get_token_index(EOS_SYMBOL, namespace=NAME_SPACE_SEQ_LABEL)

        if decoder_type == 'lstm':
            self._decoder = Seq2SeqDecoder(
                vocab=vocab,
                label_field_embedder=self._label_field_embedder,
                label_embedding_dim=self._label_field_embedder.get_output_dim(),
                input_size=self._label_field_embedder.get_output_dim() * 1 + self._context_encoder.get_output_dim(),
                decoder_hidden_size=self._context_encoder.get_output_dim() * 2,
                encoder_hidden_dim=self._context_encoder.get_output_dim() * 2,
                output_size=vocab.get_vocab_size(NAME_SPACE_SEQ_LABEL),
                dropout_rate=decoder_dropout,
                teaching_forcing_rate=teaching_forcing_rate,
                decode_max_seq_len=decode_max_seq_len,
                label_namespace=[NAME_SPACE_SEQ_LABEL, FEATURE_NAME_SEQ_LABEL],
            )
        if decoder_type == 'transformer':
            self._decoder = Seq2SeqTransformerDecoder(
                vocab=vocab,
                label_field_embedder=self._label_field_embedder,
                label_embedding_dim=self._label_field_embedder.get_output_dim(),
                input_size=self._label_field_embedder.get_output_dim() * 1 + self._context_encoder.get_output_dim(),
                decoder_hidden_size=self._context_encoder.get_output_dim(),
                encoder_hidden_dim=self._context_encoder.get_output_dim(),
                output_size=vocab.get_vocab_size(NAME_SPACE_SEQ_LABEL),
                dropout_rate=decoder_dropout,
                teaching_forcing_rate=teaching_forcing_rate,
                decode_max_seq_len=decode_max_seq_len,
                label_namespace=[NAME_SPACE_SEQ_LABEL, FEATURE_NAME_SEQ_LABEL],
            )

        self._measure = MultiLayerF1Measure(label_namespace=label_namespace, evaluation=evaluation)
        self.loss_function = MultiLayerLoss(label_namespace=label_namespace)

        if loss_type == 'cross_entropy':
            self.loss_function = Seq2SeqLoss(smoothing=0)
        if loss_type == 'match':
            self.loss_function = BipartiteMatchingLoss()

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

        # print(mapping_dic)

        return mapping_dic

    def forward(self, context, mention, mention_char,
                seq_labels: Dict[str, Dict[str, torch.Tensor]] = None,
                labels=None, fine_labels=None, ultra_fine_labels=None) -> Dict:

        # (batch_size, context_len)
        context_mask = get_text_field_mask(context)
        # (batch_size, seq_label_len)
        seq_label_mask = get_text_field_mask(seq_labels)
        # (batch_size)
        context_lengths = context_mask.sum(-1)
        # (batch_size, context_len, context_emb)
        context_embedding = self._context_field_embedder(context)
        # (batch_size, context_len, encoder_dim)
        context_hidden = self._context_encoder(context_embedding, context_mask)

        # (batch_size, mention_len)
        mention_mask = get_text_field_mask(mention)
        # (batch_size, mention_char_len)
        mention_char_mask = get_text_field_mask(mention_char)
        # (batch_size, mention_dim)
        mention_embedding = self._mention_field_embedder(mention)
        # (batch_size, mention_char_dim)
        mention_char_embedding = self._mention_char_field_embedder(mention_char)

        # 1mention char+CNN,token+LSTM+self-a,concat
        mention_token_output = self._mention_encoder(mention_embedding, mention_mask)
        mention_char_output = self._mention_char_encoder(mention_char_embedding, mention_char_mask)
        mention_output, m_attn = self._mention_token_attention(mention_token_output, mention_mask)
        mention_output = torch.cat([mention_output, mention_char_output], dim=1)

        # # (batch_size, mention_dim + mention_char_dim)
        # mention_output = torch.cat([utils.mask_average(mention_embedding, mention_mask),
        #                             utils.mask_average(mention_char_embedding, mention_char_mask)],
        #                            dim=1)

        # 3mention+context attention -》mention
        # (batch_size, context_len)
        attn = self._mention_attention(vector=mention_output, matrix=context_hidden, matrix_mask=context_mask)
        # (batch_size, context_dim)
        mention_output = (attn.unsqueeze(-1) * context_hidden).sum(dim=1)

        # # 1mention-》context attention-》context
        # # (batch_size, context_len)
        # attn = self._context_attention(vector=mention_output, matrix=context_hidden, matrix_mask=context_mask)
        # # (batch_size, context_dim)
        # context_representation = (attn.unsqueeze(-1) * context_hidden).sum(dim=1)

        # 2context self-a -》context
        # (batch_size, context_dim)
        context_representation, c_attn = self._context_attention(input_embed=context_hidden, mask=context_mask)

        # (batch_size, context_dim + mention_dim)
        init_represent = torch.cat([context_representation, mention_output], dim=1)

        # (batch_size, seq_labels_len, label_dim)
        labels_embedding = self._label_field_embedder(seq_labels) if seq_labels is not None else None

        # (batch_size, decode_len, label_num), (batch_size, decode_len)
        output, decode_ids = self._decoder(encoded_hidden_embeddings=context_hidden,
                                           hidden=init_represent.unsqueeze(0),
                                           cell=init_represent.unsqueeze(0),
                                           text_lengths=context_lengths,
                                           encoder_mask=context_mask,
                                           gold_label_ids=seq_labels[FEATURE_NAME_SEQ_LABEL]['tokens'],
                                           gold_label_embeddings=labels_embedding,
                                           )
        gold_label_len = seq_label_mask.size(1)
        output = output[:, :gold_label_len, :]
        pred_scores, measure_scores = utils.convert_output(output, decode_ids, seq_label_mask,
                                                           self.mapping_dic, self.eos_label_id,
                                                           self.training)
        forward_result = {'logits': pred_scores, 'decode_ids': decode_ids, 'measures': measure_scores}

        gold_labels = {"labels": labels}
        if fine_labels is not None:
            gold_labels["fine_labels"] = fine_labels
        if ultra_fine_labels is not None:
            gold_labels["ultra_fine_labels"] = ultra_fine_labels

        loss = self.loss_function(outputs=output, gold_label_ids=seq_labels[FEATURE_NAME_SEQ_LABEL]['tokens'], mask=seq_label_mask)

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
        output_dict['pred_seq_labels'] = [[self.vocab.get_token_from_index(x, namespace=NAME_SPACE_SEQ_LABEL)
                                          for x in decode_id if x!= self.eos_label_id] for decode_id in decode_ids]

        scores = {key: logit.cpu().data.numpy() for key, logit in output_dict['measures'].items()}

        for key, score in scores.items():
            output_dict['pred_' + key] = []
            for instance in score:
                label_idx = np.where(instance == 1)[-1]
                output_dict['pred_' + key].append([self.vocab.get_token_from_index(x, namespace=key)
                                                       for x in label_idx])

        return output_dict
