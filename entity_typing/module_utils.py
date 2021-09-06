import torch
import torch.nn as nn
import logging
from typing import Dict

from allennlp.common import Params
from allennlp.modules import Embedding, TokenEmbedder, Attention, MatrixAttention
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper, PassThroughEncoder
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper, CnnEncoder, BertPooler
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import PretrainedTransformerMismatchedEmbedder, PretrainedTransformerEmbedder, \
    TokenCharactersEncoder, ElmoTokenEmbedder
from allennlp.modules.attention import BilinearAttention, AdditiveAttention, DotProductAttention, LinearAttention
from allennlp.modules.matrix_attention import DotProductMatrixAttention, BilinearMatrixAttention, LinearMatrixAttention
from allennlp.nn import Activation

from entity_typing.modules import SelfAttentiveSum
from entity_typing.constant import NAME_SPACE_CHARACTER, NAME_SPACE_POS_TAG, NAME_SPACE_SEQ_LABEL,\
     FEATURE_NAME_TRANSFORMER, FEATURE_NAME_POS_TAG, FEATURE_NAME_CHARACTER, FEATURE_NAME_SEQ_LABEL,\
     NAME_SPACE_MEMORY, FEATURE_NAME_MEMORY, NAME_SPACE_KEY, FEATURE_NAME_KEY, NAME_SPACE_VALUE, FEATURE_NAME_VALUE, FEATURE_NAME_ELMO

logger = logging.getLogger(__name__)


def prepare_encoder(encoder_type, input_size, encoder_layer_num,
                    encoder_size=300, encoder_dropout=0.,
                    seq2seq=True, filter=None
                    ):
    if seq2seq:
        wrapper = PytorchSeq2SeqWrapper
    else:
        wrapper = PytorchSeq2VecWrapper

    print('prepare_encoder: encoder_type ', encoder_type.lower())

    if encoder_type.lower() == 'lstm':
        return wrapper(nn.LSTM(input_size=input_size,
                               hidden_size=encoder_size,
                               num_layers=encoder_layer_num,
                               bidirectional=True,
                               batch_first=True,
                               bias=True,
                               dropout=encoder_dropout,
                               )
                       )
    elif encoder_type.lower() == 'gru':
        return wrapper(nn.GRU(input_size=input_size,
                              hidden_size=encoder_size,
                              num_layers=encoder_layer_num,
                              bidirectional=True,
                              batch_first=True,
                              bias=True,
                              dropout=encoder_dropout,
                              )
                       )
    elif encoder_type.lower() == 'cnn':
        return CnnEncoder(embedding_dim=input_size, num_filters=input_size, ngram_filter_sizes=(filter))
    elif encoder_type.lower() in ['pass_through', 'none'] and seq2seq:
        return PassThroughEncoder(input_dim=input_size, )
    else:
        raise NotImplementedError('%s is not implemented' % encoder_type)


def prepare_bert_encoder(model_name, input_size):
    return BertPooler(pretrained_model=model_name), PassThroughEncoder(input_dim=input_size, )


def prepare_label_embedder(args, vocab) -> BasicTextFieldEmbedder:
    label_embedder: Dict[str, TokenEmbedder] = {}

    if args.label_emb_size > 0:
        params_dict = {
            'embedding_dim': args.label_emb_size,
            'trainable': True,
            'pretrained_file': args.label_emb,
            'vocab_namespace': NAME_SPACE_SEQ_LABEL,
        }
        logging.info("Load Label Embedding from %s" % params_dict['pretrained_file'])
        # if args.token_emb == 'random':
        #     params_dict.pop('pretrained_file')

        label_embedding_params = Params(params_dict)
        label_embedding = Embedding.from_params(
            vocab=vocab,
            params=label_embedding_params,
        )

        label_embedder[FEATURE_NAME_SEQ_LABEL] = label_embedding

    label_field_embedder = BasicTextFieldEmbedder(
        token_embedders=label_embedder
    )
    return label_field_embedder


def prepare_memory_embedder(args, vocab) -> BasicTextFieldEmbedder:
    memory_embedder: Dict[str, TokenEmbedder] = {}

    if args.memory_emb_size > 0:
        params_dict = {
            'embedding_dim': args.memory_emb_size,
            'trainable': False,
            'pretrained_file': args.memory_emb,
            'vocab_namespace': NAME_SPACE_MEMORY,
        }
        logging.info("Load Memory Embedding from %s" % params_dict['pretrained_file'])
        # if args.token_emb == 'random':
        #     params_dict.pop('pretrained_file')

        memory_embedding_params = Params(params_dict)
        memory_embedding = Embedding.from_params(
            vocab=vocab,
            params=memory_embedding_params,
        )

        memory_embedder[FEATURE_NAME_MEMORY] = memory_embedding

    memory_field_embedder = BasicTextFieldEmbedder(
        token_embedders=memory_embedder
    )
    return memory_field_embedder


def prepare_key_embedder(args, vocab) -> BasicTextFieldEmbedder:
    memory_embedder: Dict[str, TokenEmbedder] = {}

    if args.memory_emb_size > 0:
        params_dict = {
            'embedding_dim': args.memory_emb_size,
            'trainable': False,
            'pretrained_file': args.memory_emb,
            'vocab_namespace': NAME_SPACE_KEY,
        }
        logging.info("Load Memory Embedding from %s" % params_dict['pretrained_file'])
        # if args.token_emb == 'random':
        #     params_dict.pop('pretrained_file')

        memory_embedding_params = Params(params_dict)
        memory_embedding = Embedding.from_params(
            vocab=vocab,
            params=memory_embedding_params,
        )

        memory_embedder[FEATURE_NAME_KEY] = memory_embedding

    memory_field_embedder = BasicTextFieldEmbedder(
        token_embedders=memory_embedder
    )
    return memory_field_embedder


def prepare_value_embedder(args, vocab) -> BasicTextFieldEmbedder:
    memory_embedder: Dict[str, TokenEmbedder] = {}

    if args.memory_emb_size > 0:
        params_dict = {
            'embedding_dim': args.memory_emb_size,
            'trainable': False,
            'pretrained_file': args.memory_emb,
            'vocab_namespace': NAME_SPACE_VALUE,
        }
        logging.info("Load Memory Embedding from %s" % params_dict['pretrained_file'])
        # if args.token_emb == 'random':
        #     params_dict.pop('pretrained_file')

        memory_embedding_params = Params(params_dict)
        memory_embedding = Embedding.from_params(
            vocab=vocab,
            params=memory_embedding_params,
        )

        memory_embedder[FEATURE_NAME_VALUE] = memory_embedding

    memory_field_embedder = BasicTextFieldEmbedder(
        token_embedders=memory_embedder
    )
    return memory_field_embedder


def prepare_field_embedder(args, vocab=None, name_space=None, feature_name=None,
                           token=False, pos_tag=False, char=False, transformer=False, mismatch=True, elmo=False):
    logger.info(vocab)

    token_embedders: Dict[str, TokenEmbedder] = {}

    if args.token_emb_size > 0 and token:
        assert name_space is not None and name_space is not None
        # Load Token Embedding
        params_dict = {
            'embedding_dim': args.token_emb_size,
            'trainable': True,
            'pretrained_file': args.token_emb,
            'vocab_namespace': name_space,
        }

        logging.info("Load Word Embedding from %s" % params_dict['pretrained_file'])
        if args.token_emb == 'random':
            params_dict.pop('pretrained_file')

        token_embedding_params = Params(params_dict)
        token_embedding = Embedding.from_params(
            vocab=vocab,
            params=token_embedding_params,
        )

        token_embedders[feature_name] = token_embedding

    if args.pos_emb_size > 0 and pos_tag:
        pos_embedding = Embedding.from_params(
            vocab=vocab,
            params=Params({
                'embedding_dim': args.pos_emb_size,
                'trainable': True,
                'vocab_namespace': NAME_SPACE_POS_TAG
            }),
        )

        token_embedders[FEATURE_NAME_POS_TAG] = pos_embedding

    if args.char_emb_size > 0 and char:
        char_encoder = prepare_encoder(
            encoder_type=args.char_encoder_type,
            input_size=args.char_emb_size,
            encoder_layer_num=1,
            encoder_size=args.char_emb_size // 2,
            encoder_dropout=0.,
            seq2seq=False,
            filter=[2]
        )

        char_token_encoder = TokenCharactersEncoder(
            embedding=Embedding.from_params(
                vocab=vocab,
                params=Params({
                    "embedding_dim": args.char_emb_size,
                    "trainable": True,
                    "vocab_namespace": NAME_SPACE_CHARACTER
                })
            ),
            encoder=char_encoder
        )

        token_embedders[FEATURE_NAME_CHARACTER] = char_token_encoder

    if args.transformer and transformer:
        logger.info("Load Transformer ...")

        if not mismatch:
            transformer_embedding = PretrainedTransformerEmbedder(
                model_name=args.transformer,
                train_parameters=args.transformer_require_grad,
            )
        elif mismatch:
            transformer_embedding = PretrainedTransformerMismatchedEmbedder(
                model_name=args.transformer,
                max_length=args.transformer_max_length,
                train_parameters=args.transformer_require_grad,
            )

        token_embedders[FEATURE_NAME_TRANSFORMER] = transformer_embedding

    # if args.elmo and elmo:
    if elmo:
        logger.info("Load ELMo ...")
        elmo_embedding = ElmoTokenEmbedder(
            options_file=args.elmo + "options.json",
            weight_file=args.elmo + "weights.hdf5",
            do_layer_norm=False,
            dropout=0.5,
            requires_grad=args.elmo_require_grad,
        )
        token_embedders[FEATURE_NAME_ELMO] = elmo_embedding

    assert len(token_embedders) > 0

    text_field_embedder = BasicTextFieldEmbedder(
        token_embedders=token_embedders,
    )

    return text_field_embedder


def prepare_attention(args, q_dim, k_dim):
    activation = Activation.by_name(args.activation_type)()
    if args.attention_type == 'dot':
        assert q_dim == k_dim, "{0} != {1}".format(q_dim, k_dim)
        attention = DotProductAttention()
    elif args.attention_type == 'add':
        attention = AdditiveAttention(vector_dim=q_dim, matrix_dim=k_dim, normalize=True)
    elif args.attention_type == 'linear':
        attention = LinearAttention(tensor_1_dim=q_dim, tensor_2_dim=k_dim, normalize=True, activation=activation)
    elif args.attention_type == 'bilinear':
        attention = BilinearAttention(vector_dim=q_dim, matrix_dim=k_dim, normalize=True, activation=activation)
    else:
        raise ValueError('Invalid attention type')

    return attention


def prepare_mlp_attention(args, output_dim, hidden_dim):
    activation = Activation.by_name(args.activation_type)()
    attention = SelfAttentiveSum(output_dim=output_dim, hidden_dim=hidden_dim, activation=activation)
    return attention


if __name__ == "__main__":
    pass