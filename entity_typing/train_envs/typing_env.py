#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os

from allennlp.data import DatasetReader

from entity_typing import data_utils
from entity_typing import module_utils
from entity_typing.dataReader import TypingReader, TypingDistantReader
from entity_typing.models import TypingModel
from entity_typing.train_envs.train_utils import train_model
from entity_typing.constant import NAME_SPACE_CONTEXT_TOKEN, NAME_SPACE_MENTION_TOKEN, \
    FEATURE_NAME_CONTEXT_TOKEN, FEATURE_NAME_MENTION_TOKEN


class TypingTrainEnv():
    @staticmethod
    def add_arguments(parser):
        group = parser.add_argument_group('bert_typing')

        group.add_argument("-bert-type", dest='bert_type', default='sep_mention', type=str,
                           help="choose from ['sep_mention', 'entity_marker']")

    @staticmethod
    def prepare_model(args, vocab):
        context_field_embedder = module_utils.prepare_field_embedder(args, vocab=vocab,
                                                                     name_space=NAME_SPACE_CONTEXT_TOKEN,
                                                                     feature_name=FEATURE_NAME_CONTEXT_TOKEN,
                                                                     token=True, pos_tag=True,
                                                                     char=False, transformer=True)

        mention_field_embedder = module_utils.prepare_field_embedder(args, vocab=vocab,
                                                                     name_space=NAME_SPACE_MENTION_TOKEN,
                                                                     feature_name=FEATURE_NAME_MENTION_TOKEN,
                                                                     token=True, pos_tag=False,
                                                                     char=False, transformer=False)

        mention_char_field_embedder = module_utils.prepare_field_embedder(args, vocab=vocab,
                                                                          token=False, pos_tag=False,
                                                                          char=True, transformer=False)

        if args.transformer is not None:
            args.context_encoder_type = 'none'

        context_encoder = module_utils.prepare_encoder(encoder_type=args.context_encoder_type,
                                          input_size=context_field_embedder.get_output_dim(),
                                          encoder_layer_num=args.context_encoder_layer,
                                          encoder_size=args.context_encoder_size,
                                          encoder_dropout=args.dropout,
                                          seq2seq=True)
        mention_encoder = module_utils.prepare_encoder(encoder_type=args.mention_encoder_type,
                                          input_size=mention_field_embedder.get_output_dim(),
                                          encoder_layer_num=args.mention_encoder_layer,
                                          encoder_size=args.mention_encoder_size,
                                          encoder_dropout=args.dropout,
                                          seq2seq=True,
                                          filter=args.mention_filter_size)
        mention_char_encoder = module_utils.prepare_encoder(encoder_type=args.mention_char_encoder_type,
                                                       input_size=mention_char_field_embedder.get_output_dim(),
                                                       encoder_layer_num=args.mention_char_encoder_layer,
                                                       encoder_size=args.mention_char_encoder_size,
                                                       encoder_dropout=args.dropout,
                                                       seq2seq=False,
                                                       filter=args.mention_char_filter_size)
        # context_attention = module_utils.prepare_attention(args,
        #                                       q_dim=mention_field_embedder.get_output_dim() + mention_char_field_embedder.get_output_dim(),
        #                                       k_dim=context_encoder.get_output_dim())
        context_attention = module_utils.prepare_mlp_attention(args,
                                                               output_dim=context_encoder.get_output_dim(),
                                                               hidden_dim=context_encoder.get_output_dim())
        mention_token_attention = module_utils.prepare_mlp_attention(args,
                                                               output_dim=mention_encoder.get_output_dim(),
                                                               hidden_dim=mention_encoder.get_output_dim())
        mention_attention = module_utils.prepare_attention(args,
                                                           q_dim=mention_encoder.get_output_dim() +
                                                                 mention_char_encoder.get_output_dim(),
                                                           k_dim=context_encoder.get_output_dim())
        # concat_dim = context_encoder.get_output_dim() + mention_encoder.get_output_dim() + mention_char_encoder.get_output_dim()
        concat_dim = context_encoder.get_output_dim() * 2
        if 'ontonotes' in args.data_folder_path:
            label_namespace = ['labels']
        elif 'open_type' in args.data_folder_path:
            label_namespace = ['labels', 'fine_labels', 'ultra_fine_labels']
        else:
            label_namespace = None
        model = TypingModel(vocab=vocab,
                            context_field_embedder=context_field_embedder,
                            mention_field_embedder=mention_field_embedder,
                            mention_char_field_embedder=mention_char_field_embedder,
                            context_attention=context_attention,
                            mention_token_attention=mention_token_attention,
                            mention_attention=mention_attention,
                            context_encoder=context_encoder,
                            mention_encoder=mention_encoder,
                            mention_char_encoder=mention_char_encoder,
                            dropout_rate=args.dropout,
                            type_dim=concat_dim,
                            label_namespace=label_namespace)
        return model

    @staticmethod
    def prepare_dataset_reader(args) -> DatasetReader:
        transformer = True if args.transformer is not None else False
        indexers = data_utils.prepare_token_indexers(args, token=True, pos_tag=True, char=True, transformer=transformer)
        if args.distant:
            if 'open_type' in args.data_folder_path:
                reader = TypingDistantReader(lazy=args.lazy,
                                             indexers=indexers,
                                             max_sentence_length=args.max_sentence_length,
                                             test_ins=args.test_ins,
                                             distant=args.distant,
                                             head=args.distant,
                                             el=args.distant,
                                             dir_path=args.data_folder_path,
                                             batch_size=args.batch)
            else:
                reader = TypingDistantReader(lazy=args.lazy,
                                             indexers=indexers,
                                             max_sentence_length=args.max_sentence_length,
                                             test_ins=args.test_ins,
                                             distant=args.distant,
                                             dir_path=args.data_folder_path,
                                             batch_size=args.batch)
        else:
            reader = TypingReader(lazy=args.lazy,
                                  indexers=indexers,
                                  max_sentence_length=args.max_sentence_length,
                                  test_ins=args.test_ins,)
        return reader

    @staticmethod
    def train_model(args):
        reader = TypingTrainEnv.prepare_dataset_reader(args)
        data = data_utils.build_dataset(args, reader, args.data_folder_path, distant=args.distant)
        train_dataset, valid_dataset, test_dataset, vocab = data['train'], data['valid'], data['test'], data['vocab']
        data_utils.index_dataset(train_dataset, valid_dataset, test_dataset, vocab)
        train_loader, valid_loader, test_loader = data_utils.build_dataloader(args, train_dataset, valid_dataset,
                                                                              test_dataset)
        model = TypingTrainEnv.prepare_model(args, vocab)

        train_model(args, model, train_loader, valid_loader, test_loader)
