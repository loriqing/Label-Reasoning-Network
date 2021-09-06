#!/usr/bin/env python
# -*- coding:utf-8 -*-

from allennlp.data import DatasetReader

from entity_typing import data_utils
from entity_typing import module_utils
from entity_typing.dataReader import BertMarkerTypingReader, BertSepTypingReader
from entity_typing.models import BertTypingModel
from entity_typing.train_envs.train_utils import train_model
from entity_typing.constant import NAME_SPACE_CONTEXT_TOKEN, NAME_SPACE_MENTION_TOKEN, \
    FEATURE_NAME_CONTEXT_TOKEN, FEATURE_NAME_MENTION_TOKEN


class BertTypingTrainEnv():
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
                                                                     token=False, pos_tag=False,
                                                                     char=False, transformer=True, mismatch=False)

        context_encoder, _ = module_utils.prepare_bert_encoder(model_name=args.transformer, input_size=0)

        concat_dim = context_encoder.get_output_dim()

        if 'ontonotes' in args.data_folder_path:
            label_namespace = ['labels']
        elif 'open_type' in args.data_folder_path:
            label_namespace = ['labels', 'fine_labels', 'ultra_fine_labels']
        else:
            label_namespace = None

        model = BertTypingModel(vocab=vocab,
                                context_field_embedder=context_field_embedder,
                                context_encoder=context_encoder,
                                dropout_rate=args.dropout,
                                type_dim=concat_dim,
                                label_namespace=label_namespace,
                                evaluation=args.evaluation,)
        return model

    @staticmethod
    def prepare_dataset_reader(args) -> DatasetReader:
        transformer = True if args.transformer is not None else False
        tokenizers = data_utils.prepare_tokenizer(args, token=False, char=False, transformer=transformer)
        indexers = data_utils.prepare_token_indexers(args, token=False, pos_tag=False, char=False,
                                                     transformer=transformer, mismatch=False)
        if args.bert_type == 'entity_marker':
            reader = BertMarkerTypingReader(lazy=args.lazy,
                                            tokenizers=tokenizers,
                                            indexers=indexers,
                                            max_sentence_length=args.max_sentence_length,
                                            test_ins=args.test_ins)
        elif args.bert_type == 'sep_mention':
            reader = BertSepTypingReader(lazy=args.lazy,
                                         tokenizers=tokenizers,
                                         indexers=indexers,
                                         max_sentence_length=args.max_sentence_length,
                                         test_ins=args.test_ins)
        return reader

    @staticmethod
    def train_model(args):
        reader = BertTypingTrainEnv.prepare_dataset_reader(args)
        data = data_utils.build_dataset(args, reader, args.data_folder_path, distant=args.distant)
        train_dataset, valid_dataset, test_dataset, vocab = data['train'], data['valid'], data['test'], data['vocab']
        data_utils.index_dataset(train_dataset, valid_dataset, test_dataset, vocab)
        train_loader, valid_loader, test_loader = data_utils.build_dataloader(args, train_dataset, valid_dataset,
                                                                              test_dataset)
        model = BertTypingTrainEnv.prepare_model(args, vocab)

        train_model(args, model, train_loader, valid_loader, test_loader)
