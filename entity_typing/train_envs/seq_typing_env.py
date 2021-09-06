#!/usr/bin/env python
# -*- coding:utf-8 -*-
from allennlp.data import DatasetReader

from entity_typing import data_utils
from entity_typing import module_utils
from entity_typing.dataReader import SeqSepTypingReader, SeqMarkerTypingReader, SeqTypingReader
from entity_typing.models import SeqTypingModel, SeqBertTypingModel, SeqBertTypingModelOnto, SeqBertTypingModelUF
from entity_typing.train_envs.train_utils import train_model
from entity_typing.constant import NAME_SPACE_CONTEXT_TOKEN, NAME_SPACE_MENTION_TOKEN, NAME_SPACE_SEQ_LABEL, \
    FEATURE_NAME_CONTEXT_TOKEN, FEATURE_NAME_MENTION_TOKEN, MEMORY_SYMBOL, UNK_SYMBOL, EOS_SYMBOL


class SeqTypingTrainEnv():
    @staticmethod
    def add_arguments(parser):
        group = parser.add_argument_group('seq_typing')

        group.add_argument("-bert-type", dest='bert_type', default='none', type=str,
                           help="choose from ['none', 'sep_mention', 'entity_marker', 'entity_marker_memory']")
        group.add_argument("-label-emb-size", dest='label_emb_size', default=100, type=int, help='Label Embedding Dim')
        group.add_argument("-label-emb", dest='label_emb', default=None, type=str,
                           help='you can use the pretrained embedding or random')
        group.add_argument("-decoder-dropout", dest='decoder_dropout', default=0.5, type=float, help='Dropout Rate')
        group.add_argument("-teaching-forcing-rate", dest='teaching_forcing_rate', default=0, type=float,
                           help='the Rate for Teacher Forcing, 1 is Full Teacher Forcing, 0 is No Teacher Forcing')
        group.add_argument("-loss-type", dest='loss_type', default='cross_entropy', type=str,
                           help="choose loss from ['cross_entropy', 'match'] ")
        group.add_argument("-decoder-type", dest='decoder_type', default='lstm', type=str,
                           help="choose from ['lstm', 'transformer', 'transformer2']")
        group.add_argument("-smoothing-rate", dest='smoothing_rate', default=0, type=float,
                           help='the Rate for Label Smoothing, 0 is No Smoothing')
        group.add_argument("-shuffle-num", dest='shuffle_num', default=0, type=int,
                           help='the number of shuffle')
        group.add_argument("-seq-type", dest='seq_type', default='sequence', type=str,
                           help="choose from ['sequence', 'layer']")
        # group.add_argument("-memory", dest='memory', action='store_true', default=False, help="set True to use memory")
        # group.add_argument("-memory-emb", dest='memory_emb', type=str, help="you can use the pretrained embedding")
        # group.add_argument("-memory-emb-size", dest='memory_emb_size', type=int, help="Memory embedding dimension")
        # group.add_argument("-loss-lambda", dest='loss_lambda', type=float, default=1, help="Memory loss weight")
        # group.add_argument("-detach", dest='detach', action='store_true', default=False)
        # group.add_argument("-similar-thred", dest='similar_thred', type=float, default=0)
        # group.add_argument("-value-file", dest='value_file', type=str, default=None)

    @staticmethod
    def prepare_model(args, vocab):
        if 'ontonotes' in args.data_folder_path:
            label_namespace = ['labels']
        elif 'open_type' in args.data_folder_path:
            label_namespace = ['labels', 'fine_labels', 'ultra_fine_labels']
        else:
            label_namespace = None

        if args.bert_type == 'none':
            context_field_embedder = module_utils.prepare_field_embedder(args, vocab=vocab,
                                                                         name_space=NAME_SPACE_CONTEXT_TOKEN,
                                                                         feature_name=FEATURE_NAME_CONTEXT_TOKEN,
                                                                         token=True, pos_tag=True,
                                                                         char=False, transformer=False)

            mention_field_embedder = module_utils.prepare_field_embedder(args, vocab=vocab,
                                                                         name_space=NAME_SPACE_MENTION_TOKEN,
                                                                         feature_name=FEATURE_NAME_MENTION_TOKEN,
                                                                         token=True, pos_tag=False,
                                                                         char=False, transformer=False)

            mention_char_field_embedder = module_utils.prepare_field_embedder(args, vocab=vocab,
                                                                              token=False, pos_tag=False,
                                                                              char=True, transformer=False)

            label_field_embedder = module_utils.prepare_label_embedder(args, vocab)

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

            model = SeqTypingModel(vocab=vocab,
                                   context_field_embedder=context_field_embedder,
                                   mention_field_embedder=mention_field_embedder,
                                   mention_char_field_embedder=mention_char_field_embedder,
                                   label_field_embedder=label_field_embedder,
                                   context_attention=context_attention,
                                   mention_token_attention=mention_token_attention,
                                   mention_attention=mention_attention,
                                   context_encoder=context_encoder,
                                   mention_encoder=mention_encoder,
                                   mention_char_encoder=mention_char_encoder,
                                   dropout_rate=args.dropout,
                                   label_namespace=label_namespace,
                                   teaching_forcing_rate=args.teaching_forcing_rate,
                                   loss_type=args.loss_type,
                                   decoder_type=args.decoder_type,
                                   evaluation=args.evaluation,
                                   )
        else:
            context_field_embedder = module_utils.prepare_field_embedder(args, vocab=vocab,
                                                                         name_space=NAME_SPACE_CONTEXT_TOKEN,
                                                                         feature_name=FEATURE_NAME_CONTEXT_TOKEN,
                                                                         token=False, pos_tag=False,
                                                                         char=False, transformer=True, mismatch=False)
            context_vec_encoder, context_seq_encoder = module_utils.prepare_bert_encoder(model_name=args.transformer, input_size=context_field_embedder.get_output_dim())

            label_field_embedder = module_utils.prepare_label_embedder(args, vocab)

            # model = SeqBertTypingModel(
            #     vocab=vocab,
            #     context_field_embedder=context_field_embedder,
            #     label_field_embedder=label_field_embedder,
            #     context_vec_encoder=context_vec_encoder,
            #     context_seq_encoder=context_seq_encoder,
            #     dropout_rate=args.dropout,
            #     label_namespace=label_namespace,
            #     decoder_dropout=args.decoder_dropout,
            #     teaching_forcing_rate=args.teaching_forcing_rate,
            #     loss_type=args.loss_type,
            #     decoder_type=args.decoder_type,
            #     evaluation=args.evaluation,
            # )
            if "ontonotes" in args.data_folder_path:
                model = SeqBertTypingModelOnto(
                    vocab=vocab,
                    context_field_embedder=context_field_embedder,
                    label_field_embedder=label_field_embedder,
                    context_vec_encoder=context_vec_encoder,
                    context_seq_encoder=context_seq_encoder,
                    dropout_rate=args.dropout,
                    label_namespace=label_namespace,
                    decoder_dropout=args.decoder_dropout,
                    teaching_forcing_rate=args.teaching_forcing_rate,
                    loss_type=args.loss_type,
                    decoder_type=args.decoder_type,
                    evaluation=args.evaluation,
                )
            else:
                model = SeqBertTypingModel(
                    vocab=vocab,
                    context_field_embedder=context_field_embedder,
                    label_field_embedder=label_field_embedder,
                    context_vec_encoder=context_vec_encoder,
                    context_seq_encoder=context_seq_encoder,
                    dropout_rate=args.dropout,
                    label_namespace=label_namespace,
                    decoder_dropout=args.decoder_dropout,
                    teaching_forcing_rate=args.teaching_forcing_rate,
                    loss_type=args.loss_type,
                    decoder_type=args.decoder_type,
                    evaluation=args.evaluation,
                )

        return model

    @staticmethod
    def prepare_dataset_reader(args) -> DatasetReader:
        if args.bert_type == 'none':
            transformer = False
        else:
            transformer = True
        label_indexers = data_utils.prepare_label_indexer(args)
        if args.bert_type == 'sep_mention':
            tokenizers = data_utils.prepare_tokenizer(args, token=False, char=False, transformer=True)
            indexers = data_utils.prepare_token_indexers(args, token=False, pos_tag=False, char=False,
                                                         transformer=transformer, mismatch=False)
            reader = SeqSepTypingReader(lazy=args.lazy,
                                        tokenizers=tokenizers,
                                        indexers=indexers,
                                        label_indexers=label_indexers,
                                        max_sentence_length=args.max_sentence_length,
                                        max_mention_length=args.max_mention_length,
                                        test_ins=args.test_ins,
                                        seq_type=args.seq_type,
                                        shuffle_num=args.shuffle_num,
                                        )
        if args.bert_type == 'entity_marker':
            tokenizers = data_utils.prepare_tokenizer(args, token=False, char=False, transformer=True)
            indexers = data_utils.prepare_token_indexers(args, token=False, pos_tag=False, char=False,
                                                         transformer=transformer, mismatch=False)
            reader = SeqMarkerTypingReader(lazy=args.lazy,
                                           tokenizers=tokenizers,
                                           indexers=indexers,
                                           label_indexers=label_indexers,
                                           max_sentence_length=args.max_sentence_length,
                                           max_mention_length=args.max_mention_length,
                                           test_ins=args.test_ins,
                                           seq_type=args.seq_type,
                                           shuffle_num=args.shuffle_num,
                                           )
        if args.bert_type == 'none':
            indexers = data_utils.prepare_token_indexers(args, token=True, pos_tag=True, char=True,
                                                         transformer=False)
            reader = SeqTypingReader(lazy=args.lazy,
                                     indexers=indexers,
                                     label_indexers=label_indexers,
                                     max_sentence_length=args.max_sentence_length,
                                     test_ins=args.test_ins,
                                     seq_type=args.seq_type,
                                     shuffle_num=args.shuffle_num,
                                     )
        return reader

    @staticmethod
    def train_model(args):
        reader = SeqTypingTrainEnv.prepare_dataset_reader(args)
        data = data_utils.build_dataset(args, reader, args.data_folder_path, distant=args.distant)
        train_dataset, valid_dataset, test_dataset, vocab = data['train'], data['valid'], data['test'], data['vocab']
        if 'ontonotes' in args.data_folder_path:
            label_namespace = ['labels']
        elif 'open_type' in args.data_folder_path:
            label_namespace = ['labels', 'fine_labels', 'ultra_fine_labels']
        else:
            label_namespace = None

        vocab, mapping_dic = data_utils.modify_vocab(vocab=vocab, from_namespace=label_namespace,
                                                     edit_namespace=NAME_SPACE_SEQ_LABEL)

        data_utils.index_dataset(train_dataset, valid_dataset, test_dataset, vocab)
        train_loader, valid_loader, test_loader = data_utils.build_dataloader(args, train_dataset, valid_dataset,
                                                                              test_dataset)
        model = SeqTypingTrainEnv.prepare_model(args, vocab)
        train_model(args, model, train_loader, valid_loader, test_loader)
