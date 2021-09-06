import os
import json
import torch
import codecs
import random
import argparse
from typing import Tuple, Dict, List

import allennlp
from allennlp.data import Vocabulary, DataLoader, Tokenizer, TokenIndexer
from allennlp.data.tokenizers import WhitespaceTokenizer, CharacterTokenizer, PretrainedTransformerTokenizer
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenCharactersIndexer, PretrainedTransformerIndexer, \
    PretrainedTransformerMismatchedIndexer, ELMoTokenCharactersIndexer
from allennlp.data.samplers import BucketBatchSampler

from entity_typing import arg_utils
from entity_typing.constant import NAME_SPACE_CONTEXT_TOKEN, NAME_SPACE_MENTION_TOKEN, NAME_SPACE_CHARACTER, NAME_SPACE_POS_TAG, NAME_SPACE_SEQ_LABEL,\
    FEATURE_NAME_MENTION_TOKEN, FEATURE_NAME_CONTEXT_TOKEN, FEATURE_NAME_CHARACTER, FEATURE_NAME_POS_TAG, \
    FEATURE_NAME_TRANSFORMER, FEATURE_NAME_ELMO, FEATURE_NAME_SEQ_LABEL, EOS_SYMBOL, PADDING_TOKEN, UNK_SYMBOL, \
    FEATURE_NAME_MEMORY, NAME_SPACE_MEMORY, MEMORY_SYMBOL, NAME_SPACE_KEY, FEATURE_NAME_KEY, NAME_SPACE_VALUE, FEATURE_NAME_VALUE, NAME_SPACE_ELMO


def prepare_tokenizer(args, token=True, char=False, transformer=False, elmo=False) -> Dict[str, Tokenizer]:
    tokenizers: Dict[str, Tokenizer] = {}
    if token:
        tokenizers[FEATURE_NAME_MENTION_TOKEN] = WhitespaceTokenizer()
        tokenizers[FEATURE_NAME_CONTEXT_TOKEN] = WhitespaceTokenizer()
    if char:
        tokenizers[FEATURE_NAME_CHARACTER] = CharacterTokenizer()
    if transformer:
        tokenizers[FEATURE_NAME_TRANSFORMER] = PretrainedTransformerTokenizer(model_name=args.transformer, add_special_tokens=False)
    if elmo:
        tokenizers[FEATURE_NAME_ELMO] = WhitespaceTokenizer()

    return tokenizers


def process_line(line, mem=False, thred=0.1):
    # id, sentence, context, pos, mention, mention_char
    data = json.loads(line)

    # split by space
    out = {'context': data['context'].strip().split(' '), 'pos': data['pos'].strip().split(' '),
           'mention': data['mention'].strip().split(' '), 'mention_char': [c for c in data['mention']],
           'labels': {}, 'seq_labels': data['seq_labels'],
           'sentence': data['sentence'].strip().split(' '),
           'sentence_entity': data['sentence_entity'].strip().split(' '),
           'pos_entity': data['pos_entity'].strip().split(' '),
           'left_sentence': data['left_sentence'].strip().split(' '),
           'right_sentence': data['right_sentence'].strip().split(' '),}
    assert len(out['context']) == len(out['pos'])

    for key in data.keys():
        if 'labels' in key and "seq" not in key:
            out['labels'][key] = data[key]

    if mem:
        memory_list = []
        if 'synthetic_memory' in data.keys():  # train
            memory_list = data['synthetic_memory']
        elif 'mention_memory' in data:  # memory, _original
            for label, score in zip(data['bert_memory'], data['bert_scores']):
                if score > thred:  # 0.1, bert score
                    memory_list.append(label)
            memory_list = memory_list + data['mention_memory']
            # memory_list = memory_list + data['sentence_memory']
        else:  # mapping_memory
            for label, score in zip(data['memory'], data['memory_scores']):
                if score > thred:  # 0.9, glove mapping score
                    memory_list.append(label)

        memory = []
        gold_memory = []
        for word in memory_list:
            # 去重
            if word in memory: continue
            else: memory.append(word)

            if word in data['seq_labels']:
                gold_memory.append(1)
            else:
                gold_memory.append(-1)
        if len(memory) == 0:
            memory = [MEMORY_SYMBOL]
            gold_memory.append(-1)

        out['memory'] = memory
        out['gold_memory'] = gold_memory

    return out


def process_kv_line(line, value_list, thred=0.1):
    data = json.loads(line)

    # split by space
    out = {'context': data['context'].strip().split(' '), 'pos': data['pos'].strip().split(' '),
           'mention': data['mention'].strip().split(' '), 'mention_char': [c for c in data['mention']],
           'labels': {}, 'seq_labels': data['seq_labels'],
           'sentence': data['sentence'].strip().split(' '),
           'sentence_entity': data['sentence_entity'].strip().split(' '),
           'pos_entity': data['pos_entity'].strip().split(' '),
           'left_sentence': data['left_sentence'].strip().split(' '),
           'right_sentence': data['right_sentence'].strip().split(' '),}
    assert len(out['context']) == len(out['pos'])

    for key in data.keys():
        if 'labels' in key and "seq" not in key:
            out['labels'][key] = data[key]

    key_list = []
    if 'synthetic_memory' in data.keys():  # open_type_synthetic : train
        key_list = data['synthetic_memory']
    elif 'mention_memory' in data:  # _memory, _original, _original_lemma, _original_lemma_synthetic
        for label, score in zip(data['bert_memory'], data['bert_scores']):
            if score > thred:  # 0.1, bert score
                key_list.append(label)
        key_list = key_list + data['mention_memory']
    else:  # _original_mapping
        for label, score in zip(data['memory'], data['memory_scores']):
            if score > thred:  # 0.9, glove mapping score
                key_list.append(label)

    key, value, gold_key, gold_value = [], [], [], []
    for word in key_list:
        if word in key: continue
        else: key.append(word)

        if word in data['seq_labels']:
            gold_key.append(1)
        else:
            gold_key.append(-1)
    if len(key) == 0:
        key = [MEMORY_SYMBOL]
        gold_key.append(-1)

    for label in value_list:
        if label in data['seq_labels']:
            gold_value.append(1)
        else:
            gold_value.append(-1)

    out['memory_key'] = key
    out['gold_key'] = gold_key
    out['memory_value'] = value_list
    out['gold_value'] = gold_value

    return out


def process_onto_kv_line(line, value_list, thred=0.1):
    data = json.loads(line)

    # split by space
    out = {'context': data['context'].strip().split(' '), 'pos': data['pos'].strip().split(' '),
           'mention': data['mention'].strip().split(' '), 'mention_char': [c for c in data['mention']],
           'labels': {}, 'seq_labels': data['seq_labels'],
           'sentence': data['sentence'].strip().split(' '),
           'sentence_entity': data['sentence_entity'].strip().split(' '),
           'pos_entity': data['pos_entity'].strip().split(' '),
           'left_sentence': data['left_sentence'].strip().split(' '),
           'right_sentence': data['right_sentence'].strip().split(' '),}
    assert len(out['context']) == len(out['pos'])

    for key in data.keys():
        if 'labels' in key and "seq" not in key:
            out['labels'][key] = data[key]

    key_list = []
    if 'synthetic_memory' in data.keys():  # open_type_synthetic : train
        key_list = data['synthetic_memory']
    elif 'mention_memory' in data:  # _memory, _original, _original_lemma
        for label, score in zip(data['bert_memory'], data['bert_scores']):
            if score > thred:  # 0.1, bert score
                key_list.append(label)
        key_list = key_list + data['mention_memory']
        # key_list = key_list + data['sentence_memory']
    else:  # _original_mapping
        for label, score in zip(data['memory'], data['memory_scores']):
            if score > thred:  # 0.9, glove mapping score
                key_list.append(label)

    key, value, gold_key, gold_value = [], [], [], []
    for word in key_list:
        if word in key: continue
        else: key.append(word)
        label_word = '/'.join(data['seq_labels'])[1:].split('/')
        if word in label_word:
            gold_key.append(1)
        else:
            gold_key.append(-1)
    if len(key) == 0:
        key = [MEMORY_SYMBOL]
        gold_key.append(-1)

    for label in value_list:
        if label in data['seq_labels']:
            gold_value.append(1)
        else:
            gold_value.append(-1)

    out['memory_key'] = key
    out['gold_key'] = gold_key
    out['memory_value'] = value_list
    out['gold_value'] = gold_value

    return out


def prepare_label_indexer(args) -> Dict[str, TokenIndexer]:
    label_indexers: Dict[str, TokenIndexer] = {}

    if args.label_emb_size > 0:
        label_indexers[FEATURE_NAME_SEQ_LABEL] = SingleIdTokenIndexer(
            namespace=NAME_SPACE_SEQ_LABEL,
            lowercase_tokens=args.lowercase)

    return label_indexers


def prepare_memory_indexer(args) -> Dict[str, TokenIndexer]:
    memory_indexers: Dict[str, TokenIndexer] = {}

    if args.memory_emb_size > 0:
        memory_indexers[FEATURE_NAME_MEMORY] = SingleIdTokenIndexer(
            namespace=NAME_SPACE_MEMORY,
            lowercase_tokens=args.lowercase)

    return memory_indexers


def prepare_kv_indexer(args) -> Dict[str, TokenIndexer]:
    memory_indexers: Dict[str, TokenIndexer] = {}

    if args.memory_emb_size > 0:
        memory_indexers[FEATURE_NAME_KEY] = SingleIdTokenIndexer(
            namespace=NAME_SPACE_KEY,
            lowercase_tokens=args.lowercase)
        memory_indexers[FEATURE_NAME_VALUE] = SingleIdTokenIndexer(
            namespace=NAME_SPACE_VALUE,
            lowercase_tokens=args.lowercase)

    return memory_indexers


def prepare_token_indexers(args, token=False, char=False,
                           pos_tag=False, transformer=False, mismatch=True, elmo=False) -> Dict[str, TokenIndexer]:
    token_indexers: Dict[str, TokenIndexer] = {}

    if args.token_emb_size > 0 and token:
        token_indexers[FEATURE_NAME_CONTEXT_TOKEN] = SingleIdTokenIndexer(
            namespace=NAME_SPACE_CONTEXT_TOKEN,
            lowercase_tokens=args.lowercase)
        token_indexers[FEATURE_NAME_MENTION_TOKEN] = SingleIdTokenIndexer(
            namespace=NAME_SPACE_MENTION_TOKEN,
            lowercase_tokens=args.lowercase)

    # if args.elmo and elmo:
    if elmo:
        token_indexers[FEATURE_NAME_ELMO] = ELMoTokenCharactersIndexer()

    if args.char_emb_size > 0 and char:
        token_indexers[FEATURE_NAME_CHARACTER] = TokenCharactersIndexer(
            namespace=NAME_SPACE_CHARACTER,
            min_padding_length=2)

    if args.pos_emb_size > 0 and pos_tag:
        token_indexers[FEATURE_NAME_POS_TAG] = SingleIdTokenIndexer(
            namespace=NAME_SPACE_POS_TAG, feature_name='tag_')

    if args.transformer and transformer:
        if not mismatch:
            token_indexers[FEATURE_NAME_TRANSFORMER] = PretrainedTransformerIndexer(
                model_name=args.transformer)  # , max_length=args.transformer_max_length
        elif mismatch:
            token_indexers[FEATURE_NAME_TRANSFORMER] = PretrainedTransformerMismatchedIndexer(
                model_name=args.transformer, max_length=args.transformer_max_length)

    return token_indexers


def build_dataset(args, dataset_reader, data_folder_path, distant=False):
    if distant:
        train_dataset = dataset_reader.read(os.path.join(data_folder_path, 'train_m.json'))
        # train_dataset = dataset_reader.read(os.path.join(data_folder_path, 'train_merge.json'))
    else:
        train_dataset = dataset_reader.read(os.path.join(data_folder_path, 'train.json'))
    valid_dataset = dataset_reader.read(os.path.join(data_folder_path, 'dev.json'))
    if os.path.exists(os.path.join(data_folder_path, 'test.json')):
        test_dataset = dataset_reader.read(os.path.join(data_folder_path, 'test.json'))
        vocab = Vocabulary.from_instances(train_dataset + valid_dataset + test_dataset)
    else:
        test_dataset = None
        vocab = Vocabulary.from_instances(train_dataset + valid_dataset)
    data = {
        'train': train_dataset,
        'valid': valid_dataset,
        'test': test_dataset,
        'vocab': vocab,
    }
    return data


def index_dataset(train_dataset, valid_dataset, test_dataset, vocab):
    if test_dataset is not None:
        train_dataset.index_with(vocab)
        valid_dataset.index_with(vocab)
        test_dataset.index_with(vocab)
    else:
        train_dataset.index_with(vocab)
        valid_dataset.index_with(vocab)


def build_dataloader(args,
                     train_dataset: torch.utils.data.Dataset,
                     valid_dataset: torch.utils.data.Dataset,
                     test_dataset: torch.utils.data.Dataset,
                     ) -> Tuple[allennlp.data.DataLoader, allennlp.data.DataLoader, allennlp.data.DataLoader]:
    if args.distant:
        train_loader = DataLoader(train_dataset, batch_size=args.batch, batches_per_epoch=args.batches_per_epoch)
    else:
        train_sampler = BucketBatchSampler(train_dataset, batch_size=args.batch, sorting_keys=['context'])
        train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, batches_per_epoch=args.batches_per_epoch)
    valid_sampler = BucketBatchSampler(valid_dataset, batch_size=args.batch, sorting_keys=['context'])
    valid_loader = DataLoader(valid_dataset, batch_sampler=valid_sampler)

    if test_dataset:
        test_sampler = BucketBatchSampler(test_dataset, batch_size=args.batch, sorting_keys=['context'])
        test_loader = DataLoader(test_dataset, batch_sampler=test_sampler)
    else:
        test_loader = None

    return train_loader, valid_loader, test_loader


def modify_vocab(vocab, from_namespace: List[str],
                 edit_namespace=NAME_SPACE_SEQ_LABEL,
                 padding_token=PADDING_TOKEN,
                 add_tokens: List[str] = [UNK_SYMBOL, EOS_SYMBOL],):
    file_name = 'tmp_vocab.txt'
    mapping_dic = {}
    start, end = 0, 0
    with codecs.open(file_name, "w", "utf-8") as token_file:
        for namespace in from_namespace:
            mapping = vocab._index_to_token[namespace]
            num_tokens = len(mapping)
            start_index = 1 if mapping[0] == padding_token else 0
            for i in range(start_index, num_tokens):
                print(mapping[i].replace("\n", "@@NEWLINE@@"), file=token_file)
            start = end
            end += num_tokens - start_index
            mapping_dic[namespace] = [start + 1, end + 1]
        for token in add_tokens:
            print(token.replace("\n", "@@NEWLINE@@"), file=token_file)
        mapping_dic['add_token'] = [end + 1, end + len(add_tokens) + 1]
    vocab.set_from_file(file_name, is_padded=True, namespace=edit_namespace)
    # os.remove(file_name)
    return vocab, mapping_dic


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # arg_utils.add_argument(parser)
    # args = parser.parse_args()
    # print(args)
    #
    # reader = build_data_reader(args)
    # train_dataset, valid_dataset, test_dataset, vocab = build_dataset(reader, args.data_folder_path)
    # train_loader, valid_loader, test_loader = build_dataloader(args, train_dataset, valid_dataset, test_dataset)
    #
    # for batch in train_loader:
    #     for key, value in batch.items():
    #         print('->', key)
    #         if isinstance(value, dict):
    #             for k, v in value.items():
    #                 print('--->', k)
    #                 if isinstance(v, dict):
    #                     for k2, v2 in v.items():
    #                         print('----->', k2, v2.shape)
    #     break
    # for batch in valid_loader:
    #     break
    # for batch in test_loader:
    #     break
    pass
