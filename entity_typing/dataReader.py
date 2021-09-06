import torch
import json
import os
import random
from typing import Dict, List, Tuple, Iterable

import allennlp
from allennlp.data.fields import LabelField, TextField, MultiLabelField, ArrayField, ListField, NamespaceSwappingField
from allennlp.data import DataLoader, DatasetReader, Instance, TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, PretrainedTransformerTokenizer, WhitespaceTokenizer
from entity_typing.constant import NAME_SPACE_SEQ_LABEL, NAME_SPACE_KEY, NAME_SPACE_VALUE, \
    FEATURE_NAME_MENTION_TOKEN, FEATURE_NAME_CONTEXT_TOKEN, FEATURE_NAME_CHARACTER, FEATURE_NAME_POS_TAG, \
    FEATURE_NAME_TRANSFORMER, FEATURE_NAME_ELMO, FEATURE_NAME_SEQ_LABEL, EOS_SYMBOL, FEATURE_NAME_MEMORY, \
    FEATURE_NAME_KEY, FEATURE_NAME_VALUE

from entity_typing import data_utils


class TypingReader(DatasetReader):
    def __init__(self,
                 lazy: bool=False,
                 indexers: Dict[str, TokenIndexer] = None,
                 max_sentence_length: int = None,
                 test_ins: int = -1,
                 distant: bool = False,
                 head: bool = False,
                 el: bool = False,
                 dir_path: str = None,
                 ):
        super().__init__(lazy)
        self.indexers = indexers
        self.max_sentence_length = max_sentence_length
        self.test_ins = test_ins
        self.distant = distant
        self.head = head
        self.el = el
        self.dir_path = dir_path

    def _read(self, file_path: str) -> Iterable[Instance]:
        instance_num = self.test_ins
        if 'train' in file_path:
            sample_num = 0
            with open(file_path, 'r') as lines:
                for line in lines:
                    if self.test_ins > 0:
                        instance_num -= 1
                        if instance_num < 0:
                            break
                    sample_num += 1
                    data = data_utils.process_line(line)
                    pos = data['pos_entity']
                    context = data['sentence_entity']
                    mention = data['mention']
                    mention_char = data['mention_char']
                    labels = data['labels']
                    yield self.text_to_instance(context, pos, mention, mention_char, labels)
            if self.head and 'train' in file_path:
                sample = sample_num
                with open(os.path.join(self.dir_path, 'headword_train.json'), 'r') as lines:
                    for line in lines:
                        sample -= 1
                        if sample < 0:
                            break
                        data = data_utils.process_line(line)
                        pos = data['pos_entity']
                        context = data['sentence_entity']
                        mention = data['mention']
                        mention_char = data['mention_char']
                        labels = data['labels']
                        yield self.text_to_instance(context, pos, mention, mention_char, labels)
            if self.el and 'train' in file_path:
                sample = sample_num
                with open(os.path.join(self.dir_path, 'el_train.json'), 'r') as lines:
                    for line in lines:
                        sample -= 1
                        if sample < 0:
                            break
                        data = data_utils.process_line(line)
                        pos = data['pos_entity']
                        context = data['sentence_entity']
                        mention = data['mention']
                        mention_char = data['mention_char']
                        labels = data['labels']
                        yield self.text_to_instance(context, pos, mention, mention_char, labels)
        elif 'dev' in file_path:
            with open(file_path, 'r') as lines:
                for line in lines:
                    if self.test_ins > 0:
                        instance_num -= 1
                        if instance_num < 0:
                            break
                    data = data_utils.process_line(line)
                    pos = data['pos_entity']
                    context = data['sentence_entity']
                    mention = data['mention']
                    mention_char = data['mention_char']
                    labels = data['labels']
                    yield self.text_to_instance(context, pos, mention, mention_char, labels)
        elif 'test' in file_path:
            with open(file_path, 'r') as lines:
                for line in lines:
                    if self.test_ins > 0:
                        instance_num -= 1
                        if instance_num < 0:
                            break
                    data = data_utils.process_line(line)
                    pos = data['pos_entity']
                    context = data['sentence_entity']
                    mention = data['mention']
                    mention_char = data['mention_char']
                    labels = data['labels']
                    yield self.text_to_instance(context, pos, mention, mention_char, labels)

    def text_to_instance(self,
                         context: List[str],
                         pos: List[str],
                         mention: List[str],
                         mention_chars: List[str],
                         labels: Dict[str, List[str]] = None) -> Instance:
        if self.max_sentence_length:
            context = context[:self.max_sentence_length]
            pos = pos[:self.max_sentence_length]

        context_tokens = [Token(text=word, tag_=pos_token) for word, pos_token in zip(context, pos)]
        mention_tokens = [Token(text=word) for word in mention]
        mention_char_tokens = [Token(text=char) for char in mention_chars]

        context_token_indexers = {FEATURE_NAME_CONTEXT_TOKEN: self.indexers[FEATURE_NAME_CONTEXT_TOKEN],
                                  FEATURE_NAME_POS_TAG: self.indexers[FEATURE_NAME_POS_TAG]}
        if FEATURE_NAME_TRANSFORMER in self.indexers:
            context_token_indexers[FEATURE_NAME_TRANSFORMER] = self.indexers[FEATURE_NAME_TRANSFORMER]
        context_tokens_field = TextField(context_tokens, context_token_indexers)
        mention_tokens_field = TextField(mention_tokens, {FEATURE_NAME_MENTION_TOKEN: self.indexers[FEATURE_NAME_MENTION_TOKEN]})
        mention_chars_field = TextField(mention_char_tokens, {FEATURE_NAME_CHARACTER: self.indexers[FEATURE_NAME_CHARACTER]})

        fields = {'context': context_tokens_field,
                  'mention': mention_tokens_field,
                  'mention_char': mention_chars_field
                  }
        if labels is not None:
            for key, label in labels.items():
                namespace = key
                assert 'labels' in namespace
                fields[namespace] = MultiLabelField(label, label_namespace=namespace)

        return Instance(fields)


class TypingDistantReader(DatasetReader):
    def __init__(self,
                 lazy: bool=False,
                 indexers: Dict[str, TokenIndexer] = None,
                 max_sentence_length: int = None,
                 test_ins: int = -1,
                 distant: bool = False,
                 head: bool = False,
                 el: bool = False,
                 dir_path: str = None,
                 batch_size: int = 32,
                 ):
        super().__init__(lazy)
        self.indexers = indexers
        self.max_sentence_length = max_sentence_length
        self.test_ins = test_ins
        self.distant = distant
        self.head = head
        self.el = el
        self.dir_path = dir_path
        self.batch_size = batch_size

    def _read(self, file_path: str) -> Iterable[Instance]:
        instance_num = self.test_ins
        if 'train' in file_path:
            sample_num = 0
            train_dataset = []
            with open(file_path, 'r') as lines:
                for line in lines:
                    if self.test_ins > 0:
                        instance_num -= 1
                        if instance_num < 0:
                            break
                    sample_num += 1
                    train_dataset.append(data_utils.process_line(line))
            random.shuffle(train_dataset)

            if self.head and 'train' in file_path:
                sample = sample_num
                head_dataset = []
                with open(os.path.join(self.dir_path, 'headword_train.json'), 'r') as lines:
                    for line in lines:
                        sample -= 1
                        if sample < 0:
                            break
                        data = data_utils.process_line(line)
                        head_dataset.append(data)
                random.shuffle(head_dataset)

            if self.el and 'train' in file_path:
                el_dataset = []
                sample = sample_num
                with open(os.path.join(self.dir_path, 'el_train.json'), 'r') as lines:
                    for line in lines:
                        sample -= 1
                        if sample < 0:
                            break
                        el_dataset.append(data_utils.process_line(line))
                random.shuffle(el_dataset)
            batch_num = 0
            max_batch = sample_num // self.batch_size
            while True:
                if max_batch < 0:
                    break
                max_batch -= 1
                output_idx = list(range(self.batch_size))
                output_idx = [x + batch_num*self.batch_size for x in output_idx]
                batch_num += 1
                for idx in output_idx:
                    if idx >= sample_num: break
                    pos = train_dataset[idx]['pos_entity']
                    context = train_dataset[idx]['sentence_entity']
                    mention = train_dataset[idx]['mention']
                    mention_char = train_dataset[idx]['mention_char']
                    labels = train_dataset[idx]['labels']
                    yield self.text_to_instance(context, pos, mention, mention_char, labels)
                if self.head:
                    for idx in output_idx:
                        if idx >= sample_num: break
                        pos = head_dataset[idx]['pos_entity']
                        context = head_dataset[idx]['sentence_entity']
                        mention = head_dataset[idx]['mention']
                        mention_char = head_dataset[idx]['mention_char']
                        labels = head_dataset[idx]['labels']
                        yield self.text_to_instance(context, pos, mention, mention_char, labels)
                if self.el:
                    for idx in output_idx:
                        if idx >= sample_num: break
                        pos = el_dataset[idx]['pos_entity']
                        context = el_dataset[idx]['sentence_entity']
                        mention = el_dataset[idx]['mention']
                        mention_char = el_dataset[idx]['mention_char']
                        labels = el_dataset[idx]['labels']
                        yield self.text_to_instance(context, pos, mention, mention_char, labels)

        elif 'dev' in file_path:
            with open(file_path, 'r') as lines:
                for line in lines:
                    if self.test_ins > 0:
                        instance_num -= 1
                        if instance_num < 0:
                            break
                    data = data_utils.process_line(line)
                    pos = data['pos_entity']
                    context = data['sentence_entity']
                    mention = data['mention']
                    mention_char = data['mention_char']
                    labels = data['labels']
                    yield self.text_to_instance(context, pos, mention, mention_char, labels)
        elif 'test' in file_path:
            with open(file_path, 'r') as lines:
                for line in lines:
                    if self.test_ins > 0:
                        instance_num -= 1
                        if instance_num < 0:
                            break
                    data = data_utils.process_line(line)
                    pos = data['pos_entity']
                    context = data['sentence_entity']
                    mention = data['mention']
                    mention_char = data['mention_char']
                    labels = data['labels']
                    yield self.text_to_instance(context, pos, mention, mention_char, labels)

    def text_to_instance(self,
                         context: List[str],
                         pos: List[str],
                         mention: List[str],
                         mention_chars: List[str],
                         labels: Dict[str, List[str]] = None) -> Instance:
        if self.max_sentence_length:
            context = context[:self.max_sentence_length]
            pos = pos[:self.max_sentence_length]

        context_tokens = [Token(text=word, tag_=pos_token) for word, pos_token in zip(context, pos)]
        mention_tokens = [Token(text=word) for word in mention]
        mention_char_tokens = [Token(text=char) for char in mention_chars]

        context_token_indexers = {FEATURE_NAME_CONTEXT_TOKEN: self.indexers[FEATURE_NAME_CONTEXT_TOKEN],
                                  FEATURE_NAME_POS_TAG: self.indexers[FEATURE_NAME_POS_TAG]}
        if FEATURE_NAME_TRANSFORMER in self.indexers:
            context_token_indexers[FEATURE_NAME_TRANSFORMER] = self.indexers[FEATURE_NAME_TRANSFORMER]
        context_tokens_field = TextField(context_tokens, context_token_indexers)
        mention_tokens_field = TextField(mention_tokens, {FEATURE_NAME_MENTION_TOKEN: self.indexers[FEATURE_NAME_MENTION_TOKEN]})
        mention_chars_field = TextField(mention_char_tokens, {FEATURE_NAME_CHARACTER: self.indexers[FEATURE_NAME_CHARACTER]})

        fields = {'context': context_tokens_field,
                  'mention': mention_tokens_field,
                  'mention_char': mention_chars_field
                  }
        if labels is not None:
            for key, label in labels.items():
                namespace = key
                assert 'labels' in namespace
                fields[namespace] = MultiLabelField(label, label_namespace=namespace)

        return Instance(fields)


class BertMarkerTypingReader(DatasetReader):
    def __init__(self,
                 lazy: bool=False,
                 tokenizers: Dict[str, Tokenizer] = None,
                 indexers: Dict[str, TokenIndexer] = None,
                 max_sentence_length: int=None,
                 test_ins: int=-1
                 ):
        super().__init__(lazy)
        self.tokenizers = tokenizers
        self.indexers = indexers
        self.max_sentence_length = max_sentence_length
        self.test_ins = test_ins

    def _read(self, file_path: str) -> Iterable[Instance]:
        instance_num = self.test_ins
        with open(file_path, 'r') as lines:
            for line in lines:
                if self.test_ins > 0:
                    instance_num -= 1
                    if instance_num < 0:
                        break
                data = data_utils.process_line(line)
                left_sentence = data['left_sentence']
                right_sentence = data['right_sentence']
                mention = data['mention']
                labels = data['labels']
                yield self.text_to_instance(left_sentence, right_sentence, mention, labels)

    def text_to_instance(self,
                         left_sentence: List[str],
                         right_sentence: List[str],
                         mention: List[str],
                         labels: Dict[str, List[str]] = None) -> Instance:

        left_sentence = self.tokenizers[FEATURE_NAME_TRANSFORMER].tokenize(' '.join(left_sentence))
        right_sentence = self.tokenizers[FEATURE_NAME_TRANSFORMER].tokenize(' '.join(right_sentence))
        mention = self.tokenizers[FEATURE_NAME_TRANSFORMER].tokenize(' '.join(mention))

        marker1 = [Token(text='[unused0]', text_id=1, type_id=0)]
        marker2 = [Token(text='[unused1]', text_id=2, type_id=0)]

        sentence = left_sentence + marker1 + mention + marker2 + right_sentence

        random_drop = False
        if random_drop and random.random() > 0.5:
            mask = Token(text='[MASK]', text_id=103, type_id=0)
            idx = random.randint(0, len(sentence)-1)
            sentence[idx] = mask

        sentence = self.tokenizers[FEATURE_NAME_TRANSFORMER].add_special_tokens(sentence)

        context_token_indexers = {FEATURE_NAME_TRANSFORMER: self.indexers[FEATURE_NAME_TRANSFORMER]}
        context_tokens_field = TextField(sentence, context_token_indexers)
        fields = {'context': context_tokens_field,
                  }
        if labels is not None:
            for key, label in labels.items():
                namespace = key
                assert 'labels' in namespace
                fields[namespace] = MultiLabelField(label, label_namespace=namespace)

        return Instance(fields)


class BertSepTypingReader(DatasetReader):
    def __init__(self,
                 lazy: bool=False,
                 tokenizers: Dict[str, Tokenizer] = None,
                 indexers: Dict[str, TokenIndexer] = None,
                 max_sentence_length: int=None,
                 test_ins: int=-1
                 ):
        super().__init__(lazy)
        self.tokenizers = tokenizers
        self.indexers = indexers
        self.max_sentence_length = max_sentence_length
        self.test_ins = test_ins

    def _read(self, file_path: str) -> Iterable[Instance]:
        instance_num = self.test_ins
        with open(file_path, 'r') as lines:
            for line in lines:
                if self.test_ins > 0:
                    instance_num -= 1
                    if instance_num < 0:
                        break
                data = data_utils.process_line(line)
                sentence = data['sentence']
                mention = data['mention']
                labels = data['labels']
                yield self.text_to_instance(sentence, mention, labels)

    def text_to_instance(self,
                         sentence: List[str],
                         mention: List[str],
                         labels: Dict[str, List[str]] = None) -> Instance:

        sentence = self.tokenizers[FEATURE_NAME_TRANSFORMER].tokenize(' '.join(sentence))

        # for ontonotes dataset
        if self.max_sentence_length:
            sentence = sentence[:self.max_sentence_length]

        random_drop = False
        if random_drop and random.random() > 0.5:
            mask = Token(text='[MASK]', text_id=103, type_id=0)
            idx = random.randint(0, len(sentence)-1)
            sentence[idx] = mask

        mention = self.tokenizers[FEATURE_NAME_TRANSFORMER].tokenize(' '.join(mention))
        sentence_mention = self.tokenizers[FEATURE_NAME_TRANSFORMER].add_special_tokens(sentence, mention)

        context_token_indexers = {FEATURE_NAME_TRANSFORMER: self.indexers[FEATURE_NAME_TRANSFORMER]}
        context_tokens_field = TextField(sentence_mention, context_token_indexers)
        fields = {'context': context_tokens_field,
                  }
        if labels is not None:
            for key, label in labels.items():
                namespace = key
                assert 'labels' in namespace
                fields[namespace] = MultiLabelField(label, label_namespace=namespace)

        return Instance(fields)


# LSTM + LSTM
class SeqTypingReader(DatasetReader):
    def __init__(self,
                 lazy: bool=False,
                 tokenizers: Dict[str, Tokenizer] = None,
                 indexers: Dict[str, TokenIndexer] = None,
                 label_indexers: Dict[str, TokenIndexer] = None,
                 max_sentence_length: int = None,
                 max_mention_length: int = None,
                 test_ins: int = -1,
                 seq_type: str = 'sequence',
                 shuffle_num: int = 0,
                 ):
        super().__init__(lazy)
        self.tokenizers = tokenizers
        self.indexers = indexers
        self.label_indexers = label_indexers
        self.max_sentence_length = max_sentence_length
        self.max_mention_length = max_mention_length
        self.test_ins = test_ins
        self.seq_type = seq_type
        self.shuffle_num = shuffle_num

    def _read(self, file_path: str) -> Iterable[Instance]:
        instance_num = self.test_ins
        with open(file_path, 'r') as lines:
            for line in lines:
                if self.test_ins > 0:
                    instance_num -= 1
                    if instance_num < 0:
                        break
                data = data_utils.process_line(line)
                pos = data['pos']
                context = data['sentence']
                mention = data['mention']
                mention_char = data['mention_char']
                labels = data['labels']
                seq_labels = data['seq_labels']
                if 'train' in file_path and self.shuffle_num > 0:
                    for i in range(self.shuffle_num):
                        random.shuffle(seq_labels)
                        for k, v in labels.items():
                            random.shuffle(labels[k])
                        yield self.text_to_instance(pos, context, mention, mention_char, labels, seq_labels)
                else:
                    yield self.text_to_instance(pos, context, mention, mention_char, labels, seq_labels)

    def text_to_instance(self,
                         context: List[str],
                         pos: List[str],
                         mention: List[str],
                         mention_chars: List[str],
                         labels: Dict[str, List[str]] = None,
                         seq_labels: List[str] = None) -> Instance:

        context_tokens = [Token(text=word, tag_=pos_token) for word, pos_token in zip(context, pos)]
        mention_tokens = [Token(text=word) for word in mention]
        mention_char_tokens = [Token(text=char) for char in mention_chars]

        if self.max_sentence_length is not None:
            context_tokens = context_tokens[: self.max_sentence_length]
        if self.max_mention_length is not None:
            mention_tokens = mention_tokens[: self.max_mention_length]

        context_token_indexers = {FEATURE_NAME_POS_TAG: self.indexers[FEATURE_NAME_POS_TAG],
                                  FEATURE_NAME_ELMO: self.indexers[FEATURE_NAME_ELMO]}
        mention_token_indexers = {FEATURE_NAME_ELMO: self.indexers[FEATURE_NAME_ELMO]}

        context_tokens_field = TextField(context_tokens, context_token_indexers)
        mention_tokens_field = TextField(mention_tokens, mention_token_indexers)
        mention_chars_field = TextField(mention_char_tokens,
                                        {FEATURE_NAME_CHARACTER: self.indexers[FEATURE_NAME_CHARACTER]})

        fields = {'context': context_tokens_field,
                  'mention': mention_tokens_field,
                  'mention_char': mention_chars_field
                  }

        if labels is not None:
            for key, label in labels.items():
                namespace = key
                assert 'labels' in namespace
                fields[namespace] = MultiLabelField(label, label_namespace=namespace)

        if self.seq_type == 'sequence':
            seq_label_list = seq_labels
        elif self.seq_type == 'layer':
            seq_label_list = []
            for name in ['labels', 'fine_labels', 'ultra_fine_labels']:
                seq_label_list.extend(labels[name])
        else:
            raise ValueError('no such sequence type')

        seq_label_list = seq_label_list + [EOS_SYMBOL]
        seq_label_tokens = [Token(text=label) for label in seq_label_list]
        seq_label_field = TextField(seq_label_tokens, self.label_indexers)
        fields[FEATURE_NAME_SEQ_LABEL] = seq_label_field

        return Instance(fields)


# bert entity marker + LSTM
class SeqMarkerTypingReader(DatasetReader):
    def __init__(self,
                 lazy: bool=False,
                 tokenizers: Dict[str, Tokenizer] = None,
                 indexers: Dict[str, TokenIndexer] = None,
                 label_indexers: Dict[str, TokenIndexer] = None,
                 max_sentence_length: int = None,
                 max_mention_length: int = None,
                 test_ins: int = -1,
                 seq_type: str = 'sequence',
                 shuffle_num: int = 0,
                 distant: bool = False,
                 ):
        super().__init__(lazy)
        self.tokenizers = tokenizers
        self.indexers = indexers
        self.label_indexers = label_indexers
        self.max_sentence_length = max_sentence_length
        self.max_mention_length = max_mention_length  # for ontonotes
        self.test_ins = test_ins
        self.seq_type = seq_type
        self.shuffle_num = shuffle_num
        self.distant = distant  # for ontonotes

    def _read(self, file_path: str) -> Iterable[Instance]:
        instance_num = self.test_ins
        with open(file_path, 'r') as lines:
            for line in lines:
                if self.test_ins > 0:
                    instance_num -= 1
                    if instance_num < 0:
                        break
                data = data_utils.process_line(line)
                left_sentence = data['left_sentence']
                right_sentence = data['right_sentence']
                mention = data['mention']
                labels = data['labels']
                seq_labels = data['seq_labels']
                if 'train' in file_path and self.shuffle_num > 0:
                    for i in range(self.shuffle_num):
                        for k,v in labels.items():
                            random.shuffle(labels[k])
                        random.shuffle(seq_labels)
                        yield self.text_to_instance(left_sentence, right_sentence, mention, labels, seq_labels)
                else:
                    yield self.text_to_instance(left_sentence, right_sentence, mention, labels, seq_labels)

    def text_to_instance(self,
                         left_sentence: List[str],
                         right_sentence: List[str],
                         mention: List[str],
                         labels: Dict[str, List[str]] = None,
                         seq_labels: List[str] = None) -> Instance:

        left_sentence = self.tokenizers[FEATURE_NAME_TRANSFORMER].tokenize(' '.join(left_sentence))
        right_sentence = self.tokenizers[FEATURE_NAME_TRANSFORMER].tokenize(' '.join(right_sentence))
        mention = self.tokenizers[FEATURE_NAME_TRANSFORMER].tokenize(' '.join(mention))

        # for ontonotes
        if self.max_sentence_length is not None:
            left_sentence = left_sentence[:self.max_sentence_length]
            right_sentence = right_sentence[:self.max_sentence_length-len(left_sentence)]
        if self.max_mention_length is not None:
            mention = mention[:self.max_mention_length]

        marker1 = [Token(text='[unused0]', text_id=1, type_id=0)]
        marker2 = [Token(text='[unused1]', text_id=2, type_id=0)]

        sentence = left_sentence + marker1 + mention + marker2 + right_sentence
        sentence = self.tokenizers[FEATURE_NAME_TRANSFORMER].add_special_tokens(sentence)

        context_token_indexers = {FEATURE_NAME_TRANSFORMER: self.indexers[FEATURE_NAME_TRANSFORMER]}
        context_tokens_field = TextField(sentence, context_token_indexers)
        fields = {'context': context_tokens_field,
                  }

        if labels is not None:
            for key, label in labels.items():
                namespace = key
                assert 'labels' in namespace
                fields[namespace] = MultiLabelField(label, label_namespace=namespace)

        if self.seq_type == 'sequence':
            seq_label_list = seq_labels
        elif self.seq_type == 'layer':
            seq_label_list = []
            for name in ['labels', 'fine_labels', 'ultra_fine_labels']:
                seq_label_list.extend(labels[name])
        else:
            raise ValueError('no such sequence type')

        seq_label_list = seq_label_list + [EOS_SYMBOL]
        seq_label_tokens = [Token(text=label) for label in seq_label_list]
        seq_label_field = TextField(seq_label_tokens, self.label_indexers)
        fields[FEATURE_NAME_SEQ_LABEL] = seq_label_field

        return Instance(fields)


# bert sep mention + LSTM
class SeqSepTypingReader(DatasetReader):
    def __init__(self,
                 lazy: bool=False,
                 tokenizers: Dict[str, Tokenizer] = None,
                 indexers: Dict[str, TokenIndexer] = None,
                 label_indexers: Dict[str, TokenIndexer] = None,
                 max_sentence_length: int = None,
                 max_mention_length: int = None,
                 test_ins: int = -1,
                 seq_type: str = 'sequence',
                 shuffle_num: int = 0,
                 distant: bool = False,
                 ):
        super().__init__(lazy)
        self.tokenizers = tokenizers
        self.indexers = indexers
        self.label_indexers = label_indexers
        self.max_sentence_length = max_sentence_length
        self.max_mention_length = max_mention_length
        self.test_ins = test_ins
        self.seq_type = seq_type
        self.shuffle_num = shuffle_num
        self.distant = distant

    def _read(self, file_path: str) -> Iterable[Instance]:
        instance_num = self.test_ins
        with open(file_path, 'r') as lines:
            for line in lines:
                if self.test_ins > 0:
                    instance_num -= 1
                    if instance_num < 0:
                        break
                data = data_utils.process_line(line)
                sentence = data['sentence']
                mention = data['mention']
                labels = data['labels']
                seq_labels = data['seq_labels']

                if 'train' in file_path and self.shuffle_num > 0:
                    for i in range(self.shuffle_num):
                        for k,v in labels.items():
                            random.shuffle(labels[k])
                        random.shuffle(seq_labels)
                        yield self.text_to_instance(sentence, mention, labels, seq_labels)
                else:
                    yield self.text_to_instance(sentence, mention, labels, seq_labels)

    def text_to_instance(self,
                         sentence: List[str],
                         mention: List[str],
                         labels: Dict[str, List[str]] = None,
                         seq_labels: List[str] = None) -> Instance:

        sentence = self.tokenizers[FEATURE_NAME_TRANSFORMER].tokenize(' '.join(sentence))
        # for ontonotes dataset
        if self.max_sentence_length is not None:
            sentence = sentence[:self.max_sentence_length]

        mention = self.tokenizers[FEATURE_NAME_TRANSFORMER].tokenize(' '.join(mention))
        if self.max_mention_length is not None:
            mention = mention[:self.max_mention_length]
        # sentence_mention = self.tokenizers[FEATURE_NAME_TRANSFORMER].add_special_tokens(sentence, mention)
        sentence_mention = self.tokenizers[FEATURE_NAME_TRANSFORMER].add_special_tokens(mention, sentence)
        context_token_indexers = {FEATURE_NAME_TRANSFORMER: self.indexers[FEATURE_NAME_TRANSFORMER]}
        context_tokens_field = TextField(sentence_mention, context_token_indexers)
        fields = {'context': context_tokens_field,
                  }

        if labels is not None:
            for key, label in labels.items():
                namespace = key
                assert 'labels' in namespace
                fields[namespace] = MultiLabelField(label, label_namespace=namespace)

        if self.seq_type == 'sequence':
            seq_label_list = seq_labels
        elif self.seq_type == 'layer':
            seq_label_list = []
            for name in ['labels', 'fine_labels', 'ultra_fine_labels']:
                seq_label_list.extend(labels[name])
        else:
            raise ValueError('no such sequence type')

        seq_label_list = seq_label_list + [EOS_SYMBOL]
        seq_label_tokens = [Token(text=label) for label in seq_label_list]
        seq_label_field = TextField(seq_label_tokens, self.label_indexers)
        fields[FEATURE_NAME_SEQ_LABEL] = seq_label_field

        return Instance(fields)


# bert entity marker + LSTM
class MarMemoryTypingReader(DatasetReader):
    def __init__(self,
                 lazy: bool=False,
                 tokenizers: Dict[str, Tokenizer] = None,
                 indexers: Dict[str, TokenIndexer] = None,
                 label_indexers: Dict[str, TokenIndexer] = None,
                 memory_indexers: Dict[str, TokenIndexer] = None,
                 max_sentence_length: int = None,
                 test_ins: int = -1,
                 seq_type: str = 'sequence',
                 shuffle_num: int = 0,
                 ):
        super().__init__(lazy)
        self.tokenizers = tokenizers
        self.indexers = indexers
        self.label_indexers = label_indexers
        self.memory_indexers = memory_indexers
        self.max_sentence_length = max_sentence_length
        self.test_ins = test_ins
        self.seq_type = seq_type
        self.shuffle_num = shuffle_num

    def _read(self, file_path: str) -> Iterable[Instance]:
        instance_num = self.test_ins
        with open(file_path, 'r') as lines:
            for line in lines:
                if self.test_ins > 0:
                    instance_num -= 1
                    if instance_num < 0:
                        break
                if 'mention_memory' in line:
                    data = data_utils.process_line(line, mem=True, thred=0.1)
                else:
                    data = data_utils.process_line(line, mem=True, thred=0.9)
                left_sentence = data['left_sentence']
                right_sentence = data['right_sentence']
                mention = data['mention']
                labels = data['labels']
                seq_labels = data['seq_labels']
                memory = data['memory']
                gold_memory = data['gold_memory']
                if 'train' in file_path and self.shuffle_num > 0:
                    for i in range(self.shuffle_num):
                        for k, v in labels.items():
                            random.shuffle(labels[k])
                        random.shuffle(seq_labels)
                        yield self.text_to_instance(left_sentence, right_sentence, mention, labels, seq_labels, memory, gold_memory)
                else:
                    yield self.text_to_instance(left_sentence, right_sentence, mention, labels, seq_labels, memory, gold_memory)

    def text_to_instance(self,
                         left_sentence: List[str],
                         right_sentence: List[str],
                         mention: List[str],
                         labels: Dict[str, List[str]] = None,
                         seq_labels: List[str] = None,
                         memory: List[str] = None,
                         gold_memory: List[int] = None) -> Instance:
        left_sentence = self.tokenizers[FEATURE_NAME_TRANSFORMER].tokenize(' '.join(left_sentence))
        right_sentence = self.tokenizers[FEATURE_NAME_TRANSFORMER].tokenize(' '.join(right_sentence))
        mention = self.tokenizers[FEATURE_NAME_TRANSFORMER].tokenize(' '.join(mention))

        marker1 = [Token(text='[unused0]', text_id=1, type_id=0)]
        marker2 = [Token(text='[unused1]', text_id=2, type_id=0)]
        sentence = left_sentence + marker1 + mention + marker2 + right_sentence
        sentence = self.tokenizers[FEATURE_NAME_TRANSFORMER].add_special_tokens(sentence)

        context_token_indexers = {FEATURE_NAME_TRANSFORMER: self.indexers[FEATURE_NAME_TRANSFORMER]}
        context_tokens_field = TextField(sentence, context_token_indexers)
        fields = {'context': context_tokens_field,
                  }

        if labels is not None:
            for key, label in labels.items():
                namespace = key
                assert 'labels' in namespace
                fields[namespace] = MultiLabelField(label, label_namespace=namespace)

        if self.seq_type == 'sequence':
            seq_label_list = seq_labels
        elif self.seq_type == 'layer':
            seq_label_list = []
            for name in ['labels', 'fine_labels', 'ultra_fine_labels']:
                seq_label_list.extend(labels[name])
        else:
            raise ValueError('no such sequence type')

        seq_label_list = seq_label_list + [EOS_SYMBOL]
        seq_label_tokens = [Token(text=label) for label in seq_label_list]
        seq_label_field = TextField(seq_label_tokens, self.label_indexers)
        fields[FEATURE_NAME_SEQ_LABEL] = seq_label_field

        if len(memory) > 0:
            memory_tokens = [Token(text=label) for label in memory]
            memory_field = TextField(memory_tokens, self.memory_indexers)
            fields[FEATURE_NAME_MEMORY] = memory_field

            gold_memory_list = []
            for flag in gold_memory:
                gold_memory_list.append(LabelField(flag, label_namespace='memory_labels', skip_indexing=True))
            fields['memory_labels'] = ListField(gold_memory_list)

        return Instance(fields)


# bert sep mention + LSTM
class SepMemoryTypingReader(DatasetReader):
    def __init__(self,
                 lazy: bool=False,
                 tokenizers: Dict[str, Tokenizer] = None,
                 indexers: Dict[str, TokenIndexer] = None,
                 label_indexers: Dict[str, TokenIndexer] = None,
                 memory_indexers: Dict[str, TokenIndexer] = None,
                 max_sentence_length: int = None,
                 max_mention_length: int = None,
                 test_ins: int = -1,
                 seq_type: str = 'sequence',
                 shuffle_num: int = 0,
                 ):
        super().__init__(lazy)
        self.tokenizers = tokenizers
        self.indexers = indexers
        self.label_indexers = label_indexers
        self.memory_indexers = memory_indexers
        self.max_sentence_length = max_sentence_length
        self.max_mention_length = max_mention_length
        self.test_ins = test_ins
        self.seq_type = seq_type
        self.shuffle_num = shuffle_num

    def _read(self, file_path: str) -> Iterable[Instance]:
        instance_num = self.test_ins
        with open(file_path, 'r') as lines:
            for line in lines:
                if self.test_ins > 0:
                    instance_num -= 1
                    if instance_num < 0:
                        break
                data = data_utils.process_line(line, mem=True, thred=0.8)
                sentence = data['sentence']
                mention = data['mention']
                labels = data['labels']
                seq_labels = data['seq_labels']
                memory = data['memory']
                gold_memory = data['gold_memory']
                if 'train' in file_path and self.shuffle_num > 0:
                    for i in range(self.shuffle_num):
                        for k, v in labels.items():
                            random.shuffle(labels[k])
                        random.shuffle(seq_labels)
                        yield self.text_to_instance(sentence, mention, labels, seq_labels, memory, gold_memory)
                else:
                    yield self.text_to_instance(sentence, mention, labels, seq_labels, memory, gold_memory)

    def text_to_instance(self,
                         sentence: List[str],
                         mention: List[str],
                         labels: Dict[str, List[str]] = None,
                         seq_labels: List[str] = None,
                         memory: List[str] = None,
                         gold_memory: List[int] = None) -> Instance:
        sentence = self.tokenizers[FEATURE_NAME_TRANSFORMER].tokenize(' '.join(sentence))
        if self.max_sentence_length is not None:
            sentence = sentence[:self.max_sentence_length]

        mention = self.tokenizers[FEATURE_NAME_TRANSFORMER].tokenize(' '.join(mention))
        if self.max_mention_length is not None:
            mention = mention[:self.max_mention_length]
        sentence_mention = self.tokenizers[FEATURE_NAME_TRANSFORMER].add_special_tokens(sentence, mention)

        context_token_indexers = {FEATURE_NAME_TRANSFORMER: self.indexers[FEATURE_NAME_TRANSFORMER]}
        context_tokens_field = TextField(sentence_mention, context_token_indexers)
        fields = {'context': context_tokens_field,
                  }

        if labels is not None:
            for key, label in labels.items():
                namespace = key
                assert 'labels' in namespace
                fields[namespace] = MultiLabelField(label, label_namespace=namespace)

        if self.seq_type == 'sequence':
            seq_label_list = seq_labels
        elif self.seq_type == 'layer':
            seq_label_list = []
            for name in ['labels', 'fine_labels', 'ultra_fine_labels']:
                seq_label_list.extend(labels[name])
        else:
            raise ValueError('no such sequence type')

        seq_label_list = seq_label_list + [EOS_SYMBOL]
        seq_label_tokens = [Token(text=label) for label in seq_label_list]
        seq_label_field = TextField(seq_label_tokens, self.label_indexers)
        fields[FEATURE_NAME_SEQ_LABEL] = seq_label_field

        if len(memory) > 0:
            memory_tokens = [Token(text=label) for label in memory]
            memory_field = TextField(memory_tokens, self.memory_indexers)
            fields[FEATURE_NAME_MEMORY] = memory_field

            gold_memory_list = []
            for flag in gold_memory:
                gold_memory_list.append(LabelField(flag, label_namespace='memory_labels', skip_indexing=True))
            fields['memory_labels'] = ListField(gold_memory_list)

        return Instance(fields)


# bert entity marker + LSTM -> soft mapping kv
class MarSoftMemoryTypingReader(DatasetReader):
    def __init__(self,
                 lazy: bool=False,
                 tokenizers: Dict[str, Tokenizer] = None,
                 indexers: Dict[str, TokenIndexer] = None,
                 label_indexers: Dict[str, TokenIndexer] = None,
                 memory_indexers: Dict[str, TokenIndexer] = None,
                 max_sentence_length: int = None,
                 test_ins: int = -1,
                 seq_type: str = 'sequence',
                 shuffle_num: int = 0,
                 value_file: str = None,
                 bert_thred: float = 0.1,
                 ):
        super().__init__(lazy)
        self.tokenizers = tokenizers
        self.indexers = indexers
        self.label_indexers = label_indexers
        self.memory_indexers = memory_indexers
        self.max_sentence_length = max_sentence_length
        self.test_ins = test_ins
        self.seq_type = seq_type
        self.shuffle_num = shuffle_num
        self.value_list = open(value_file, 'r').read().strip().split('\n')
        self.bert_thred = bert_thred

    def _read(self, file_path: str) -> Iterable[Instance]:
        instance_num = self.test_ins
        with open(file_path, 'r') as lines:
            for line in lines:
                if self.test_ins > 0:
                    instance_num -= 1
                    if instance_num < 0:
                        break
                data = data_utils.process_kv_line(line, self.value_list, thred=self.bert_thred)
                left_sentence = data['left_sentence']
                right_sentence = data['right_sentence']
                mention = data['mention']
                labels = data['labels']
                seq_labels = data['seq_labels']
                memory_key = data['memory_key']
                gold_key = data['gold_key']
                memory_value = data['memory_value']
                gold_value = data['gold_value']

                if 'train' in file_path and self.shuffle_num > 0:
                    for i in range(self.shuffle_num):
                        for k, v in labels.items():
                            random.shuffle(labels[k])
                        random.shuffle(seq_labels)
                        yield self.text_to_instance(left_sentence, right_sentence, mention, labels, seq_labels,
                                                    memory_key, gold_key, memory_value, gold_value)
                else:
                    yield self.text_to_instance(left_sentence, right_sentence, mention, labels, seq_labels,
                                                memory_key, gold_key, memory_value, gold_value)

    def text_to_instance(self,
                         left_sentence: List[str],
                         right_sentence: List[str],
                         mention: List[str],
                         labels: Dict[str, List[str]] = None,
                         seq_labels: List[str] = None,
                         memory_key: List[str] = None,
                         gold_key: List[int] = None,
                         memory_value: List[str] = None,
                         gold_value: List[int] = None) -> Instance:
        left_sentence = self.tokenizers[FEATURE_NAME_TRANSFORMER].tokenize(' '.join(left_sentence))
        right_sentence = self.tokenizers[FEATURE_NAME_TRANSFORMER].tokenize(' '.join(right_sentence))
        mention = self.tokenizers[FEATURE_NAME_TRANSFORMER].tokenize(' '.join(mention))
        marker1 = [Token(text='[unused0]', text_id=1, type_id=0)]
        marker2 = [Token(text='[unused1]', text_id=2, type_id=0)]
        sentence = left_sentence + marker1 + mention + marker2 + right_sentence
        sentence = self.tokenizers[FEATURE_NAME_TRANSFORMER].add_special_tokens(sentence)

        context_token_indexers = {FEATURE_NAME_TRANSFORMER: self.indexers[FEATURE_NAME_TRANSFORMER]}
        context_tokens_field = TextField(sentence, context_token_indexers)
        fields = {'context': context_tokens_field,
                  }

        if labels is not None:
            for key, label in labels.items():
                namespace = key
                assert 'labels' in namespace
                fields[namespace] = MultiLabelField(label, label_namespace=namespace)

        if self.seq_type == 'sequence':
            seq_label_list = seq_labels
        elif self.seq_type == 'layer':
            seq_label_list = []
            for name in ['labels', 'fine_labels', 'ultra_fine_labels']:
                seq_label_list.extend(labels[name])
        else:
            raise ValueError('no such sequence type')

        seq_label_list = seq_label_list + [EOS_SYMBOL]
        seq_label_tokens = [Token(text=label) for label in seq_label_list]
        seq_label_field = TextField(seq_label_tokens, self.label_indexers)
        fields[FEATURE_NAME_SEQ_LABEL] = seq_label_field

        if len(memory_key) > 0:
            key_tokens = [Token(text=label) for label in memory_key]
            key_indexers = {FEATURE_NAME_KEY: self.memory_indexers[FEATURE_NAME_KEY]}
            key_field = TextField(key_tokens, key_indexers)
            fields[FEATURE_NAME_KEY] = key_field

            gold_key_list = []
            for flag in gold_key:
                gold_key_list.append(LabelField(flag, label_namespace='key_labels', skip_indexing=True))
            fields["key_labels"] = ListField(gold_key_list)

            value_tokens = [Token(text=label) for label in memory_value]
            value_indexers = {FEATURE_NAME_VALUE: self.memory_indexers[FEATURE_NAME_VALUE]}
            value_field = TextField(value_tokens, value_indexers)
            fields[FEATURE_NAME_VALUE] = value_field
            gold_value_list = []
            for flag in gold_value:
                gold_value_list.append(LabelField(flag, label_namespace='value_labels', skip_indexing=True))
            fields["value_labels"] = ListField(gold_value_list)

            fields["value2label"] = NamespaceSwappingField(source_tokens=value_tokens,
                                                           target_namespace=NAME_SPACE_SEQ_LABEL)

        return Instance(fields)


# bert sep mention + LSTM -> soft mapping kv
class SepSoftMemoryTypingReader(DatasetReader):
    def __init__(self,
                 lazy: bool=False,
                 tokenizers: Dict[str, Tokenizer] = None,
                 indexers: Dict[str, TokenIndexer] = None,
                 label_indexers: Dict[str, TokenIndexer] = None,
                 memory_indexers: Dict[str, TokenIndexer] = None,
                 max_sentence_length: int = None,
                 max_mention_length: int = None,
                 test_ins: int = -1,
                 seq_type: str = 'sequence',
                 shuffle_num: int = 0,
                 value_file: str = None,
                 bert_thred: float = 0.1,
                 ):
        super().__init__(lazy)
        self.tokenizers = tokenizers
        self.indexers = indexers
        self.label_indexers = label_indexers
        self.memory_indexers = memory_indexers
        self.max_sentence_length = max_sentence_length
        self.max_mention_length = max_mention_length
        self.test_ins = test_ins
        self.seq_type = seq_type
        self.shuffle_num = shuffle_num
        self.value_list = open(value_file, 'r').read().strip().split('\n')
        self.bert_thred = bert_thred

    def _read(self, file_path: str) -> Iterable[Instance]:
        instance_num = self.test_ins
        with open(file_path, 'r') as lines:
            for line in lines:
                if self.test_ins > 0:
                    instance_num -= 1
                    if instance_num < 0:
                        break
                data = data_utils.process_onto_kv_line(line, self.value_list, thred=self.bert_thred)
                sentence = data['sentence']
                mention = data['mention']
                labels = data['labels']
                seq_labels = data['seq_labels']
                memory_key = data['memory_key']
                gold_key = data['gold_key']
                memory_value = data['memory_value']
                gold_value = data['gold_value']

                if 'train' in file_path and self.shuffle_num > 0:
                    for i in range(self.shuffle_num):
                        for k, v in labels.items():
                            random.shuffle(labels[k])
                        random.shuffle(seq_labels)
                        yield self.text_to_instance(sentence, mention, labels, seq_labels,
                                                    memory_key, gold_key, memory_value, gold_value)
                else:
                    yield self.text_to_instance(sentence, mention, labels, seq_labels,
                                                memory_key, gold_key, memory_value, gold_value)

    def text_to_instance(self,
                         sentence: List[str],
                         mention: List[str],
                         labels: Dict[str, List[str]] = None,
                         seq_labels: List[str] = None,
                         memory_key: List[str] = None,
                         gold_key: List[int] = None,
                         memory_value: List[str] = None,
                         gold_value: List[int] = None) -> Instance:
        sentence = self.tokenizers[FEATURE_NAME_TRANSFORMER].tokenize(' '.join(sentence))
        if self.max_sentence_length is not None:
            sentence = sentence[:self.max_sentence_length]

        mention = self.tokenizers[FEATURE_NAME_TRANSFORMER].tokenize(' '.join(mention))
        if self.max_mention_length is not None:
            mention = mention[:self.max_mention_length]
        sentence_mention = self.tokenizers[FEATURE_NAME_TRANSFORMER].add_special_tokens(sentence, mention)

        context_token_indexers = {FEATURE_NAME_TRANSFORMER: self.indexers[FEATURE_NAME_TRANSFORMER]}
        context_tokens_field = TextField(sentence_mention, context_token_indexers)
        fields = {'context': context_tokens_field,
                  }

        if labels is not None:
            for key, label in labels.items():
                namespace = key
                assert 'labels' in namespace
                fields[namespace] = MultiLabelField(label, label_namespace=namespace)

        if self.seq_type == 'sequence':
            seq_label_list = seq_labels
        elif self.seq_type == 'layer':
            seq_label_list = []
            for name in ['labels', 'fine_labels', 'ultra_fine_labels']:
                seq_label_list.extend(labels[name])
        else:
            raise ValueError('no such sequence type')

        seq_label_list = seq_label_list + [EOS_SYMBOL]
        seq_label_tokens = [Token(text=label) for label in seq_label_list]
        seq_label_field = TextField(seq_label_tokens, self.label_indexers)
        fields[FEATURE_NAME_SEQ_LABEL] = seq_label_field

        if len(memory_key) > 0:
            key_tokens = [Token(text=label) for label in memory_key]
            key_indexers = {FEATURE_NAME_KEY: self.memory_indexers[FEATURE_NAME_KEY]}
            key_field = TextField(key_tokens, key_indexers)
            fields[FEATURE_NAME_KEY] = key_field

            gold_key_list = []
            for flag in gold_key:
                gold_key_list.append(LabelField(flag, label_namespace='key_labels', skip_indexing=True))
            fields["key_labels"] = ListField(gold_key_list)

            value_tokens = [Token(text=label) for label in memory_value]
            value_indexers = {FEATURE_NAME_VALUE: self.memory_indexers[FEATURE_NAME_VALUE]}
            value_field = TextField(value_tokens, value_indexers)
            fields[FEATURE_NAME_VALUE] = value_field
            gold_value_list = []
            for flag in gold_value:
                gold_value_list.append(LabelField(flag, label_namespace='value_labels', skip_indexing=True))
            fields["value_labels"] = ListField(gold_value_list)

            fields["value2label"] = NamespaceSwappingField(source_tokens=value_tokens,
                                                           target_namespace=NAME_SPACE_SEQ_LABEL)

        return Instance(fields)


# elmo sep mention + LSTM
class SeqElmoTypingReader(DatasetReader):
    def __init__(self,
                 lazy: bool=False,
                 tokenizers: Dict[str, Tokenizer] = None,
                 indexers: Dict[str, TokenIndexer] = None,
                 label_indexers: Dict[str, TokenIndexer] = None,
                 max_sentence_length: int = None,
                 max_mention_length: int = None,
                 test_ins: int = -1,
                 seq_type: str = 'sequence',
                 shuffle_num: int = 0,
                 distant: bool = False,
                 ):
        super().__init__(lazy)
        self.tokenizers = tokenizers
        self.indexers = indexers
        self.label_indexers = label_indexers
        self.max_sentence_length = max_sentence_length
        self.max_mention_length = max_mention_length
        self.test_ins = test_ins
        self.seq_type = seq_type
        self.shuffle_num = shuffle_num
        self.distant = distant

    def _read(self, file_path: str) -> Iterable[Instance]:
        instance_num = self.test_ins
        with open(file_path, 'r') as lines:
            for line in lines:
                if self.test_ins > 0:
                    instance_num -= 1
                    if instance_num < 0:
                        break
                data = data_utils.process_line(line)
                sentence = data['sentence']
                mention = data['mention']
                labels = data['labels']
                seq_labels = data['seq_labels']

                if 'train' in file_path and self.shuffle_num > 0:
                    for i in range(self.shuffle_num):
                        for k,v in labels.items():
                            random.shuffle(labels[k])
                        random.shuffle(seq_labels)
                        yield self.text_to_instance(sentence, mention, labels, seq_labels)
                else:
                    yield self.text_to_instance(sentence, mention, labels, seq_labels)

    def text_to_instance(self,
                         sentence: List[str],
                         mention: List[str],
                         labels: Dict[str, List[str]] = None,
                         seq_labels: List[str] = None) -> Instance:
        # for ontonotes dataset
        if self.max_sentence_length is not None:
            sentence = sentence[:self.max_sentence_length]
        if self.max_mention_length is not None:
            mention = mention[:self.max_mention_length]

        sentence_mention = sentence + ['[SEP]'] + mention + ['[SEP]']
        sentence_mention = self.tokenizers[FEATURE_NAME_ELMO].tokenize(' '.join(sentence_mention))
        mention_token = self.tokenizers[FEATURE_NAME_MENTION_TOKEN].tokenize(' '.join(mention))
        mention_char = self.tokenizers[FEATURE_NAME_CHARACTER].tokenize(' '.join(mention))

        context_token_indexers = {FEATURE_NAME_ELMO: self.indexers[FEATURE_NAME_ELMO]}
        mention_token_indexers = {FEATURE_NAME_MENTION_TOKEN: self.indexers[FEATURE_NAME_MENTION_TOKEN]}
        mention_char_indexers = {FEATURE_NAME_CHARACTER: self.indexers[FEATURE_NAME_CHARACTER]}
        context_tokens_field = TextField(sentence_mention, context_token_indexers)
        mention_tokens_field = TextField(mention_token, mention_token_indexers)
        mention_char_tokens_field = TextField(mention_char, mention_char_indexers)

        fields = {'context': context_tokens_field,
                  'mention': mention_tokens_field,
                  'mention_char': mention_char_tokens_field
                  }

        if labels is not None:
            for key, label in labels.items():
                namespace = key
                assert 'labels' in namespace
                fields[namespace] = MultiLabelField(label, label_namespace=namespace)

        if self.seq_type == 'sequence':
            seq_label_list = seq_labels
        elif self.seq_type == 'layer':
            seq_label_list = []
            for name in ['labels', 'fine_labels', 'ultra_fine_labels']:
                seq_label_list.extend(labels[name])
        else:
            raise ValueError('no such sequence type')

        seq_label_list = seq_label_list + [EOS_SYMBOL]
        seq_label_tokens = [Token(text=label) for label in seq_label_list]
        seq_label_field = TextField(seq_label_tokens, self.label_indexers)
        fields[FEATURE_NAME_SEQ_LABEL] = seq_label_field

        return Instance(fields)


# elmo + class
class ElmoTypingReader(DatasetReader):
    def __init__(self,
                 lazy: bool=False,
                 tokenizers: Dict[str, Tokenizer] = None,
                 indexers: Dict[str, TokenIndexer] = None,
                 label_indexers: Dict[str, TokenIndexer] = None,
                 max_sentence_length: int = None,
                 max_mention_length: int = None,
                 test_ins: int = -1,
                 distant: bool = False,
                 ):
        super().__init__(lazy)
        self.tokenizers = tokenizers
        self.indexers = indexers
        self.label_indexers = label_indexers
        self.max_sentence_length = max_sentence_length
        self.max_mention_length = max_mention_length
        self.test_ins = test_ins
        self.distant = distant

    def _read(self, file_path: str) -> Iterable[Instance]:
        instance_num = self.test_ins
        with open(file_path, 'r') as lines:
            for line in lines:
                if self.test_ins > 0:
                    instance_num -= 1
                    if instance_num < 0:
                        break
                data = data_utils.process_line(line)
                sentence = data['sentence']
                mention = data['mention']
                labels = data['labels']

                yield self.text_to_instance(sentence, mention, labels)

    def text_to_instance(self,
                         sentence: List[str],
                         mention: List[str],
                         labels: Dict[str, List[str]] = None) -> Instance:
        # for ontonotes dataset
        if self.max_sentence_length is not None:
            sentence = sentence[:self.max_sentence_length]
        if self.max_mention_length is not None:
            mention = mention[:self.max_mention_length]

        # sentence_mention = ['[SEP]'] + mention + ['[SEP]'] + sentence
        sentence_mention = sentence + ['[SEP]'] + mention + ['[SEP]']
        sentence_mention = self.tokenizers[FEATURE_NAME_ELMO].tokenize(' '.join(sentence_mention))

        context_token_indexers = {FEATURE_NAME_ELMO: self.indexers[FEATURE_NAME_ELMO]}
        context_tokens_field = TextField(sentence_mention, context_token_indexers)
        fields = {'context': context_tokens_field,
                  }

        if labels is not None:
            for key, label in labels.items():
                namespace = key
                assert 'labels' in namespace
                fields[namespace] = MultiLabelField(label, label_namespace=namespace)

        return Instance(fields)
