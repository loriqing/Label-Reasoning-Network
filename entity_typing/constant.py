#!/usr/bin/env python
# -*- coding:utf-8 -*-
NIL = "NIL"
NAME_SPACE_TOKEN = "token_vocab"
NAME_SPACE_CONTEXT_TOKEN = "context_token_vocab"
NAME_SPACE_MENTION_TOKEN = "mention_token_vocab"
NAME_SPACE_CHARACTER = "character_vocab"
NAME_SPACE_POS_TAG = "pos_tag_vocab"
NAME_SPACE_LEMMA = "lemma_vocab"

FEATURE_NAME_TOKEN = "tokens"
FEATURE_NAME_CONTEXT_TOKEN = "context_tokens"
FEATURE_NAME_MENTION_TOKEN = "mention_tokens"
FEATURE_NAME_CHARACTER = "token_characters"
FEATURE_NAME_POS_TAG = "pos_tokens"
FEATURE_NAME_TYPE_IDS = "type_ids"
FEATURE_NAME_LEMMA = "lemma_tokens"
FEATURE_NAME_ELMO = "elmo"
FEATURE_NAME_TRANSFORMER = "transformer"

NAME_SPACE_ELMO = "elmo"
NAME_SPACE_LABELS = "labels"
SENT_IDS = "sent_ids"
SENTENCE_LEN = "max_length"
GOLD_SPANS = "source_gold_spans"
MAX_VAL = 9999999

PRECISION = "precision"
RECALL = "recall"
FSCORE = "fscore"
DETECT_PRECESION = "d_precision"
DETECT_RECALL = "d_recall"
DETECT_FSCORE = "d_f1"

# seq2seq typing
OTHER_TYPE_LABEL = "O"
EOS_SYMBOL = '<eos>'
UNK_SYMBOL = '@@UNKNOWN@@'
PADDING_TOKEN = "@@PADDING@@"
MEMORY_SYMBOL = '<mem>'
NAME_SPACE_SEQ_LABEL = "seq_label_vocab"
FEATURE_NAME_SEQ_LABEL = "seq_labels"
# for key=value memory
FEATURE_NAME_MEMORY = "memory"
NAME_SPACE_MEMORY = "memory_vocab"
# for key value mapping memory
FEATURE_NAME_KEY = "memory_key"
NAME_SPACE_KEY = "key_vocab"
FEATURE_NAME_VALUE = "memory_value"
NAME_SPACE_VALUE = "value_vocab"


IGNORE_SERVICE_KEY = {"tokens", "token_start", "token_num"}

BITMATCH_ERROR = 0