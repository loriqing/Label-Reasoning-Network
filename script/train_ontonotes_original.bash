#!/usr/bin/env bash
# -*- coding:utf-8 -*-

###### sep mention
# bert baseline
python -m entity_typing.train bert_typing \
  -model-path model/ontonotes/bert_typing_ontonotes_macro_sep_5e-4edit0.05 \
  -data-folder-path data/ontonotes \
  -bert-type sep_mention \
  -device 0 \
  -batch 32 \
  -patience 3 \
  -fine-tune-transformer \
  -transformer bert-base-uncased \
  -lr 5e-4 \
  -lr-diff \
  -edit 0.05 \
  -overwrite-model-path \
  -evaluation macro \
  -max-sentence-length 128 \
  -max-mention-length 20 \
  -dropout 0.5 \
  -batches-per-epoch 300 \
  $*

python -m entity_typing.train bert_typing \
  -model-path model/ontonotes/bert_typing_ontonotes_macro_sep_1e-4edit0.05 \
  -data-folder-path data/ontonotes \
  -bert-type sep_mention \
  -device 1 \
  -batch 32 \
  -patience 3 \
  -fine-tune-transformer \
  -transformer bert-base-uncased \
  -lr 1e-4 \
  -lr-diff \
  -edit 0.05 \
  -overwrite-model-path \
  -evaluation macro \
  -max-sentence-length 128 \
  -max-mention-length 20 \
  -dropout 0.5 \
  -batches-per-epoch 300 \
  $*

# sequence to sequence Sep mention
python -m entity_typing.train seq_typing \
  -model-path model/ontonotes/seq_typing_ontonotes_macro_att1_sep_1e-4edit0.05 \
  -data-folder-path data/ontonotes \
  -label-emb-size 100 \
  -bert-type sep_mention \
  -device 6 \
  -batch 32 \
  -patience 3 \
  -fine-tune-transformer \
  -transformer bert-base-uncased \
  -lr 1e-4 \
  -lr-diff \
  -edit 0.05 \
  -overwrite-model-path \
  -teaching-forcing-rate 0 \
  -seq-type sequence \
  -shuffle-num 0 \
  -loss-type match \
  -decoder-type lstm \
  -evaluation macro \
  -max-sentence-length 128 \
  -max-mention-length 20 \
  -batches-per-epoch 300 \
  -dropout 0.5 \
  -decoder-dropout 0.5 \
  $*

python -m entity_typing.train seq_typing \
  -model-path model/ontonotes/seq_typing_ontonotes_macro_att1_att2_sep_1e-4edit0.05 \
  -data-folder-path data/ontonotes \
  -label-emb-size 100 \
  -bert-type sep_mention \
  -device 3 \
  -batch 32 \
  -patience 5 \
  -fine-tune-transformer \
  -transformer bert-base-uncased \
  -lr 1e-4 \
  -lr-diff \
  -edit 0.05 \
  -overwrite-model-path \
  -teaching-forcing-rate 0 \
  -seq-type sequence \
  -shuffle-num 0 \
  -loss-type match \
  -decoder-type lstm2 \
  -evaluation macro \
  -max-sentence-length 128 \
  -max-mention-length 20 \
  -batches-per-epoch 300 \
  -dropout 0.3 \
  -decoder-dropout 0.6 \
  $*

# -dropout 0.3 -decoder-dropout 0.6  att1_att2
# -dropout 0.5 -decoder-dropout 0.5  att1

# kv
python -m entity_typing.train mem_typing \
  -model-path model/ontonotes/kv_ontonotes_macro_att1_sep_lemma_1e-4edit0.05 \
  -label-emb-size 100 \
  -data-folder-path data/ontonotes_original_lemma \
  -value-file probe_experiment/data/onto_ontology.txt \
  -memory-emb probe_experiment/data/glove/glove.ontonotes_all.300d.txt \
  -memory-emb-size 300 \
  -bert-type sep_mention_kv \
  -device 4 \
  -batch 32 \
  -patience 3 \
  -fine-tune-transformer \
  -transformer bert-base-uncased \
  -lr 1e-4 \
  -lr-diff \
  -edit 0.05 \
  -overwrite-model-path \
  -teaching-forcing-rate 0 \
  -seq-type sequence \
  -shuffle-num 0 \
  -loss-type match \
  -decoder-type lstm \
  -evaluation macro \
  -max-sentence-length 128 \
  -max-mention-length 20 \
  -batches-per-epoch 300 \
  -dropout 0.5 \
  -decoder-dropout 0.5 \
  -loss-lambda 0 \
  -loss-lambda2 1 \
  -similar-thred 0.1 \
  $*

python -m entity_typing.train mem_typing \
  -model-path model/ontonotes/kv_ontonotes_macro_att1_att2_sep_lemma_1e-4edit0.05 \
  -label-emb-size 100 \
  -data-folder-path data/ontonotes_original_lemma_synthetic \
  -value-file probe_experiment/data/onto_ontology.txt \
  -memory-emb probe_experiment/data/glove/glove.ontonotes_all.300d.txt \
  -memory-emb-size 300 \
  -bert-type sep_mention_kv \
  -device 5 \
  -batch 32 \
  -patience 3 \
  -fine-tune-transformer \
  -transformer bert-base-uncased \
  -lr 1e-4 \
  -lr-diff \
  -edit 0.05 \
  -overwrite-model-path \
  -teaching-forcing-rate 0 \
  -seq-type sequence \
  -shuffle-num 0 \
  -loss-type match \
  -decoder-type lstm2 \
  -evaluation macro \
  -max-sentence-length 128 \
  -max-mention-length 20 \
  -batches-per-epoch 300 \
  -dropout 0.3 \
  -decoder-dropout 0.6 \
  -loss-lambda 0 \
  -loss-lambda2 1 \
  -similar-thred 0.1 \
  $*
