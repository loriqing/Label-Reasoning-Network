#!/usr/bin/env bash
# -*- coding:utf-8 -*-

# bert baseline
python -m entity_typing.train bert_typing \
  -model-path model/ultra_fine/bert_typing_entity_marker_dropout_macro \
  -device 1 \
  -batch 32 \
  -patience 10 \
  -overwrite-model-path \
  -fine-tune-transformer \
  -transformer bert-base-uncased \
  -lr 1e-3 \
  -lr-diff \
  -edit 0.05 \
  -bert-type entity_marker \
  -evaluation macro \
  $*

# sequence to sequence
python -m entity_typing.train seq_typing \
  -model-path model/ultra_fine/seq_typing_entity_marker_macro \
  -label-emb-size 100 \
  -bert-type entity_marker \
  -device 1 \
  -batch 32 \
  -patience 10 \
  -fine-tune-transformer \
  -transformer bert-base-uncased \
  -lr 1e-3 \
  -overwrite-model-path \
  -lr-diff \
  -edit 0.05 \
  -teaching-forcing-rate 0 \
  -seq-type sequence \
  -shuffle-num 0 \
  -loss-type match \
  -decoder-type lstm0 \
  -evaluation macro \
  $*

python -m entity_typing.train seq_typing \
  -model-path model/ultra_fine/seq_typing_entity_marker_macro_att1 \
  -label-emb-size 100 \
  -bert-type entity_marker \
  -device 2 \
  -batch 32 \
  -patience 10 \
  -fine-tune-transformer \
  -transformer bert-base-uncased \
  -lr 1e-3 \
  -overwrite-model-path \
  -lr-diff \
  -edit 0.05 \
  -teaching-forcing-rate 0 \
  -seq-type sequence \
  -shuffle-num 0 \
  -loss-type match \
  -decoder-type lstm \
  -evaluation macro \
  $*

python -m entity_typing.train seq_typing \
  -model-path model/ultra_fine/seq_typing_entity_marker_macro_att1_att2 \
  -label-emb-size 100 \
  -bert-type entity_marker \
  -device 2 \
  -batch 32 \
  -patience 10 \
  -fine-tune-transformer \
  -transformer bert-base-uncased \
  -lr 1e-3 \
  -overwrite-model-path \
  -lr-diff \
  -edit 0.05 \
  -teaching-forcing-rate 0 \
  -seq-type sequence \
  -shuffle-num 0 \
  -loss-type match \
  -decoder-type lstm2 \
  -evaluation macro \
  $*

python -m entity_typing.train seq_typing \
  -model-path model/ultra_fine/seq_typing_entity_marker_macro_att1_att2_xs \
  -label-emb-size 100 \
  -bert-type entity_marker \
  -device 2 \
  -batch 32 \
  -patience 10 \
  -fine-tune-transformer \
  -transformer bert-base-uncased \
  -lr 1e-3 \
  -overwrite-model-path \
  -lr-diff \
  -edit 0.05 \
  -teaching-forcing-rate 0 \
  -seq-type sequence \
  -shuffle-num 0 \
  -loss-type cross_entropy \
  -decoder-type lstm2 \
  -evaluation macro \
  $*

# kv
python -m entity_typing.train mem_typing \
  -model-path model/ultra_fine/kv_typing_entity_marker_macro \
  -label-emb-size 100 \
  -data-folder-path data/open_type_original_lemma_synthetic \
  -value-file probe_experiment/data/all_types.txt \
  -memory-emb data/glove.6B.300d.txt \
  -memory-emb-size 300 \
  -bert-type entity_marker_kv \
  -device 3 \
  -batch 32 \
  -patience 10 \
  -fine-tune-transformer \
  -transformer bert-base-uncased \
  -lr 1e-3 \
  -overwrite-model-path \
  -lr-diff \
  -edit 0.05 \
  -teaching-forcing-rate 0 \
  -seq-type sequence \
  -shuffle-num 0 \
  -loss-type match \
  -decoder-type lstm0 \
  -evaluation macro \
  -loss-lambda 0 \
  -loss-lambda2 1 \
  -similar-thred 0.1 \
  -sparce-rate 0.9 \
  $*

python -m entity_typing.train mem_typing \
  -model-path model/ultra_fine/kv_typing_entity_marker_macro_att1 \
  -label-emb-size 100 \
  -data-folder-path data/open_type_original_lemma_synthetic \
  -value-file probe_experiment/data/all_types.txt \
  -memory-emb data/glove.6B.300d.txt \
  -memory-emb-size 300 \
  -bert-type entity_marker_kv \
  -device 3 \
  -batch 32 \
  -patience 10 \
  -fine-tune-transformer \
  -transformer bert-base-uncased \
  -lr 1e-3 \
  -overwrite-model-path \
  -lr-diff \
  -edit 0.05 \
  -teaching-forcing-rate 0 \
  -seq-type sequence \
  -shuffle-num 0 \
  -loss-type match \
  -decoder-type lstm \
  -evaluation macro \
  -loss-lambda 0 \
  -loss-lambda2 1 \
  -similar-thred 0.1 \
  -sparce-rate 0.9 \
  $*

python -m entity_typing.train mem_typing \
  -model-path model/ultra_fine/kv_typing_entity_marker_macro_att1_att2 \
  -label-emb-size 100 \
  -data-folder-path data/open_type_original_lemma_synthetic \
  -value-file probe_experiment/data/all_types.txt \
  -memory-emb data/glove.6B.300d.txt \
  -memory-emb-size 300 \
  -bert-type entity_marker_kv \
  -device 3 \
  -batch 32 \
  -patience 10 \
  -fine-tune-transformer \
  -transformer bert-base-uncased \
  -lr 1e-3 \
  -overwrite-model-path \
  -lr-diff \
  -edit 0.05 \
  -teaching-forcing-rate 0 \
  -seq-type sequence \
  -shuffle-num 0 \
  -loss-type match \
  -decoder-type lstm2 \
  -evaluation macro \
  -loss-lambda 0 \
  -loss-lambda2 1 \
  -similar-thred 0.2 \
  -sparce-rate 0.9 \
  $*

python -m entity_typing.train mem_typing \
  -model-path model/ultra_fine/kv_typing_entity_marker_macro_att1_att2_xs \
  -label-emb-size 100 \
  -data-folder-path data/open_type_original_lemma_synthetic \
  -value-file probe_experiment/data/all_types.txt \
  -memory-emb data/glove.6B.300d.txt \
  -memory-emb-size 300 \
  -bert-type entity_marker_kv \
  -device 5 \
  -batch 32 \
  -patience 10 \
  -fine-tune-transformer \
  -transformer bert-base-uncased \
  -lr 1e-3 \
  -overwrite-model-path \
  -lr-diff \
  -edit 0.05 \
  -teaching-forcing-rate 0 \
  -seq-type sequence \
  -shuffle-num 0 \
  -loss-type cross_entropy \
  -decoder-type lstm2 \
  -evaluation macro \
  -loss-lambda 0 \
  -loss-lambda2 1 \
  -similar-thred 0.2 \
  -sparce-rate 0.9 \
  $*

python -m entity_typing.train mem_typing \
  -model-path model/ultra_fine/kv_typing_entity_marker_macro_att1_att2_bert \
  -label-emb-size 100 \
  -data-folder-path data/open_type_original_lemma_synthetic \
  -value-file probe_experiment/data/all_types.txt \
  -memory-emb data/glove.6B.300d.txt \
  -memory-emb-size 300 \
  -bert-type entity_marker_kv \
  -device 5 \
  -batch 32 \
  -patience 10 \
  -fine-tune-transformer \
  -transformer bert-base-uncased \
  -lr 1e-3 \
  -overwrite-model-path \
  -lr-diff \
  -edit 0.05 \
  -teaching-forcing-rate 0 \
  -seq-type sequence \
  -shuffle-num 0 \
  -loss-type match \
  -decoder-type lstm2 \
  -evaluation macro \
  -loss-lambda 0 \
  -loss-lambda2 1 \
  -similar-thred 0.2 \
  -sparce-rate 0.9 \
  $*
