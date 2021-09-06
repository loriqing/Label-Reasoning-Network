#!/usr/bin/env bash
# -*- coding:utf-8 -*-

# bert_thred
# kv
python -m entity_typing.train mem_typing \
  -model-path model/ultra_fine/kv_typing_entity_marker_macro_att1_att2_bert0 \
  -label-emb-size 100 \
  -data-folder-path data/open_type_original_lemma_synthetic \
  -value-file probe_experiment/data/all_types.txt \
  -memory-emb data/glove.6B.300d.txt \
  -memory-emb-size 300 \
  -bert-type entity_marker_kv \
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
  -decoder-type lstm2 \
  -evaluation macro \
  -loss-lambda 0 \
  -loss-lambda2 1 \
  -similar-thred 0.2 \
  -bert-thred 0 \
  -sparce-rate 0.9 \
  $*

python -m entity_typing.train mem_typing \
  -model-path model/ultra_fine/kv_typing_entity_marker_macro_att1_att2_bert0.2 \
  -label-emb-size 100 \
  -data-folder-path data/open_type_original_lemma_synthetic \
  -value-file probe_experiment/data/all_types.txt \
  -memory-emb data/glove.6B.300d.txt \
  -memory-emb-size 300 \
  -bert-type entity_marker_kv \
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
  -loss-lambda 0 \
  -loss-lambda2 1 \
  -similar-thred 0.2 \
  -bert-thred 0.2 \
  -sparce-rate 0.9 \
  $*

python -m entity_typing.train mem_typing \
  -model-path model/ultra_fine/kv_typing_entity_marker_macro_att1_att2_bert0.3 \
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
  -bert-thred 0.3 \
  -sparce-rate 0.9 \
  $*

python -m entity_typing.train mem_typing \
  -model-path model/ultra_fine/kv_typing_entity_marker_macro_att1_att2_bert0.4 \
  -label-emb-size 100 \
  -data-folder-path data/open_type_original_lemma_synthetic \
  -value-file probe_experiment/data/all_types.txt \
  -memory-emb data/glove.6B.300d.txt \
  -memory-emb-size 300 \
  -bert-type entity_marker_kv \
  -device 4 \
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
  -bert-thred 0.4 \
  -sparce-rate 0.9 \
  $*

python -m entity_typing.train mem_typing \
  -model-path model/ultra_fine/kv_typing_entity_marker_macro_att1_att2_bert0.2_bert0.5 \
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
  -bert-thred 0.5 \
  -sparce-rate 0.9 \
  $*

python -m entity_typing.train mem_typing \
  -model-path model/ultra_fine/kv_typing_entity_marker_macro_att1_att2_bert0.2_bert1 \
  -label-emb-size 100 \
  -data-folder-path data/open_type_original_lemma_synthetic \
  -value-file probe_experiment/data/all_types.txt \
  -memory-emb data/glove.6B.300d.txt \
  -memory-emb-size 300 \
  -bert-type entity_marker_kv \
  -device 6 \
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
  -bert-thred 1 \
  -sparce-rate 0.9 \
  $*
