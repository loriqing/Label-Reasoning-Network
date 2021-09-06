#!/usr/bin/env bash
# -*- coding:utf-8 -*-

python -m entity_typing.train seq_typing \
  -model-path model/ontonotes/seq_typing_ontonotes_macro_distant_4e-4edit0.005_l2 \
  -data-folder-path data/ontonotes \
  -label-emb-size 100 \
  -bert-type sep_mention \
  -device 4 \
  -batch 32 \
  -patience 5 \
  -fine-tune-transformer \
  -transformer bert-base-uncased \
  -lr 4e-4 \
  -lr-diff \
  -edit 0.005 \
  -overwrite-model-path \
  -teaching-forcing-rate 0 \
  -seq-type sequence \
  -shuffle-num 0 \
  -loss-type match \
  -decoder-type lstm0 \
  -evaluation macro \
  -max-sentence-length 128 \
  -max-mention-length 20 \
  -batches-per-epoch 300 \
  -dropout 0.3 \
  -decoder-dropout 0.6 \
  -distant \
  -optim Adam \
  -test-ins 100 \
  $*

python -m entity_typing.train mem_typing \
  -model-path model/final/kv_ontonotes_macro_att1_att2_distant_4e-4edit0.005_l2_RD0 \
  -label-emb-size 100 \
  -data-folder-path data/ontonotes_original_lemma \
  -value-file probe_experiment/data/onto_ontology.txt \
  -memory-emb probe_experiment/data/glove/glove.ontonotes_all.300d.txt \
  -memory-emb-size 300 \
  -bert-type sep_mention_kv \
  -device 5 \
  -batch 32 \
  -patience 5 \
  -fine-tune-transformer \
  -transformer bert-base-uncased \
  -lr 4e-4 \
  -lr-diff \
  -edit 0.005 \
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
  -distant \
  -optim Adam \
  -test-ins 100 \
  $*

python -m entity_typing.train mem_typing \
  -model-path model/kv_ontonotes_macro_att1_att2_distant_4e-4edit0.005_l4 \
  -label-emb-size 100 \
  -data-folder-path data/ontonotes_original_lemma \
  -value-file probe_experiment/data/onto_ontology.txt \
  -memory-emb probe_experiment/data/glove/glove.ontonotes_all.300d.txt \
  -memory-emb-size 300 \
  -bert-type sep_mention_kv \
  -device 4 \
  -batch 32 \
  -patience 5 \
  -fine-tune-transformer \
  -transformer bert-base-uncased \
  -lr 4e-4 \
  -lr-diff \
  -edit 0.005 \
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
  -distant \
  -optim Adam \
  -test-ins 100 \
  $*

###### sep mention
# bert baseline
python -m entity_typing.train bert_typing \
  -model-path model/ontonotes/bert_typing_ontonotes_macro_sep_distant_4e-4edit0.005 \
  -data-folder-path data/ontonotes \
  -bert-type sep_mention \
  -device 2 \
  -batch 32 \
  -patience 3 \
  -fine-tune-transformer \
  -transformer bert-base-uncased \
  -lr 4e-4 \
  -lr-diff \
  -edit 0.005 \
  -overwrite-model-path \
  -evaluation macro \
  -max-sentence-length 128 \
  -max-mention-length 20 \
  -dropout 0.5 \
  -batches-per-epoch 300 \
  -distant \
  -test-ins 100 \
  $*

python -m entity_typing.train bert_typing \
  -model-path model/ontonotes/bert_typing_ontonotes_macro_sep_distant_5e-4edit0.004 \
  -data-folder-path data/ontonotes \
  -bert-type sep_mention \
  -device 3 \
  -batch 32 \
  -patience 3 \
  -fine-tune-transformer \
  -transformer bert-base-uncased \
  -lr 5e-4 \
  -lr-diff \
  -edit 0.004 \
  -overwrite-model-path \
  -evaluation macro \
  -max-sentence-length 128 \
  -max-mention-length 20 \
  -dropout 0.5 \
  -batches-per-epoch 300 \
  -distant \
  -test-ins 100 \
  $*

# sequence to sequence Sep mention
python -m entity_typing.train seq_typing \
  -model-path model/ontonotes/seq_typing_ontonotes_macro_att1_sep_distant_4e-4edit0.005 \
  -data-folder-path data/ontonotes \
  -label-emb-size 100 \
  -bert-type sep_mention \
  -device 1 \
  -batch 32 \
  -patience 3 \
  -fine-tune-transformer \
  -transformer bert-base-uncased \
  -lr 4e-4 \
  -lr-diff \
  -edit 0.005 \
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
  -distant \
  -test-ins 100 \
  $*

python -m entity_typing.train seq_typing \
  -model-path model/ontonotes/seq_typing_ontonotes_macro_att1_sep_distant_5e-4edit0.004 \
  -data-folder-path data/ontonotes \
  -label-emb-size 100 \
  -bert-type sep_mention \
  -device 2 \
  -batch 32 \
  -patience 3 \
  -fine-tune-transformer \
  -transformer bert-base-uncased \
  -lr 5e-4 \
  -lr-diff \
  -edit 0.004 \
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
  -distant \
  -test-ins 100 \
  $*

python -m entity_typing.train seq_typing \
  -model-path model/ontonotes/seq_typing_ontonotes_macro_att1_att2_sep_distant_4e-4edit0.005 \
  -data-folder-path data/ontonotes \
  -label-emb-size 100 \
  -bert-type sep_mention \
  -device 3 \
  -batch 32 \
  -patience 5 \
  -fine-tune-transformer \
  -transformer bert-base-uncased \
  -lr 4e-4 \
  -lr-diff \
  -edit 0.005 \
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
  -distant \
  -optim Adam \
  -test-ins 100 \
  $*

python -m entity_typing.train seq_typing \
  -model-path model/ontonotes/seq_typing_ontonotes_macro_att1_att2_sep_distant_5e-4edit0.004 \
  -data-folder-path data/ontonotes \
  -label-emb-size 100 \
  -bert-type sep_mention \
  -device 7 \
  -batch 32 \
  -patience 5 \
  -fine-tune-transformer \
  -transformer bert-base-uncased \
  -lr 5e-4 \
  -lr-diff \
  -edit 0.004 \
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
  -distant \
  -optim Adam \
  -test-ins 100 \
  $*

# -dropout 0.3 -decoder-dropout 0.6  att1_att2
# -dropout 0.5 -decoder-dropout 0.5  att1
# -lr 4e-4edit0.005  5e-4edit0.004 -distant

# kv
python -m entity_typing.train mem_typing \
  -model-path model/ontonotes/kv_ontonotes_macro_att1_sep_lemma_distant_4e-4edit0.005 \
  -label-emb-size 100 \
  -data-folder-path data/ontonotes_original_lemma \
  -value-file probe_experiment/data/onto_ontology.txt \
  -memory-emb probe_experiment/data/glove/glove.ontonotes_all.300d.txt \
  -memory-emb-size 300 \
  -bert-type sep_mention_kv \
  -device 2 \
  -batch 32 \
  -patience 3 \
  -fine-tune-transformer \
  -transformer bert-base-uncased \
  -lr 4e-4 \
  -lr-diff \
  -edit 0.005 \
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
  -distant \
  -test-ins 100 \
  $*

python -m entity_typing.train mem_typing \
  -model-path model/ontonotes/kv_ontonotes_macro_att1_sep_lemma_distant_5e-4edit0.004 \
  -label-emb-size 100 \
  -data-folder-path data/ontonotes_original_lemma \
  -value-file probe_experiment/data/onto_ontology.txt \
  -memory-emb probe_experiment/data/glove/glove.ontonotes_all.300d.txt \
  -memory-emb-size 300 \
  -bert-type sep_mention_kv \
  -device 3 \
  -batch 32 \
  -patience 3 \
  -fine-tune-transformer \
  -transformer bert-base-uncased \
  -lr 5e-4 \
  -lr-diff \
  -edit 0.004 \
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
  -distant \
  -test-ins 100 \
  $*

python -m entity_typing.train mem_typing \
  -model-path model/ontonotes/kv_ontonotes_macro_att1_att2_sep_lemma_distant_4e-4edit0.005 \
  -label-emb-size 100 \
  -data-folder-path data/ontonotes_original_lemma \
  -value-file probe_experiment/data/onto_ontology.txt \
  -memory-emb probe_experiment/data/glove/glove.ontonotes_all.300d.txt \
  -memory-emb-size 300 \
  -bert-type sep_mention_kv \
  -device 6 \
  -batch 32 \
  -patience 3 \
  -fine-tune-transformer \
  -transformer bert-base-uncased \
  -lr 4e-4 \
  -lr-diff \
  -edit 0.005 \
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
  -distant \
  -test-ins 100 \
  $*

python -m entity_typing.train mem_typing \
  -model-path model/ontonotes/kv_ontonotes_macro_att1_att2_sep_lemma_distant_5e-4edit0.004 \
  -label-emb-size 100 \
  -data-folder-path data/ontonotes_original_lemma \
  -value-file probe_experiment/data/onto_ontology.txt \
  -memory-emb probe_experiment/data/glove/glove.ontonotes_all.300d.txt \
  -memory-emb-size 300 \
  -bert-type sep_mention_kv \
  -device 6 \
  -batch 32 \
  -patience 3 \
  -fine-tune-transformer \
  -transformer bert-base-uncased \
  -lr 5e-4 \
  -lr-diff \
  -edit 0.004 \
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
  -distant \
  -test-ins 100 \
  $*

