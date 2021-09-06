#!/usr/bin/env bash
# -*- coding:utf-8 -*-

python -m entity_typing.evaluate \
  -device 3 \
  -data data/open_type/test.json \
  -model model/ultra_fine/seq_typing_entity_marker_macro_att1_att2 \
  -batch 32 \
  -test-ins -1 \
  -show \
  $*

python -m entity_typing.evaluate \
  -device 3 \
  -data data/open_type_original_lemma_synthetic/test.json \
  -model model/ultra_fine/kv_typing_entity_marker_macro_att1_att2 \
  -batch 32 \
  -test-ins -1 \
  -show \
  $*
