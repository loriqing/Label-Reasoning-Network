#!/usr/bin/env bash
# -*- coding:utf-8 -*-

python -m entity_typing.view_ultrafine \
  -device 0 \
  -data data/open_type/test.json \
  -model model/ultra_fine/seq_typing_entity_marker_macro_att1_att2 \
  -batch 32 \
  -test-ins -1 \
  -show \
  $*
