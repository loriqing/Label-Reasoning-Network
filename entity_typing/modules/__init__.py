#!/usr/bin/env python
# -*- coding:utf-8 -*-

from entity_typing.modules.seq2seq_transformer_decoder2 import Seq2SeqTransformerDecoder2
from entity_typing.modules.seq2seq_transformer_decoder import Seq2SeqTransformerDecoder
from entity_typing.modules.seq2seq_decoder import Seq2SeqDecoder
from entity_typing.modules.seq2seq_decoder_attention import Seq2SeqAttnDecoder
from entity_typing.modules.seq2seq_decoder_attention2 import Seq2SeqAttn2Decoder
from entity_typing.modules.seq2mem_decoder_attention import Seq2MemAttnDecoder
from entity_typing.modules.seq2mem_decoder_attention2 import Seq2MemAttn2Decoder
from entity_typing.modules.seq2kv_decoder import Seq2KVDecoder
from entity_typing.modules.seq2kv_decoder_attention import Seq2KVAttnDecoder
from entity_typing.modules.seq2kv_decoder_attention2 import Seq2KVAttn2Decoder
from entity_typing.modules.seq2seq_encoder import Seq2SeqEncoder

from entity_typing.modules.seq2seq_decoder_onto import Seq2SeqDecoderOnto
from entity_typing.modules.seq2seq_decoder_attention2_onto import Seq2SeqAttn2DecoderOnto
from entity_typing.modules.seq2kv_decoder_onto import Seq2KVDecoderOnto
from entity_typing.modules.seq2kv_decoder_attention2_onto import Seq2KVAttn2DecoderOnto

from entity_typing.modules.seq2seq_decoder_uf import Seq2SeqDecoderUF
from entity_typing.modules.seq2seq_decoder_attention2_uf import Seq2SeqAttn2DecoderUF
from entity_typing.modules.seq2kv_decoder_uf import Seq2KVDecoderUF
from entity_typing.modules.seq2kv_decoder_attention2_uf import Seq2KVAttn2DecoderUF

from entity_typing.modules.seq2seq_loss import Seq2SeqLoss
from entity_typing.modules.multi_layer_loss import MultiLayerLoss
from entity_typing.modules.bitpartite_matching_loss import BipartiteMatchingLoss
from entity_typing.modules.similar_loss import SimilarLoss
from entity_typing.modules.mlp_attention import SelfAttentiveSum