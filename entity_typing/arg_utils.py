
def add_evaluate_argument(parser):
    parser.add_argument('-device', dest='device', type=int, default=0)
    parser.add_argument('-model', dest='model', type=str, required=True)
    parser.add_argument('-data', dest='data_path', type=str, required=True)
    parser.add_argument('-batch', dest='batch', type=int, default=32)
    parser.add_argument('-show', dest='show', action='store_true')
    parser.add_argument('-test-ins', dest='test_ins', type=int, default=-1)


def add_view_argument(parser):
    parser.add_argument('-device', dest='device', type=int, default=0)
    parser.add_argument('-model', dest='model', type=str, default=None)
    parser.add_argument('-data', dest='data_path', type=str, default=None)
    parser.add_argument('-batch', dest='batch', type=int, default=32)
    parser.add_argument('-show', dest='show', action='store_true')
    parser.add_argument('-test-ins', dest='test_ins', type=int, default=-1)
    parser.add_argument('-evaluation', dest='evaluation', type=str, default='macro',
                        help="choose evaluation from ['macro', 'micro']")


def add_argument(parser):
    parser.add_argument('-model-path', dest='model_path', default='model/debug_model')
    parser.add_argument('-overwrite-model-path', dest='overwrite_model_path', action='store_true')
    parser.add_argument('-device', dest='device', type=int, default=0)
    parser.add_argument('-test-checkpoint', dest='test_checkpoint', type=str, default=None)
    parser.add_argument('-seed', dest='seed', type=int, default=13370)
    parser.add_argument('-fp16', dest='fp16', choices=['O0', 'O1', 'O2', 'O3'], type=str, default=None)
    parser.add_argument('-test-ins', dest='test_ins', type=int, default=-1)
    parser.add_argument('-evaluation', dest='evaluation', type=str, default='macro',
                        help="choose evaluation from ['macro', 'micro']")

    data_group = parser.add_argument_group('data')

    data_group.add_argument('-cased', dest='lowercase', action='store_false')
    data_group.add_argument('-data-folder-path', dest='data_folder_path', type=str,
                        default='data/open_type/')
    data_group.add_argument('-max-sentence-length', dest='max_sentence_length', type=int, default=None,
                            help='you can set the max sentence length of input')
    data_group.add_argument('-max-mention-length', dest='max_mention_length', type=int, default=None,
                            help='you can set the max mention length of input')
    data_group.add_argument('-lazy', dest='lazy', type=bool, default=False)
    data_group.add_argument('-distant', dest='distant', action='store_true')

    embedding_group = parser.add_argument_group('embedding')

    embedding_group.add_argument('-token-emb-size', dest='token_emb_size', type=int, default=100)
    embedding_group.add_argument('-token-emb', dest='token_emb', type=str, default=None,
                        help='you can use the pretrained embedding or random')
    embedding_group.add_argument('-char-emb-size', dest='char_emb_size', type=int, default=100)
    embedding_group.add_argument('-pos-emb-size', dest='pos_emb_size', type=int, default=50)
    embedding_group.add_argument('-transformer', dest='transformer', type=str, default=None,
                        help='transformer model name you want to use, bert-base-uncased ...')
    embedding_group.add_argument('-transformer-max-length', dest='transformer_max_length', type=int, default=128)
    embedding_group.add_argument('-fine-tune-transformer', dest='transformer_require_grad', action='store_true',
                        help='Fine Tune Transformer Layers')

    embedding_group.add_argument('-elmo', dest='elmo', type=str, default=None,
                                 help='elmo model name you want to use ...')
    embedding_group.add_argument('-elmo-max-length', dest='elmo_max_length', type=int, default=128)
    embedding_group.add_argument('-fine-tune-elmo', dest='elmo_require_grad', action='store_true',
                                 help='Fine Tune ElMo Layers')

    embedding_group.add_argument('-char-encoder-type', dest='char_encoder_type', type=str, default='cnn')

    encoder_group = parser.add_argument_group('encoder')

    encoder_group.add_argument('-context-encoder-type', dest='context_encoder_type', type=str, default='lstm')
    encoder_group.add_argument('-context-encoder-layer', dest='context_encoder_layer', type=int, default=1)
    encoder_group.add_argument('-context-encoder-size', dest='context_encoder_size', type=int, default=100)
    encoder_group.add_argument('-mention-encoder-type', dest='mention_encoder_type', type=str, default='lstm')
    encoder_group.add_argument('-mention-filter-size', dest='mention_filter_size', type=list, default=[2, ])
    encoder_group.add_argument('-mention-encoder-layer', dest='mention_encoder_layer', type=int, default=1)
    encoder_group.add_argument('-mention-encoder-size', dest='mention_encoder_size', type=int, default=100)
    encoder_group.add_argument('-mention-char-encoder-type', dest='mention_char_encoder_type', type=str, default='cnn')
    encoder_group.add_argument('-mention-char-filter-size', dest='mention_char_filter_size', type=list, default=[2, ])
    encoder_group.add_argument('-mention-char-encoder-layer', dest='mention_char_encoder_layer', type=int, default=1)
    encoder_group.add_argument('-mention-char-encoder-size', dest='mention_char_encoder_size', type=int, default=100)
    encoder_group.add_argument('-attention-type', dest='attention_type', type=str, default='add',
                        help="choose attention from ['dot', 'add', 'linear', 'bilinear']")
    encoder_group.add_argument('-activation-type', dest='activation_type', type=str, default='tanh',
                        help="choose activation type")

    optimize_group = parser.add_argument_group('optimize')
    optimize_group.add_argument('-dropout', dest='dropout', type=float, default=0.3)
    optimize_group.add_argument('-batch', dest='batch', type=int, default=32)
    optimize_group.add_argument('-batches-per-epoch', dest='batches_per_epoch', type=int, default=None)
    optimize_group.add_argument('-epoch', dest='epoch', type=int, default=100)
    optimize_group.add_argument('-optim', dest='optim', type=str, default='Adam')
    optimize_group.add_argument('-patience', dest='patience', type=int, default=10)
    optimize_group.add_argument('-lr', dest='lr', default=None, type=float)
    optimize_group.add_argument('-lr-reduce-factor', dest='lr_reduce_factor', default=0.5, type=float)
    optimize_group.add_argument('-lr-reduce-patience', dest='lr_reduce_patience', default=3, type=int)
    optimize_group.add_argument('-lr-diff', dest='lr_diff', action='store_true')
    optimize_group.add_argument('-edit', dest='edit', default=0.05, type=float)

    optimize_group.add_argument('-weight-decay', dest='weight_decay', default=0., type=float)
    optimize_group.add_argument('-grad-norm', dest='grad_norm', default=4., type=float)

