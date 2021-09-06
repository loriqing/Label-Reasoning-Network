import argparse
import logging
import sys

sys.path.append(sys.path[0] + '/../')

from entity_typing import arg_utils
from entity_typing import utils
from entity_typing.train_envs import train_env_map

logger = logging.getLogger(__name__)


def backup(model_path, overwrite_model_path):
    import os, shutil
    def _ignore_copy_files(path, content):
        # print(path)
        # print(content)
        to_ignore = []
        for file_ in content:
            if 'debug_model' in path or '.pyc' in file_ or '.git' in file_:
                to_ignore.append(file_)
        # print(to_ignore)
        # print()
        return to_ignore

    sourceSrcDir = os.getcwd() + '/entity_typing/'
    dstSrcDir = os.path.join(model_path, 'backup/')
    # print(sourceSrcDir, dstSrcDir)
    # if not os.path.exists(dstSrcDir):
    #     os.makedirs(dstSrcDir)

    if overwrite_model_path and os.path.exists(dstSrcDir):
        shutil.rmtree(dstSrcDir)

    shutil.copytree(sourceSrcDir, dstSrcDir, ignore=_ignore_copy_files)


def main(args):
    logger.info(args)

    utils.prepare_env(args)
    utils.prepare_model_path(model_path=args.model_path,
                             overwrite_model_path=args.overwrite_model_path)
    utils.write_args_config(sys.argv, args)

    backup(args.model_path, args.overwrite_model_path)
    train_env_map[args.model_name].train_model(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub_parsers = parser.add_subparsers(dest='model_name')
    sub_parsers.required = True
    for key, value in train_env_map.items():
        sub_parser = sub_parsers.add_parser(key)
        sub_parser.set_defaults(model_name=key)
        arg_utils.add_argument(sub_parser)
        value.add_arguments(sub_parser)

    # arg_utils.add_argument(parser)
    args = parser.parse_args()

    # for debug
    args.model_name = 'mem_typing'
    args.bert_type = 'entity_marker_kv'
    args.overwrite_model_path = True
    # args.data_folder_path = 'data/open_type'
    args.data_folder_path = 'data/open_type_original_lemma_synthetic'
    args.memory_emb = 'data/glove.6B.300d.txt'
    args.memory_emb_size = 300
    # args.data_folder_path = 'data/ontonotes'
    args.epoch = 1
    args.batch = 2
    args.lr = 1e-3
    args.lr_diff = True
    args.device = 6
    args.transformer = 'bert-base-uncased'
    args.transformer_require_grad = True
    args.lr_diff = True
    args.edit = 0.05
    args.teaching_forcing_rate = 0
    args.dropout = 0.5
    args.decoder_dropout = 0.5
    args.loss_type = 'match'
    args.distant = False
    args.decoder_type = 'lstm2'
    args.loss_lambda = 1
    args.detach = False
    args.similar_thred = 0
    args.value_file = 'probe_experiment/data/all_types.txt'
    args.test_ins = 100
    args.activation_type = 'relu'

    main(args)
