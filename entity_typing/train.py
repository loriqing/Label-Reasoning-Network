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
    # save code
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

    args = parser.parse_args()

    main(args)
