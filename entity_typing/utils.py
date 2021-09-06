import argparse
import logging
import os
import torch
import time
from typing import Dict, List
# import sys

from allennlp.models import Model
from allennlp.data import Vocabulary

from entity_typing import data_utils
from entity_typing.constant import FEATURE_NAME_TRANSFORMER, MAX_VAL, FEATURE_NAME_ELMO
from entity_typing.train_envs import train_env_map

logger = logging.getLogger(__name__)


def save_model_options(file_path, options: argparse.Namespace):
    from pprint import pprint
    with open(file_path, 'w') as output:
        pprint(options.__dict__, stream=output)


def load_model_options(file_path):
    namespace = argparse.Namespace()

    with open(file_path, 'r') as fin:
        option_dict = eval(fin.read())

    if isinstance(option_dict, argparse.Namespace):
        raise RuntimeError("The model `%s` is not supported in this version." % file_path)

    for key, value in option_dict.items():
        setattr(namespace, key, value)

    return namespace


def prepare_optimizer(args, model: Model):
    num_parameters = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_parameters += param.numel()
            logging.info("[upd] %s: %s" % (name, param.size()))
        else:
            logging.info("[fix] %s: %s" % (name, param.size()))
    logging.info("Number of trainable parameters: %s", num_parameters)

    optimizer_type = getattr(torch.optim, args.optim)

    if args.transformer is not None:
        context_id = list(map(id, model._context_field_embedder._token_embedders[FEATURE_NAME_TRANSFORMER].parameters()))
        # context_id = list(map(id, model._context_field_embedder.parameters()))
        other_params = filter(lambda p: p.requires_grad and id(p) not in context_id,
                              model.parameters())
        specific_params = filter(lambda p: p.requires_grad and id(p) in context_id,
                                model.parameters())
    if args.elmo is not None:
        context_id = list(
            map(id, model._context_field_embedder._token_embedders[FEATURE_NAME_ELMO].parameters()))
        # context_id = list(map(id, model._context_field_embedder.parameters()))
        other_params = filter(lambda p: p.requires_grad and id(p) not in context_id,
                              model.parameters())
        specific_params = filter(lambda p: p.requires_grad and id(p) in context_id,
                                 model.parameters())

    edit = args.edit

    to_update_params = filter(lambda p: p.requires_grad, model.parameters())
    if args.lr:
        if args.lr_diff and args.transformer:
            optimizer = optimizer_type([{'params': specific_params, 'lr': args.lr * edit},
                                        {'params': other_params, 'lr': args.lr}],
                                       lr=args.lr, weight_decay=args.weight_decay)
        elif args.lr_diff and args.elmo:
            optimizer = optimizer_type([{'params': specific_params, 'lr': args.lr * edit},
                                        {'params': other_params, 'lr': args.lr}],
                                       lr=args.lr, weight_decay=args.weight_decay)
        else:
            optimizer = optimizer_type(to_update_params, lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = optimizer_type(to_update_params, weight_decay=args.weight_decay)

    logging.info(optimizer)
    return optimizer


def get_select_idx(labels):
    sum = labels.sum(dim=1)
    select_idx = torch.where(sum > 0)[-1]
    if len(select_idx) > 0:
        return select_idx
    else:
        return None


def load_model(model_path: str, device: int, show=False, test_ins=-1):   # , evaluation='macro'
    print('Load models from %s ...' % model_path)

    start = time.time()
    model_option = load_model_options(os.path.join(model_path, 'model.option'))
    vocab = Vocabulary.from_files(os.path.join(model_path, 'vocab'))
    model_env = train_env_map[model_option.model_name]

    model_option.test_ins = test_ins
    model_option.device = device
    model_option.show = show
    # model_option.evaluation = evaluation

    dataset_reader = model_env.prepare_dataset_reader(model_option)
    if model_option.token_emb:
        model_option.token_emb = 'random'
    model = model_env.prepare_model(model_option, vocab=vocab)

    with open(os.path.join(model_path, 'best.th'), 'rb') as model_fin:
        model.load_state_dict(torch.load(model_fin, map_location=lambda storage, loc: storage.cpu()))

    model.eval()

    if torch.cuda.is_available() and device >= 0:
        cuda_device = device
        model = model.cuda(cuda_device)
        torch.cuda.set_device(cuda_device)

    print(model)
    print("models Load using %.2fs" % (time.time() - start))

    return model, dataset_reader, vocab


def prepare_env(args):
    import random
    import numpy

    if args.seed >= 0:
        seed = args.seed
    else:
        seed = random.randint(10000, 99999)
    random.seed(seed)
    numpy.random.seed(int(seed / 10))
    torch.manual_seed(int(seed / 100))
    if torch.cuda.is_available():
        torch.cuda.manual_seed(int(seed / 1000))
        torch.backends.cudnn.deterministic = True

    if args.device >= 0:
        torch.cuda.set_device(args.device)


def pre_logger(log_file_name=None, file_handler_level=logging.DEBUG, screen_handler_level=logging.INFO):
    # Logging configuration
    # Set the basic configuration of the logging system
    log_formatter = logging.Formatter(fmt='%(asctime)s [%(processName)s, %(process)s] [%(levelname)-5.5s]  %(message)s',
                                      datefmt='%m-%d %H:%M')
    init_logger = logging.getLogger()
    init_logger.setLevel(logging.INFO)

    if log_file_name:
        # File logger
        file_handler = logging.FileHandler(log_file_name)
        file_handler.setFormatter(log_formatter)
        file_handler.setLevel(file_handler_level)
        init_logger.addHandler(file_handler)

    # Screen logger
    screen_handler = logging.StreamHandler()
    screen_handler.setLevel(screen_handler_level)
    init_logger.addHandler(screen_handler)
    return init_logger


def prepare_model_path(model_path="model/debug_model", overwrite_model_path=False):
    import shutil
    if os.path.exists(model_path):
        if overwrite_model_path:
            shutil.rmtree(model_path)
        else:
            print('models Path: %s is existed, overwrite (y/n)?' % model_path)
            answer = input()
            if answer.strip().lower() == 'y':
                shutil.rmtree(model_path)
            else:
                exit(1)
    os.makedirs(model_path, exist_ok=True)

    pre_logger(os.path.join(model_path, ".log"))

    return model_path


def write_args_config(source_args, parsed_args):
    logger.info('Shell source args:\n{}'.format(" ".join(source_args)))
    message = "Parsed args:\n"
    for arg in vars(parsed_args):
        message += '{} {}\n'.format(arg, getattr(parsed_args, arg))
    logger.info(message)


def convert_output(output, pred_ids, mask,
                   mapping_dic: Dict[str, List[int]], eos_label_id: int,
                   training=True):
    '''
    :param output: (batch_size, decode_len, label_num)
    :param pred_ids: (batch_size, decode_len)
    :param mask: (batch_size, gold_label_len)
    :param mapping_dic
    :param eos_label_id
    :param training
    :return:
    '''
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    check = (pred_ids > eos_label_id)
    assert check.sum().sum() == 0, '<eos> is not in the last of the seq label vocab.'

    batch_size = output.size(0)
    decode_len = pred_ids.size(1)
    label_num = output.size(2)
    gold_label_len = mask.size(1)

    output = output.masked_fill(~mask.unsqueeze(2), -MAX_VAL)

    pred_scores = {}
    measure_scores = {}

    for namespace, (start, end) in mapping_dic.items():
        # (batch_size, decode_len, specific_label_num)  [value, index]
        pred_scores[namespace] = output[:, :, start:end].max(dim=1)[0]

    if training:
        max_scores = output.max(dim=2)[0].unsqueeze(2)
        max_scores = max_scores.masked_fill(~mask.unsqueeze(2), 0)
        for namespace, (start, end) in mapping_dic.items():
            scores = output[:, :, start:end]
            mask = scores < max_scores
            scores = scores.masked_fill(mask, 0)
            scores = scores.masked_fill(~mask, 1)
            # [value, index]
            measure_scores[namespace] = scores.max(dim=1)[0]
    else:
        pred_score = torch.zeros(batch_size, label_num).to(device)
        for i in range(batch_size):
            for j in range(decode_len):
                if pred_ids[i, j] == eos_label_id:
                    break
                else:
                    pred_score[i, pred_ids[i, j]] = 1
        for namespace, (start, end) in mapping_dic.items():
            measure_scores[namespace] = pred_score[:, start:end]

    return pred_scores, measure_scores


def convert_memory_output(output, pred_ids, mask, similar_thred,
                          similar, mapping_dic: Dict[str, List[int]], eos_label_id: int, value2label,
                          training=True, show=False):
    '''
    :param output: (batch_size, decode_len, label_num)
    :param pred_ids: (batch_size, decode_len)
    :param mask: (batch_size, gold_label_len)
    :param similar_thred: float
    :param similar: (batch_size, memory_len)
    :param mapping_dic
    :param eos_label_id
    :param value2label
    :param training
    :return:
    '''
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    check = (pred_ids > eos_label_id)
    assert check.sum().sum() == 0, '<eos> is not in the last of the seq label vocab.'

    batch_size = output.size(0)
    decode_len = pred_ids.size(1)
    label_num = output.size(2)
    gold_label_len = mask.size(1)

    output = output.masked_fill(~mask.unsqueeze(2), -MAX_VAL)

    pred_scores = {}
    measure_scores = {}

    for namespace, (start, end) in mapping_dic.items():
        # (batch_size, decode_len, specific_label_num)  [value, index]
        pred_scores[namespace] = output[:, :, start:end].max(dim=1)[0]

    # (batch_size, memory_len)
    memory_choose = similar > similar_thred
    if show:
        xxx = memory_choose.sum()
        yyy = 0
        if memory_choose.sum() > 0:
            yyy = 1
    memory_scores = output.new_zeros((batch_size, label_num))
    for i in range(batch_size):
        choose_index = value2label[i].masked_select(memory_choose[i])
        if show and yyy == 1:
            print(choose_index)
        memory_scores[i] = memory_scores[i].index_fill(0, choose_index, 1)

    if training:
        max_scores = output.max(dim=2)[0].unsqueeze(2)
        max_scores = max_scores.masked_fill(~mask.unsqueeze(2), 0)
        for namespace, (start, end) in mapping_dic.items():
            scores = output[:, :, start:end]
            mask = scores < max_scores
            scores = scores.masked_fill(mask, 0)
            scores = scores.masked_fill(~mask, 1)
            # [value, index]
            measure_scores[namespace] = scores.max(dim=1)[0] + memory_scores[:, start:end]
    else:
        pred_score = torch.zeros(batch_size, label_num).to(device)
        for i in range(batch_size):
            for j in range(decode_len):
                if pred_ids[i, j] == eos_label_id:
                    break
                else:
                    pred_score[i, pred_ids[i, j]] = 1
        for namespace, (start, end) in mapping_dic.items():
            measure_scores[namespace] = pred_score[:, start:end] + memory_scores[:, start:end]
            # a = torch.where(measure_scores[namespace] == 2)[0]
            # if len(a) > 0:
            #     print('yes!')

    return pred_scores, measure_scores


def mask_average(input, mask):
    '''
    :param input: (batch_size, length, dim)
    :param mask: (batch_size, length)
    :return: output: (batch_size, dim)
    '''
    batch_size = mask.size(0)
    length = mask.size(1)
    weight = torch.ones(batch_size, length)
    pad_value_mask = (mask.logical_not() * -MAX_VAL)  # (bacth_size, length)
    if torch.cuda.is_available():
        weight = weight.to('cuda')
        pad_value_mask = pad_value_mask.to('cuda')
    weight = (weight + pad_value_mask).unsqueeze(2).softmax(1)  # (batch_size, length, 1)
    average = (input * weight).sum(dim=1)  # (batch_size, dim)
    return average


