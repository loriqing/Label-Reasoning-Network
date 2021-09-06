import argparse
import os
import logging
import torch
import sys

from allennlp.models import Model
from allennlp.common.util import dump_metrics
from allennlp.data import DataLoader
from allennlp.training import GradientDescentTrainer, Checkpointer
from allennlp.training.learning_rate_schedulers import LearningRateScheduler, ReduceOnPlateauLearningRateScheduler
from allennlp.training.util import evaluate

from entity_typing import utils
# from entity_typing.utils import save_model_options, prepare_optimizer
from entity_typing.constant import FSCORE

logger = logging.getLogger(__name__)


def train_model(args,
                model: Model,
                train_loader: DataLoader,
                valid_loader: DataLoader,
                test_loader: DataLoader = None):
    output_model_path = args.model_path
    model.vocab.save_to_files(os.path.join(output_model_path, 'vocab'))
    utils.save_model_options(file_path=os.path.join(output_model_path, 'model.option'),
                       options=args)
    optimizer = utils.prepare_optimizer(args, model)

    if torch.cuda.is_available() and args.device >= 0:
        cuda_device = args.device

        model = model.cuda(cuda_device)
    else:
        cuda_device = -1

    logger.info(model)
    check_pointer = Checkpointer(serialization_dir=output_model_path, num_serialized_models_to_keep=1)
    reduce_on_plateau_learning_rate_scheduler = ReduceOnPlateauLearningRateScheduler(optimizer=optimizer,
                                                                                     mode='max',
                                                                                     factor=args.lr_reduce_factor,
                                                                                     patience=args.lr_reduce_patience,
                                                                                     verbose=True)
    trainer = GradientDescentTrainer(model=model,
                                     optimizer=optimizer,
                                     data_loader=train_loader,
                                     validation_data_loader=valid_loader,
                                     patience=args.patience,
                                     num_epochs=args.epoch,
                                     cuda_device=cuda_device,
                                     serialization_dir=output_model_path,
                                     checkpointer=check_pointer,
                                     validation_metric='+' + FSCORE,
                                     learning_rate_scheduler=reduce_on_plateau_learning_rate_scheduler,
                                     opt_level=args.fp16
                                     )
    train_result = trainer.train()
    dump_metrics(os.path.join(output_model_path, f'metrics.json'), train_result)

    valid_result = {
        'loss': train_result['best_validation_loss'],
        'precision': train_result['best_validation_precision'],
        'recall': train_result['best_validation_recall'],
        'f1': train_result['best_validation_' + FSCORE]
    }

    result_str = "Final Valid Loss: %.4f, P: %.3f, R: %.3f, F1: %.3f\n" % (valid_result['loss'],
                                                                         valid_result['precision'],
                                                                         valid_result['recall'],
                                                                         valid_result['f1'])
    other_result = {}
    for k, v in train_result.items():
        if 'best_validation_' in k:
            other_result[k.split('best_validation_')[-1]] = v
    other_str = str(other_result)
    result_str = result_str + other_str

    logger.info(result_str)

    if test_loader:
        test_result = evaluate(model, test_loader, cuda_device=cuda_device, batch_weight_key="")
        if 'loss' in test_result:
            result_str = "Final Test  Loss: %.4f, P: %.3f, R: %.3f, F1: %.3f\n" % (test_result['loss'],
                                                                                 test_result['precision'],
                                                                                 test_result['recall'],
                                                                                 test_result[FSCORE])
        else:
            result_str = "Final Test P: %.3f, R: %.3f, F1: %.3f\n" % (test_result['precision'],
                                                                      test_result['recall'],
                                                                      test_result[FSCORE])
        other_result = {}
        for k, v in test_result.items():
            if '_' in k:
                other_result[k] = v
        other_str = str(other_result)
        result_str = result_str + other_str
        logger.info(result_str)

    logger.info("models Path: %s" % output_model_path)


if __name__ == "__main__":
    pass
