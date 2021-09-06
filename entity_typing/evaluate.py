import argparse
import sys
import os
import logging
from pprint import pprint

from allennlp.data import DataLoader
from allennlp.training.util import evaluate

sys.path.append(sys.path[0] + '/../')

from entity_typing import arg_utils
from entity_typing import utils


def eval_model(model_path, data_path, device, batch_size, show, test_ins=-1):
    model, dataset_reader, vocab = utils.load_model(model_path=model_path,
                                              device=device, show=show, test_ins=test_ins)

    test_data = dataset_reader.read(data_path)
    test_data.index_with(vocab)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    model.eval()

    eval_result = evaluate(model=model,
                           data_loader=test_loader,
                           cuda_device=device,
                           batch_weight_key="")
    return eval_result


def main():
    parser = argparse.ArgumentParser()
    arg_utils.add_evaluate_argument(parser)
    args = parser.parse_args()

    utils.pre_logger(os.path.join(args.model, "evaluate.log"))

    eval_result = eval_model(model_path=args.model,
                             data_path=args.data_path,
                             device=args.device,
                             batch_size=args.batch,
                             show=args.show,
                             test_ins=args.test_ins)
    pprint(eval_result)


if __name__ == "__main__":
    main()
