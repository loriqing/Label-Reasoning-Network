import argparse
import sys
import os
import logging
import json
from pprint import pprint
from tqdm import tqdm

from allennlp.data import DataLoader
from allennlp.training.util import evaluate

sys.path.append(sys.path[0] + '/../')

from entity_typing import arg_utils
from entity_typing import utils

logger = logging.getLogger(__name__)


def eval_model(model_path, data_path, device, batch_size, show, test_ins=-1):  # , evaluation='macro'
    model, dataset_reader, vocab = utils.load_model(model_path=model_path,
                                              device=device, show=show, test_ins=test_ins)  # , evaluation=evaluation

    test_data = dataset_reader.read(data_path)
    test_data.index_with(vocab)

    model.eval()

    output_f = None
    if 'dev' in data_path:
        output_f = open(os.path.join(model_path, 'output_dev.json'), 'w')
    else:
        output_f = open(os.path.join(model_path, 'output.json'), 'w')

    for ins in tqdm(test_data):
        output_dict = model.forward_on_instance(ins)

        # context, mention, labels, fine_labels, ultra_fine_labels
        context = ' '.join([token.text for token in ins.fields['context'].tokens])
        if "mention" in ins.fields:
            mention = ' '.join([token.text for token in ins.fields['mention'].tokens])
        labels = ' '.join(ins.fields['labels'].labels)
        if 'fine_labels' in ins.fields:
            fine_labels = ' '.join(ins.fields['fine_labels'].labels)
        if 'ultra_fine_labels' in ins.fields:
            ultra_fine_labels = ' '.join(ins.fields['ultra_fine_labels'].labels)

        if output_f is not None:
            output = {}
            output["context"] = context
            if "mention" in ins.fields:
                output["mention"] = mention
            if "pred_seq_labels" in output_dict:
                output["pred_seq_labels"] = output_dict['pred_seq_labels']

            output["labels"] = ins.fields['labels'].labels
            output["pred_labels"] = output_dict['pred_labels']

            if 'fine_labels' in ins.fields:
                output["fine_labels"] = ins.fields['fine_labels'].labels
                output["pred_fine_labels"] = output_dict['pred_fine_labels']

            if 'ultra_fine_labels' in ins.fields:
                output["ultra_fine_labels"] = ins.fields['ultra_fine_labels'].labels
                output["pred_ultra_fine_labels"] = output_dict['pred_ultra_fine_labels']

            output_f.write(json.dumps(output) + '\n')

    if output_f is not None: output_f.close()
    metric = model.get_metrics()
    print(metric)


def main():
    parser = argparse.ArgumentParser()
    arg_utils.add_view_argument(parser)
    args = parser.parse_args()

    utils.pre_logger(os.path.join(args.model, "view.log"))

    eval_model(model_path=args.model,
               data_path=args.data_path,
               device=3,
               batch_size=1,
               show=args.show,
               test_ins=args.test_ins,)


if __name__ == "__main__":
    main()
