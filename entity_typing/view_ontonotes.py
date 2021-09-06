import argparse
import sys
import os
import logging
import json
from pprint import pprint

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

    if 'dev' in data_path:
        output_f = open(os.path.join(model_path, 'output_dev.json'), 'w')
    else:
        output_f = open(os.path.join(model_path, 'output.json'), 'w')
    # if '_lemma' in data_path:
    #     output_f = open(os.path.join(model_path, 'find_case.json'), 'w')
    # else:
    #     output_f = None

    id = 0
    for ins in test_data:
        id += 1
        print('----------' + str(id) + '-----------')
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

    # for ontonotes bert
    # args.data_path = 'data/ontonotes/test.json'
    # args.model = "fine-grained-entity-typing/model/ontonotes/bert_typing_ontonotes_macro_sep_5e-4edit0.05"
    # args.model = "fine-grained-entity-typing/model/ontonotes/bert_typing_ontonotes_macro_sep_distant_4e-4edit0.005"

    # for ontonotes sequence typing
    # args.data_path = 'data/ontonotes/test.json'
    # args.model = "fine-grained-entity-typing/model/ontonotes/seq_typing_ontonotes_macro_att1_sep_1e-4edit0.05"
    # args.model = "fine-grained-entity-typing/model/ontonotes/seq_typing_ontonotes_macro_att1_att2_sep_1e-4edit0.05"
    # args.model = "fine-grained-entity-typing/model/ontonotes/seq_typing_ontonotes_macro_att1_att2_sep_distant_4e-4edit0.005"

    # for ontonotes ky typing
    args.data_path = 'data/ontonotes_original_lemma/test.json'
    # args.model = "fine-grained-entity-typing/model/ontonotes/kv_ontonotes_macro_att1_sep_lemma_1e-4edit0.05"
    # args.model = "fine-grained-entity-typing/model/ontonotes/kv_ontonotes_macro_att1_att2_sep_lemma_1e-4edit0.05"
    # args.model = "fine-grained-entity-typing/model/ontonotes/kv_ontonotes_macro_att1_att2_sep_lemma_distant_4e-4edit0.005"
    args.model = "fine-grained-entity-typing/model/ontonotes/kv_ontonotes_macro_att1_att2_sep_lemma_distant_5e-4edit0.004"

    args.test_ins = -1
    args.show = False

    # utils.pre_logger(os.path.join(args.model, "view.log"))

    eval_model(model_path=args.model,
               data_path=args.data_path,
               device=5,
               batch_size=1,
               show=args.show,
               test_ins=args.test_ins,)


if __name__ == "__main__":
    main()
