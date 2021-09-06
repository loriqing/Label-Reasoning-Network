import json
from entity_typing import eval_metric


def score(input_path, evaluation='macro'):
    data = open(input_path, 'r').read().strip().split('\n')
    if 'ontonotes' in input_path:
        gold_labels = {"labels": [], "all": []}
        pred_labels = {"labels": [], "all": []}
        layer_name = ["labels"]
    else:
        gold_labels = {"labels":[], "fine_labels":[], "ultra_fine_labels":[], "all": []}
        pred_labels = {"labels":[], "fine_labels":[], "ultra_fine_labels":[], "all": []}
        layer_name = ["labels", "fine_labels", "ultra_fine_labels"]
    for line in data:
        d = json.loads(line)

        all_gold_labels = []
        all_pred_labels = []
        for name in layer_name:
            gold_labels[name].append(d[name])
            pred_labels[name].append(d["pred_" + name])
            all_gold_labels += d[name]
            all_pred_labels += d["pred_" + name]
        gold_labels["all"].append(all_gold_labels)
        pred_labels["all"].append(all_pred_labels)

    for (n1, true), (n2, pred) in zip(gold_labels.items(), pred_labels.items()):
        assert n1 == n2
        if evaluation == 'macro':
            out = eval_metric.macro(list(zip(true, pred)))
            print(n1, 'macro', out[-3:])
        if evaluation == 'micro':
            out = eval_metric.micro(list(zip(true, pred)))
            print(n1, 'micro', out[-3:])
        if evaluation == 'strict':
            out = eval_metric.strict(list(zip(true, pred)))
            print(n1, 'strict', out[-3:])


if __name__ == "__main__":
    # input_path = "../model/seq_typing_entity_marker_macro_att1/output.json"
    # score(input_path, evaluation='macro')
    # input_path = "../model/seq_typing_entity_marker_micro_att1/output.json"
    # score(input_path, evaluation='micro')
    # input_path = "../model/seq_typing_entity_marker_macro_att1_att2/output_dev.json"
    # score(input_path, evaluation='macro')
    # input_path = "../model/seq_typing_entity_marker_micro_att1_att2/output.json"
    # score(input_path, evaluation='micro')

    # input_path = "../model/bert_typing_model_entitymarker_dropout_macro/output_dev.json"
    # score(input_path, evaluation='macro')
    # input_path = "../model/bert_typing_model_entitymarker_dropout_micro/output.json"
    # score(input_path, evaluation='micro')
    # input_path = "../model/bert_typing_model_entitymarker_macro/output.json"
    # score(input_path, evaluation='macro')
    # input_path = "../model/bert_typing_model_entitymarker_micro/output.json"
    # score(input_path, evaluation='micro')

    # ontonotes original
    # input_path = "../model/ontonotes/bert_typing_ontonotes_macro_sep_5e-4edit0.05/output.json"  # 51.5, 76.6, 69.7
    # input_path = "../model/ontonotes/seq_typing_ontonotes_macro_att1_sep_1e-4edit0.05/output.json"  # 54.0, 76.9, 69.2
    # input_path = "../model/seq_typing_ontonotes_macro_att1_sep_2e-3edit0.01/output.json"  # 57.3, 77.5, 71.4
    # input_path = "../model/ontonotes/seq_typing_ontonotes_macro_att1_att2_sep_1e-4edit0.05/output.json"  # 55.3, 77.3, 70.4
    # input_path = "../model/ontonotes/kv_ontonotes_macro_att1_sep_lemma_1e-4edit0.05/output.json"  # 51.7, 76.1, 70.6
    # input_path = "../model/ontonotes/kv_ontonotes_macro_att1_att2_sep_lemma_1e-4edit0.05/output.json"  # 56.6, 77.6, 71.8

    # ontonotes distant
    # input_path = "../model/ontonotes/bert_typing_ontonotes_macro_sep_distant_4e-4edit0.005/output.json"  # 62.2, 83.4, 78.8
    # input_path = "../model/ontonotes/seq_typing_ontonotes_macro_att1_att2_sep_distant_4e-4edit0.005/output.json"  # 65.8, 84.4, 79.2
    # input_path = "../model/ontonotes/kv_ontonotes_macro_att1_att2_sep_lemma_distant_4e-4edit0.005/output.json"  # 64.5, 84.5, 79.3
    # input_path = "../model/ontonotes/kv_ontonotes_macro_att1_att2_sep_lemma_distant_5e-4edit0.004/output.json"  # 61.9, 84.1, 78.6
    input_path = "../model/ontonotes/seq_typing_ontonotes_macro_distant_4e-4edit0.005_l2/output.json"  # 66.1, 84.8, 80.1

    score(input_path, evaluation='strict')
    score(input_path, evaluation='macro')
    score(input_path, evaluation='micro')
