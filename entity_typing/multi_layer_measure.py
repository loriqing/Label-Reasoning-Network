import torch
import logging

logger = logging.getLogger(__name__)


class MultiLayerF1Measure:

    def __init__(self, label_namespace=['labels'], threshold=0.5, beta=1.0, evaluation='macro'):
        self.label_namespace = label_namespace
        self.evaluation = evaluation
        self._tp, self._tn, self._fp, self._fn = {}, {}, {}, {}
        self._p, self._r, self.pred_example_count, self.gold_label_count = {}, {}, {}, {}
        for namespace in self.label_namespace + ['all']:
            # for micro
            self._tp[namespace] = 0.
            self._tn[namespace] = 0.
            self._fp[namespace] = 0.
            self._fn[namespace] = 0.
            # for macro
            self._p[namespace] = 0
            self._r[namespace] = 0
            self.pred_example_count[namespace] = 0.
            self.gold_label_count[namespace] = 0.

        self._threshold = threshold
        self._beta = beta

    def __call__(self, y_preds, y_trues):
        keys = list(y_preds.keys())
        y_preds = list(y_preds.values())
        y_trues = list(y_trues.values())
        p_list, r_list = [], []
        for (key, y_pred_batch, y_true_batch) in zip(keys, y_preds, y_trues):
            all_tp, all_tn, all_fp, all_fn = 0., 0., 0., 0.
            for y_pred, y_true in zip(y_pred_batch, y_true_batch):
                # for micro
                pred = (y_pred > self._threshold)
                tp = torch.sum((pred.float() * y_true.float())).item()
                tn = torch.sum((1 - pred.float()) * (1 - y_true).float()).item()
                fp = torch.sum(pred.float() * (1 - y_true.float())).item()
                fn = torch.sum((1 - pred.float()) * y_true.float()).item()
                all_tp += tp
                all_tn += tn
                all_fp += fp
                all_fn += fn
                self._tp[key] += tp
                self._tn[key] += tn
                self._fp[key] += fp
                self._fn[key] += fn
                # for macro
                if (tp+fp) > 0:
                    self._p[key] += tp / (tp + fp)
                    self.pred_example_count[key] += 1
                if (tp+fn) > 0:
                    self._r[key] += tp / (tp + fn)
                    self.gold_label_count[key] += 1

        for (y_pred, y_true) in zip(torch.cat(y_preds, dim=1), torch.cat(y_trues, dim=1)):
            # for micro
            pred = (y_pred > self._threshold)
            tp = torch.sum((pred.float() * y_true.float())).item()
            tn = torch.sum((1 - pred.float()) * (1 - y_true).float()).item()
            fp = torch.sum(pred.float() * (1 - y_true.float())).item()
            fn = torch.sum((1 - pred.float()) * y_true.float()).item()
            self._tp["all"] += tp
            self._tn["all"] += tn
            self._fp["all"] += fp
            self._fn["all"] += fn
            # for macro
            if (tp + fp) > 0:
                self._p["all"] += tp / (tp + fp)
                self.pred_example_count["all"] += 1
                p_list.append(tp / (tp + fp))
            if (tp + fn) > 0:
                self._r["all"] += tp / (tp + fn)
                self.gold_label_count["all"] += 1
                r_list.append(tp / (tp + fn))

    def get_metric(self, reset):
        ''' you can choose 'macro' or 'micro' '''
        ret = {}
        all_tp, all_tn, all_fp, all_fn = 0, 0, 0, 0  # for micro
        all_p, all_r = 0, 0  # for macro
        if self.evaluation == 'micro':
            for namespace in self.label_namespace + ['all']:
                tp = self._tp[namespace]
                tn = self._tn[namespace]
                fp = self._fp[namespace]
                fn = self._fn[namespace]
                precision = tp / (tp + fp + 1e-8)
                recall = tp / (tp + fn + 1e-8)
                f1 = 2 * precision * recall / (precision + recall + 1e-8)
                if namespace == 'all':
                    ret["precision"] = precision
                    ret["recall"] = recall
                    ret["fscore"] = f1
                else:
                    ret[namespace + "_p"] = precision
                    ret[namespace + "_r"] = recall
                    ret[namespace + "_f"] = f1
        elif self.evaluation == 'macro':
            for namespace in self.label_namespace + ['all']:
                precision, recall = 0., 0.
                if self.pred_example_count[namespace] > 0:
                    precision = self._p[namespace] / self.pred_example_count[namespace]
                if self.gold_label_count[namespace] > 0:
                    recall = self._r[namespace] / self.gold_label_count[namespace]
                f1 = 2 * precision * recall / (precision + recall + 1e-8)
                if namespace == 'all':
                    ret["precision"] = precision
                    ret["recall"] = recall
                    ret["fscore"] = f1
                else:
                    ret[namespace + "_p"] = precision
                    ret[namespace + "_r"] = recall
                    ret[namespace + "_f"] = f1

        else:
            ValueError('you should choose evaluate from ["micro", "macro"]')

        if reset:
            self._tp, self._tn, self._fp, self._fn = {}, {}, {}, {}
            self._p, self._r, self.pred_example_count, self.gold_label_count = {}, {}, {}, {}
            for namespace in self.label_namespace + ['all']:
                # for micro
                self._tp[namespace] = 0.
                self._tn[namespace] = 0.
                self._fp[namespace] = 0.
                self._fn[namespace] = 0.
                # for macro
                self._p[namespace] = 0
                self._r[namespace] = 0
                self.pred_example_count[namespace] = 0.
                self.gold_label_count[namespace] = 0.

        return ret


def f1(p, r):
  if r == 0.:
    return 0.
  return 2 * p * r / float(p + r)


def macro(true_and_prediction):
  num_examples = len(true_and_prediction)
  p = 0.
  r = 0.
  pred_example_count = 0.
  pred_label_count = 0.
  gold_label_count = 0.
  per_p_list = []
  per_r_list = []
  for true_labels, predicted_labels in true_and_prediction:
    if predicted_labels:
      pred_example_count += 1
      pred_label_count += len(predicted_labels)
      per_p = len(set(predicted_labels).intersection(set(true_labels))) / float(len(predicted_labels))
      p += per_p
      per_p_list.append(per_p)
    if len(true_labels):
      gold_label_count += 1
      per_r = len(set(predicted_labels).intersection(set(true_labels))) / float(len(true_labels))
      r += per_r
      per_r_list.append(per_r)
  if pred_example_count > 0:
    precision = p / pred_example_count
  else:
    precision = 0
  if gold_label_count > 0:
    recall = r / gold_label_count
  else:
    recall = 0
  if pred_example_count == 0:
    avg_elem_per_pred = 0
  else:
    avg_elem_per_pred = pred_label_count / pred_example_count
  return per_p_list, per_r_list, precision, recall, f1(precision, recall)


def micro(true_and_prediction):
  num_examples = len(true_and_prediction)
  num_predicted_labels = 0.
  num_true_labels = 0.
  num_correct_labels = 0.
  pred_example_count = 0.
  for true_labels, predicted_labels in true_and_prediction:
    if predicted_labels:
      pred_example_count += 1
    num_predicted_labels += len(predicted_labels)
    num_true_labels += len(true_labels)
    num_correct_labels += len(set(predicted_labels).intersection(set(true_labels)))
  if pred_example_count == 0:
    return num_examples, 0, 0, 0, 0, 0
  precision = num_correct_labels / num_predicted_labels
  recall = num_correct_labels / num_true_labels
  avg_elem_per_pred = num_predicted_labels / pred_example_count
  return num_examples, pred_example_count, avg_elem_per_pred, precision, recall, f1(precision, recall)


if __name__ == "__main__":

    measure = MultiLayerF1Measure(label_namespace=['label1', 'label2'], evaluate='macro')

    pred_label1 = torch.Tensor([[0,0,1,1], [1,0,0,1]])
    gold_label1 = torch.Tensor([[0,0,0,1], [1,0,0,1]])
    pred_label2 = torch.Tensor([[1,1],[1,0]])
    gold_label2 = torch.Tensor([[1,1],[0,0]])
    measure(y_preds={'label1': pred_label1, 'label2': pred_label2},
            y_trues={'label1': gold_label1, 'label2': gold_label2})
    # ret = measure.get_metric(reset=True)
    # print(ret)

    pred_label1 = torch.Tensor([[1, 1, 1, 1], [0, 0, 0, 1]])
    gold_label1 = torch.Tensor([[0, 0, 0, 0], [0, 0, 0, 1]])
    pred_label2 = torch.Tensor([[0,1],[1,0]])
    gold_label2 = torch.Tensor([[0,0],[1,0]])
    measure(y_preds={'label1': pred_label1, 'label2': pred_label2},
            y_trues={'label1': gold_label1, 'label2': gold_label2})

    ret = measure.get_metric(reset=False)
    print(ret)

    # pred_label1 = [['c', 'd', 'A', 'B'], ['a', 'd', 'A']]
    # gold_label1 = [['d', 'A', 'B'], ['a', 'd']]
    # true = gold_label1
    # pred = pred_label1
    # out = macro(list(zip(true, pred)))
    # print(out)
    # pred_label1 = [['a', 'b', 'c', 'd', 'B'], ['d', 'A']]
    # gold_label1 = [[], ['d', 'A']]
    # true = gold_label1
    # pred = pred_label1
    # out = macro(list(zip(true, pred)))
    # print(out)

    pred_label1 = [['c', 'd', 'A', 'B'], ['a', 'd', 'A'], ['a', 'b', 'c', 'd', 'B'], ['d', 'A']]
    gold_label1 = [['d', 'A', 'B'], ['a', 'd'], [], ['d', 'A']]
    true = gold_label1
    pred = pred_label1
    out = macro(list(zip(true, pred)))
    print(out)
    out = micro(list(zip(true, pred)))
    print(out)
