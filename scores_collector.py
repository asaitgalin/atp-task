from collections import defaultdict
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from itertools import izip
import pytest

EPS = 1e-2


class ScoresCollector(object):
    def __init__(self, **metrics):
        self._metrics = {}
        for key, func in metrics.iteritems():
            self._metrics[key] = func
        self._scores = defaultdict(list)
        self._append_count = 0

    def append_scores(self, y_true, y_predict, p=False):
        for key, func in self._metrics.iteritems():
            self._scores[key].append(func(y_true, y_predict))
        self._append_count += 1

    def get_mean_scores(self):
        if self._append_count == 0:
            raise RuntimeError("Cannot get mean scores. No data")

        result = {}
        for key in self._scores:
            result[key] = np.mean(self._scores[key])

        return result


def _get_len_one_dim(v):
    if hasattr(v, "shape"):
        if len(v.shape) != 1:
            raise RuntimeError("Input vector is not one dimensional")
        return v.shape[0]
    return len(v)


def _check_lenghts(y_true, y_predict):
    len_true = _get_len_one_dim(y_true)
    len_predict = _get_len_one_dim(y_predict)

    if len_true != len_predict:
        raise RuntimeError("y_true and y_predict have different lenghts ({0} != {1})"
                           .format(len_true, len_predict))


def binary_true_positive(y_true, y_predict, true_class_label=1):
    _check_lenghts(y_true, y_predict)
    return np.sum(v1 == true_class_label and v1 == v2
                  for v1, v2 in izip(y_true, y_predict))


def binary_false_positive(y_true, y_predict, true_class_label=1):
    _check_lenghts(y_true, y_predict)
    return np.sum(v1 != true_class_label and v2 == true_class_label
                  for v1, v2 in izip(y_true, y_predict))


# Tests
def test_binary_true_positive():
    assert abs(binary_true_positive([0, 1, 1], [0, 1, 0]) - 1.0) < EPS
    assert abs(binary_true_positive([1, 1, 1], [1, 1, 1]) - 3.0) < EPS


def test_binary_false_positive():
    assert abs(binary_false_positive([0, 0, 1], [0, 1, 0]) - 1.0) < EPS
    assert abs(binary_false_positive([1, 0, 1], [0, 0, 1]) - 0.0) < EPS


def test_scores_collector():
    scores_collector = ScoresCollector(accuracy=accuracy_score,
                                       precision=precision_score,
                                       recall=recall_score,
                                       f1=f1_score,
                                       tp=binary_true_positive,
                                       fp=binary_false_positive)

    with pytest.raises(RuntimeError):
        scores_collector.get_mean_scores()

    scores_collector.append_scores([0, 1, 1], [0, 1, 0])
    scores_collector.append_scores([0, 0, 0], [1, 1, 1])
    scores_collector.append_scores([1, 1, 0], [1, 1, 0])
    mean_scores = scores_collector.get_mean_scores()

    assert abs(mean_scores["accuracy"] - 0.55) < EPS
    assert abs(mean_scores["precision"] - 0.66) < EPS
    assert abs(mean_scores["recall"] - 0.5) < EPS
    assert abs(mean_scores["f1"] - 0.55) < EPS
    assert abs(mean_scores["tp"] - 1.0) < EPS
    assert abs(mean_scores["fp"] - 1.0) < EPS
