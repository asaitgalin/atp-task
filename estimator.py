from sklearn.base import BaseEstimator
from scipy.misc import logsumexp

import numpy as np
import pytest


class MultinomialBayesEstimator(BaseEstimator):
    """Estimator for naive bayes text classificator.
    All computations are vectorized.

    :param smooth_factor: (float) not all documents contain all words,
    so some probabilities can be zero and taking log of them causes None.
    This factor is added to each word count to make computations stable.

    fit, predict methods take matrix X, each row represents sample, each column i represents
    i-th word count in the sample.

    See also: https://en.wikipedia.org/wiki/Naive_Bayes_classifier#Document_classification
    """
    def __init__(self, smooth_factor=1.0):
        self._log_word_probs = None
        self._log_class_priors = None
        self._smooth_factor = smooth_factor

    def _check_shape(self, X, y):
        if X.shape[0] != y.shape[0]:
            raise RuntimeError("X and y should have appropriate row dimension (got [{0}, {1}])"
                               .format(X.shape[0], y.shape[0]))

    def fit(self, X, y):
        self._check_shape(X, y)

        sum_y = y.sum(axis=0)
        self._log_class_priors = np.log(np.array([y.shape[0] - sum_y, sum_y]).T) - np.log(y.shape[0])

        # Flatten y
        y_ = y.ravel()
        product = np.concatenate((X[y_ == 0, :].sum(axis=0),
                                  X[y_ == 1, :].sum(axis=0)), axis=0) + self._smooth_factor

        repeated_total_counts = np.repeat(product.sum(axis=1), X.shape[1], axis=1)
        self._log_word_probs = np.log(product) - np.log(repeated_total_counts)

        return self

    def _check_fit(self):
        if self._log_word_probs is None or self._log_class_priors is None:
            raise RuntimeError("Model should be fit before making predictions")

    def _compute_log_likelihood(self, X):
        return X * self._log_word_probs.T + self._log_class_priors

    def predict(self, X):
        self._check_fit()
        scores = self._compute_log_likelihood(X)
        return np.argmax(scores, axis=1)

    def _predict_log_proba(self, X):
        log_likelihood = self._compute_log_likelihood(X)
        normalizer = logsumexp(log_likelihood, axis=1)
        return log_likelihood - normalizer.reshape(-1, 1)

    def predict_proba(self, X):
        self._check_fit()
        return np.exp(self._predict_log_proba(X))


def test_log_class_priors():
    estimator = MultinomialBayesEstimator()
    X = np.matrix(np.random.randint(1, 10, (5, 1)))
    y = np.array([0, 1, 0, 1, 0], dtype=int).reshape(-1, 1)
    estimator.fit(X, y)
    np.testing.assert_almost_equal(estimator._log_class_priors.ravel(),
                                   np.array([np.log(0.6), np.log(0.4)]))


def _get_test_data():
    X = np.matrix([[10, 10, 1], [1, 1, 10], [7, 7, 3]])
    y = np.array([0, 1, 0], dtype=int).reshape(-1, 1)

    log_word_probs = np.concatenate((
        np.log([17, 17, 4]) - np.log(38),
        np.log([1, 1, 10]) - np.log(12))).reshape(2, 3)

    return X, y, log_word_probs


def test_fit():
    estimator = MultinomialBayesEstimator(smooth_factor=0.0)
    X, y, log_word_priors = _get_test_data()
    estimator.fit(X, y)
    np.testing.assert_almost_equal(estimator._log_word_probs, log_word_priors)


def test_predict():
    estimator = MultinomialBayesEstimator(smooth_factor=0.0)
    X, y, log_word_probs = _get_test_data()
    with pytest.raises(RuntimeError):
        estimator.predict(X)
    estimator.fit(X, y)

    X_test = np.matrix([[7, 5, 2], [1, 4, 17]])
    np.testing.assert_almost_equal(np.array(estimator.predict(X_test)).ravel(), np.array([0, 1]))


def test_smooth_factor():
    estimator = MultinomialBayesEstimator(smooth_factor=5.0)
    X = np.matrix([[0, 0, 5], [0, 1, 0]])
    y = np.array([0, 1], dtype=int).reshape(-1, 1)
    estimator.fit(X, y)
    np.testing.assert_almost_equal(np.array(estimator._log_word_probs[0, :]).ravel(),
                                   np.log([5, 5, 10]) - np.log(20))
    np.testing.assert_almost_equal(np.array(estimator._log_word_probs[1, :]).ravel(),
                                   np.log([5, 6, 5]) - np.log(16))
