#!/usr/bin/env python

from scores_collector import ScoresCollector
from estimator import MultinomialBayesEstimator

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.cross_validation import LeavePOut

import numpy as np
import matplotlib.pyplot as plt

import os
import argparse
import itertools
import logging
import pickle

DESCRIPTION = """Spam detector.

Supported commands:
    train
    classify

If command is "train" then model is trained and compared to sklearn.MultinomialNB model.
Comparison is done with precision, accuracy, recall, f1 scores. Also learning curves are plotted.

If command is "classify" model is loaded and used for prediction.

Examples:
$ ./spam_detector.py train --data-path bare --model-path $(pwd)/model
...
$ ./spam_detector.py classify --model-path model --document-path msg1.txt msg2.txt
1
0
"""

logger = logging.getLogger("SpamDetector")


def _configure_logger():
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler())
    logger.handlers[0].setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))


def load_lingspam_dataset(path):
    parts = []
    for part in os.listdir(path):
        part_documents = []

        part_path = os.path.join(path, part)
        for document_name in os.listdir(part_path):
            is_spam = document_name.startswith("spmsg")
            with open(os.path.join(part_path, document_name)) as f:
                part_documents.append((f.read(), is_spam))

        parts.append(part_documents)

    return parts


def split_train_test_k_fold(dataset):
    for fold_index in xrange(len(dataset)):
        test_set = dataset[fold_index]
        train_set = itertools.chain.from_iterable(
            docs for index, docs in enumerate(dataset) if index != fold_index)
        yield train_set, test_set


def split_train_test_p_out(dataset, p):
    lpo = LeavePOut(10, p=p)
    train_test_list = list(lpo)
    for i in np.random.choice(range(len(train_test_list)), 10):
        train, test = train_test_list[i]
        test_set = itertools.chain.from_iterable(dataset[i] for i in test)
        train_set = itertools.chain.from_iterable(dataset[i] for i in train)
        yield train_set, test_set


def split_train_test(dataset, test_size):
    train_size = len(dataset) - test_size
    if train_size <= 0:
        raise RuntimeError("Test size should be less than data set size")

    permutation = np.random.permutation(len(dataset))
    test_indices = permutation[:test_size]
    train_indices = permutation[test_size:]

    test_set = itertools.chain.from_iterable(dataset[i] for i in test_indices)
    train_set = itertools.chain.from_iterable(dataset[i] for i in train_indices)

    return train_set, test_set


def make_test_train(train_set, test_set, **count_vectorizer_kwargs):
    vectorizer = CountVectorizer(**count_vectorizer_kwargs)

    train_docs, train_labels = zip(*train_set)
    test_docs, test_labels = zip(*test_set)

    X_train = vectorizer.fit_transform(train_docs)
    X_test = vectorizer.transform(test_docs)

    return X_train, np.array(train_labels, dtype=int).reshape(-1, 1), \
           X_test, np.array(test_labels, dtype=int).reshape(-1, 1)


def plot_learning_curves(train_sizes, sklearn_scores, scores, intervals):
    figure, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(12, 5))

    ax1.plot(train_sizes, sklearn_scores)
    ax1.set_xlabel("Train size")
    ax1.set_ylabel("F1 score")

    ax2.errorbar(train_sizes, scores, zip(*intervals))
    ax2.set_xlabel("Train size")
    ax2.set_ylabel("F1 score")

    plt.show()


def f1_score(y_true, y_predict):
    precision = precision_score(y_true, y_predict)
    recall = recall_score(y_true, y_predict)
    return 2 * precision * recall / (precision + recall)


def create_scores_collector():
    return ScoresCollector(accuracy=accuracy_score,
                           precision=precision_score,
                           recall=recall_score,
                           f1=f1_score)


def log_scores(scores_collector, clf_name):
    logger.info("%s scores:", clf_name)
    logger.info("  accuracy {accuracy}, precision {precision}, recall {recall}, f1 {f1} "
                .format(**scores_collector.get_mean_scores()))


def run_k_fold_cross_validation_experiment(dataset):
    logger.info("Starting %d-fold cross-validation...", len(dataset))

    clf_sklearn = MultinomialNB()
    clf = MultinomialBayesEstimator()

    sklearn_scores = create_scores_collector()
    scores = create_scores_collector()

    for train_set, test_set in split_train_test_k_fold(dataset):
        X_train, y_train, X_test, y_test = make_test_train(train_set, test_set)

        # Sklearn
        clf_sklearn.fit(X_train, y_train.ravel())
        predictions = clf_sklearn.predict(X_test)
        sklearn_scores.append_scores(y_test, predictions)

        # Our bayes without ngrams
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        scores.append_scores(y_test, predictions)

    logger.info("%d-fold cross validation finished", len(dataset))
    log_scores(sklearn_scores, "Sklearn")
    log_scores(scores, "MBE")


def run_k_fold_cross_validation_experiment_with_bigrams(dataset):
    logger.info("Starting %d-fold cross-validation...", len(dataset))
    clf = MultinomialBayesEstimator()

    scores_test = create_scores_collector()
    scores_train = create_scores_collector()
    scores_test_bigram = create_scores_collector()
    scores_train_bigram = create_scores_collector()

    # Without bigrams
    for train_set, test_set in split_train_test_k_fold(dataset):
        X_train, y_train, X_test, y_test = make_test_train(train_set, test_set)
        clf.fit(X_train, y_train)
        scores_test.append_scores(y_test, clf.predict(X_test))
        scores_train.append_scores(y_train, clf.predict(X_train))
    # With bigrams
    for train_set, test_set in split_train_test_k_fold(dataset):
        X_train, y_train, X_test, y_test = make_test_train(train_set, test_set, ngram_range=(1, 2))
        clf.fit(X_train, y_train)
        scores_test_bigram.append_scores(y_test, clf.predict(X_test))
        scores_train_bigram.append_scores(y_train, clf.predict(X_train))

    logger.info("%d-fold cross validation finished", len(dataset))
    log_scores(scores_test, "MBE test (w/o bigram)")
    log_scores(scores_train, "MBE train (w/o bigram)")
    log_scores(scores_test_bigram, "MBE test (bigram)")
    log_scores(scores_train_bigram, "MBE train (bigram)")


def calculate_confidence_interval(samples):
    sample_mean = np.mean(samples)
    m = len(samples) * 2
    deltas = []
    for i in xrange(m):
        resample = np.random.choice(samples, len(samples), replace=True)
        deltas.append(np.mean(resample) - sample_mean)
    sorted_deltas = sorted(deltas)
    return (np.abs(sorted_deltas[int(0.95 * m) - 1]),
            np.abs(sorted_deltas[int(0.05 * m) - 1]))


def run_learning_curves_experiment(dataset):
    logger.info("Now starting experiment with learning curves...")
    scores = []
    sklearn_scores = []
    train_sizes = []

    clf = MultinomialBayesEstimator()
    sklearn_clf = MultinomialNB()
    # Constructing confidence intervals using empiric bootstrap
    intervals = []
    for test_size in xrange(1, len(dataset)):
        f_scores = []
        f_scores_sklearn = []
        for train_set, test_set in split_train_test_p_out(dataset, test_size):
            train_set, test_set = split_train_test(dataset, test_size)
            X_train, y_train, X_test, y_test = make_test_train(train_set, test_set)
            clf.fit(X_train, y_train)
            f_scores.append(f1_score(y_test, clf.predict(X_test)))
            sklearn_clf.fit(X_train, y_train.ravel())
            f_scores_sklearn.append(f1_score(y_test, sklearn_clf.predict(X_test)))
        intervals.append(calculate_confidence_interval(f_scores))
        scores.append(np.mean(f_scores))
        sklearn_scores.append(np.mean(f_scores_sklearn))
        train_sizes.append(len(dataset) - test_size)

    plot_learning_curves(train_sizes, sklearn_scores, scores, intervals)


def train(dataset_path, output_model_path):
    logger.info("Starting training...")
    dataset = load_lingspam_dataset(dataset_path)
    logger.info("Loaded train dataset from %s", dataset_path)

    run_k_fold_cross_validation_experiment(dataset)
    run_k_fold_cross_validation_experiment_with_bigrams(dataset)
    run_learning_curves_experiment(dataset)

    logger.info("Learning classifier on all train and dumping model...")
    texts, labels = zip(*itertools.chain.from_iterable(dataset))

    vectorizer = CountVectorizer()
    vectorizer.fit(texts)

    clf = MultinomialBayesEstimator()
    clf.fit(vectorizer.transform(texts), np.array(labels, dtype=int).reshape(-1, 1))

    with open(output_model_path, "wb") as f:
        pickle.dump((clf, vectorizer.vocabulary_), f)

    logger.info("Model saved to %s", output_model_path)


def classify(model_path, document_paths):
    with open(model_path, "rb") as f:
        clf, vocabulary = pickle.load(f)
    vectorizer = CountVectorizer(input='filename', vocabulary=vocabulary)
    X = vectorizer.transform(document_paths)
    for ans in clf.predict(X):
        print ans


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(metavar="command", dest="command")

    train_parser = subparsers.add_parser("train", help="Train classifier on data")
    train_parser.add_argument("--data-path", help="Dataset path", required=True)
    train_parser.add_argument("--model-path", help="Output model path", required=True)

    classify_parser = subparsers.add_parser("classify", help="Use trained model to classify data")
    classify_parser.add_argument("--model-path", help="Path to trained model", required=True)
    classify_parser.add_argument("--document-path", action="append", help="Paths to documents")

    _configure_logger()

    args = dict(vars(parser.parse_args()))
    if args["command"] == "train":
        train(args["data_path"], args["model_path"])
    elif args["command"] == "classify":
        classify(args["model_path"], args["document_path"])
    else:
        assert False, "Undefined command {0}".format(args["command"])


if __name__ == "__main__":
    main()
