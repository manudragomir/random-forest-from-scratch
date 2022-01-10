import logging
import math
import time

import numpy as np

from utils.metrics import compute_accuracy, evaluate

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Node:
    pass


class DecisionTree:
    def __init__(self, criterion='entropy', splitter='best', max_depth=5, min_samples_split=2, min_samples_leaf=1,
                 max_thresholds=10):
        self.__criterion = criterion
        self.__splitter = splitter
        self.__max_depth = max_depth
        self.__min_samples_split = min_samples_split
        self.__min_samples_leaf = min_samples_leaf
        self.__max_thresholds = max_thresholds
        self.training_time = 0
        self.__root = None

    def fit(self, X, y):
        start = time.time()
        self.__no_classes = len(np.unique(y))
        self.__classes = np.unique(y)
        self.__root = self.__build_decision_tree(X, y, curr_depth=0, max_depth=self.__max_depth)
        logger.info('DONE training')
        end = time.time()
        self.training_time = end - start

    def predict(self, X):
        predictions = []
        for x in X:
            predictions.append(self.__predict_sample(x, self.__root))
        return np.array(predictions)

    def __predict_sample(self, x, node: Node):
        if isinstance(node, Leaf):
            return node.get_max_class()
        if node.question.match(x):
            return self.__predict_sample(x, node.true_branch)
        return self.__predict_sample(x, node.false_branch)

    def score(self, X, y):
        y_pred = self.predict(X)
        return compute_accuracy(y_pred=y_pred, y_true=y)

    def analysis(self, y_true, y_pred):
        performances = evaluate(y_true=y_true, y_pred=y_pred)
        performances['max_depth'] = self.__max_depth
        performances['criterion'] = self.__criterion
        performances['training_time'] = self.training_time
        performances['max_thresholds'] = self.__max_thresholds
        return performances

    def __build_decision_tree(self, X, y, curr_depth, max_depth):
        curr_information_gain, question = self.__find_best_split(X, y)
        if curr_information_gain == -1 or curr_depth == max_depth:
            return Leaf(X, y)

        true_branch_idxs, false_branch_idxs = self.__split_samples_by_question(X, question)

        true_branch_X, false_branch_X = X[true_branch_idxs], X[false_branch_idxs]
        true_branch_y, false_branch_y = y[true_branch_idxs], y[false_branch_idxs]

        true_branch_tree = self.__build_decision_tree(true_branch_X, true_branch_y, curr_depth + 1, max_depth)
        false_branch_tree = self.__build_decision_tree(false_branch_X, false_branch_y, curr_depth + 1, max_depth)

        return DecisionNode(true_branch_tree, false_branch_tree, question)

    def __find_best_split(self, X, y):
        if self.__compute_disorder_grade(y) == 0:
            return -1, None

        no_features = len(X[0])
        best_information_gain = -1
        best_test_question = None
        for feature_idx in range(no_features):
            thresholds = np.unique(X[:, feature_idx])
            # logger.info(f'total thresholds={len(thresholds)} for feature_idx={feature_idx}')

            trimmed_thresholds = np.random.choice(thresholds, size=min(len(thresholds), self.__max_thresholds),
                                                  replace=False)
            # logger.info(f'trimmed thresholds={len(trimmed_thresholds)} for feature_idx={feature_idx}')

            for threshold in trimmed_thresholds:
                test_question = Question(feature_idx, threshold)
                true_question_idxs, false_question_idxs = self.__split_samples_by_question(X, test_question)
                if len(true_question_idxs) == 0 or len(false_question_idxs) == 0:
                    continue
                information_gain = self.__compute_information_gain(y[true_question_idxs], y[false_question_idxs])
                if information_gain > best_information_gain:
                    best_information_gain, best_test_question = information_gain, test_question
        return best_information_gain, best_test_question

    def __compute_information_gain(self, y_true, y_false):
        true_len = len(y_true)
        false_len = len(y_false)
        total = true_len + false_len
        return 2 - (self.__compute_disorder_grade(y_true) * true_len / total +
                    self.__compute_disorder_grade(y_false) * false_len / total)

    def __compute_disorder_grade(self, y):
        entropy = 0
        p = {}
        for idx_class in self.__classes:
            p[idx_class] = np.count_nonzero(y == idx_class) / len(y) if len(y) > 0 else 0

        for idx_class in range(self.__no_classes):
            if p[idx_class] != 0:
                entropy = entropy - p[idx_class] * math.log2(p[idx_class])
        return entropy

    def __split_samples_by_question(self, X, question):
        positives_idx = []
        negatives_idx = []
        for idx, x in enumerate(X):
            positives_idx.append(idx) if question.match(x) else negatives_idx.append(idx)
        return np.array(positives_idx), np.array(negatives_idx)


class DecisionNode(Node):
    def __init__(self, true_branch, false_branch, question):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch


class Leaf(Node):
    def __init__(self, X, y):
        self.__X = X
        self.__y = y
        self.class_proba = {}
        self.__compute_probabilities()

    def __compute_probabilities(self):
        for idx_class in np.unique(self.__y):
            self.class_proba[idx_class] = np.count_nonzero(self.__y == idx_class) / len(self.__y)

    def get_max_class(self):
        return max(self.class_proba, key=self.class_proba.get)

    def get_proba_class(self):
        return np.random.choice(self.class_proba.keys(), p=self.class_proba.values())


class Question:
    def __init__(self, feature_index, threshold):
        self.__feature_index = feature_index
        self.__threshold = threshold

    def match(self, x):
        return x[self.__feature_index] <= self.__threshold
