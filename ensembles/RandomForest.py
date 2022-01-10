import logging
import math
import random
import time
from multiprocessing import Process, Manager

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from utils.metrics import compute_accuracy, evaluate

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class RandomForest:
    def __init__(self, n_estimators=2, criterion='entropy', splitter='best', max_depth=15, min_samples_split=3,
                 min_samples_leaf=1, n_jobs=4):
        self.__n_estimators = n_estimators
        self.__criterion = criterion
        self.__splitter = splitter
        self.__max_depth = max_depth
        self.__min_samples_split = min_samples_split
        self.__min_samples_leaf = min_samples_leaf
        self.__min_rate_of_subsampling = 0.5
        self.__estimators_features = [None] * n_estimators
        self.__estimators = [None] * n_estimators
        self.__n_jobs = 4 if n_jobs == -1 else n_jobs
        self.training_time = 0

    def fit(self, X, y):
        start = time.time()
        self.__no_classes = len(np.unique(y))

        if self.__n_jobs is None:
            self.train_estimator(list(range(self.__n_estimators)), X, y, self.__estimators, self.__estimators_features)
        else:
            estimators_per_process = math.ceil(self.__n_estimators / self.__n_jobs)
            logger.info(f'estimators_per_process={estimators_per_process}')

            chunks = [list(range(i, min(i + estimators_per_process, self.__n_estimators)))
                      for i in range(0, self.__n_estimators, estimators_per_process)]

            manager = Manager()
            estimators = manager.list(self.__estimators)
            estimators_features = manager.list(self.__estimators_features)
            processes = []
            for i, chunk in enumerate(chunks):
                pr = Process(target=self.train_estimator, args=(chunk, X, y, estimators, estimators_features, i))
                processes.append(pr)
                pr.start()

            for current_process in processes:
                current_process.join()

            self.__estimators = estimators
            self.__estimators_features = estimators_features

        logger.info('DONE training')
        end = time.time()
        self.training_time = end - start

    def train_estimator(self, estimator_indexes, X, y, estimators, estimators_features, process=0):
        logger.info(f'start training process={process} with indexes={estimator_indexes}')
        for estimator_index in estimator_indexes:
            Xb, yb = self.__bootstrapping(X, y)
            Xbs, subset_of_features = self.__select_subset_of_features(Xb)
            new_estimator = DecisionTree(criterion=self.__criterion,
                                         splitter=self.__splitter,
                                         max_depth=self.__max_depth,
                                         min_samples_split=self.__min_samples_split,
                                         min_samples_leaf=self.__min_samples_leaf)
            new_estimator.fit(Xbs, yb)
            estimators[estimator_index] = new_estimator
            estimators_features[estimator_index] = subset_of_features

        logger.info(f'finish training process={process}')

    def analysis(self, y_true, y_pred):
        performances = evaluate(y_true=y_true, y_pred=y_pred)
        performances['n_estimators'] = self.__n_estimators
        performances['max_depth'] = self.__max_depth
        performances['n_jobs'] = self.__n_jobs
        performances['criterion'] = self.__criterion
        performances['training_time'] = self.training_time
        return performances

    def predict(self, X):
        no_samples = len(X)
        rf_predictions = np.zeros((no_samples, self.__no_classes))
        for estimator, estimator_features_indexes in zip(self.__estimators, self.__estimators_features):
            curr_predictions = estimator.predict(X[:, estimator_features_indexes])
            for idx_sample in range(no_samples):
                prediction = curr_predictions[idx_sample]
                rf_predictions[idx_sample][prediction] += 1

        rf_predictions = np.array(rf_predictions)
        return np.argmax(rf_predictions, axis=1)

    def score(self, X, y):
        y_pred = self.predict(X)
        return compute_accuracy(y_pred=y_pred, y_true=y)

    def __bootstrapping(self, X, y):
        bootstrapped_rows = np.random.choice(len(X), size=len(X), replace=True)
        return X[bootstrapped_rows, :], y[bootstrapped_rows]

    def __select_subset_of_features(self, X):
        len_features = len(X[0])
        len_subset_features = random.randint(int(len_features * self.__min_rate_of_subsampling), len_features)
        subset_of_features = np.random.choice(len_features, size=len_subset_features, replace=False)
        Xsubset = X[:, subset_of_features]
        return Xsubset, subset_of_features


# for idx_estimator in range(self.__n_estimators):
        #     Xb, yb = self.__bootstrapping(X, y)
        #     Xbs, subset_of_features = self.__select_subset_of_features(Xb)
        #     new_estimator = DecisionTreeClassifier(criterion=self.__criterion,
        #                                            splitter=self.__splitter,
        #                                            max_depth=self.__max_depth,
        #                                            min_samples_split=self.__min_samples_split,
        #                                            min_samples_leaf=self.__min_samples_leaf)
        #     new_estimator.fit(Xbs, yb)
        #     self.__estimators[idx_estimator] = new_estimator
        #     self.__estimators_features[idx_estimator] = subset_of_features
