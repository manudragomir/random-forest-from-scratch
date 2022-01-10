import math
from typing import List

import numpy as np

from ensembles.RandomForest import RandomForest
from utils.metrics import evaluate


def cross_validation(model, X, y, cv=4):
    X = np.array(X)
    y = np.array(y)

    chunks = np.array_split(np.arange(len(y)), cv)
    fold_indexes = [chunk for chunk in chunks if len(chunk) > 0]

    models = []
    performances = []
    for fold in range(len(fold_indexes)):
        X_train, y_train = [], []
        for curr_fold in range(len(fold_indexes)):
            if curr_fold != fold:
                X_train = X_train + list(X[fold_indexes[curr_fold]])
                y_train = y_train + list(y[fold_indexes[curr_fold]])

        X_train, y_train = np.array(X_train), np.array(y_train)
        X_val, y_val = X[fold_indexes[fold]], y[fold_indexes[fold]]

        model.fit(X_train, y_train)
        models.append(model)
        performances.append(evaluate(y_true=y_val, y_pred=model.predict(X_val)))

    confident_performances = compute_confident_performances(performances, cv)
    return models, confident_performances


def compute_confident_performances(performances: List[dict], k: int):
    accuracy_mean, accuracy_ci = compute_confidence(performances, 'accuracy', k)
    precision_mean, precision_ci = compute_confidence(performances, 'precision', k)
    recall_mean, recall_ci = compute_confidence(performances, 'recall', k)
    f1_mean, f1_ci = compute_confidence(performances, 'f1', k)
    return {
        'accuracy_mean': accuracy_mean,
        'accuracy_confidence': accuracy_ci,
        'precision_mean': precision_mean,
        'precision_confidence': precision_ci,
        'recall_mean': recall_mean, 
        'recall_confidence': recall_ci,
        'f1_mean': f1_mean,
        'f1_confidence': f1_ci
    }


def compute_confidence(performances: List[dict], metric: str, k: int):
    metric_performances = [performance[metric] for performance in performances]
    mean_value = np.mean(np.array(metric_performances))
    std_numerator = abs(metric_performances - mean_value) ** 2
    stddev = math.sqrt(np.mean(std_numerator))
    ci = 1.96 * stddev / math.sqrt(k)
    return mean_value, ci
