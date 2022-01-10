import numpy as np


def compute_accuracy(y_true, y_pred):
    return np.mean(np.array(y_pred) == np.array(y_true))


def compute_precision(y_true, y_pred, binary=False):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    if binary is True:
        return ((y_pred == y_true) & (y_pred == 1)).sum() / (y_pred == 1).sum()

    classes = np.unique(np.concatenate([y_pred, y_true]))
    precision_sum = 0
    for idx_class in classes:
        if (y_pred == idx_class).sum():
            precision = ((y_pred == y_true) & (y_pred == idx_class)).sum() / (y_pred == idx_class).sum()
            precision_sum += precision

    return precision_sum / len(classes)


def compute_recall(y_true, y_pred, binary=False):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    if binary is True:
        return ((y_pred == y_true) & (y_pred == 1)).sum() / (y_true == 1).sum()

    classes = np.unique(np.concatenate([y_pred, y_true]))
    recall_sum = 0
    for idx_class in classes:
        if (y_pred == idx_class).sum():
            recall = ((y_pred == y_true) & (y_pred == idx_class)).sum() / (y_true == idx_class).sum()
            recall_sum += recall

    return recall_sum / len(classes)


def sensitivity(y_true, y_pred):
    return compute_recall(y_true, y_pred)


def compute_f1(y_true, y_pred, binary=False):
    precision = compute_precision(y_true, y_pred, binary=binary)
    recall = compute_recall(y_true, y_pred, binary=binary)
    if recall + precision == 0:
        return 0
    return 2 * (precision * recall) / (recall + precision)


def evaluate(y_true, y_pred, binary=False):
    return {
        'accuracy': compute_accuracy(y_true, y_pred),
        'precision': compute_precision(y_true, y_pred, binary=binary),
        'recall': compute_recall(y_true, y_pred, binary=binary),
        'f1': compute_f1(y_true, y_pred, binary=binary)
    }
