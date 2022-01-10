import unittest
from utils.metrics import compute_accuracy, compute_precision, compute_recall, compute_f1
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class TestingMetrics(unittest.TestCase):
    def test_accuracy(self):
        y_true, y_pred = [0, 0, 1], [0, 0, 1]
        self.assertEqual(compute_accuracy(y_true, y_pred), accuracy_score(y_true, y_pred))

        y_true, y_pred = [0, 0, 1], [0, 0, 0]
        self.assertAlmostEqual(compute_accuracy(y_true, y_pred), accuracy_score(y_true, y_pred))

        y_true, y_pred = [1, 1, 0], [0, 0, 1]
        self.assertEqual(compute_accuracy(y_true, y_pred), accuracy_score(y_true, y_pred))

        y_true, y_pred = [1, 2, 3], [1, 2, 3]
        self.assertEqual(compute_accuracy(y_true, y_pred), accuracy_score(y_true, y_pred))

        y_true, y_pred = [1, 2, 3], [3, 2, 1]
        self.assertAlmostEqual(compute_accuracy(y_true, y_pred), accuracy_score(y_true, y_pred))

        y_true, y_pred = [0, 1, 2, 0, 1, 2, 0, 1, 2], [0, 1, 1, 1, 0, 2, 0, 2, 1]
        self.assertAlmostEqual(compute_accuracy(y_true, y_pred), accuracy_score(y_true, y_pred))

    def test_precision(self):
        y_true, y_pred = [0, 0, 1], [0, 0, 1]
        self.assertEqual(compute_precision(y_true, y_pred, binary=True), precision_score(y_true, y_pred))

        y_true, y_pred = [0, 0, 1], [1, 0, 1]
        self.assertAlmostEqual(compute_precision(y_true, y_pred, binary=True), precision_score(y_true, y_pred))

        y_true, y_pred = [1, 1, 0], [0, 0, 1]
        self.assertEqual(compute_precision(y_true, y_pred, binary=True), precision_score(y_true, y_pred))

        y_true, y_pred = [1, 2, 3], [1, 2, 3]
        self.assertEqual(compute_precision(y_true, y_pred), precision_score(y_true, y_pred, average='macro'))

        y_true, y_pred = [1, 2, 3], [3, 2, 1]
        self.assertAlmostEqual(compute_precision(y_true, y_pred), precision_score(y_true, y_pred, average='macro'))

        y_true, y_pred = [0, 1, 2, 0, 1, 2, 0, 1, 2], [0, 1, 1, 1, 0, 2, 0, 2, 1]
        self.assertAlmostEqual(compute_precision(y_true, y_pred), precision_score(y_true, y_pred, average='macro'))

    def test_recall(self):
        y_true, y_pred = [0, 0, 1], [0, 0, 1]
        self.assertEqual(compute_recall(y_true, y_pred, binary=True), recall_score(y_true, y_pred))

        y_true, y_pred = [0, 0, 1], [1, 0, 1]
        self.assertAlmostEqual(compute_recall(y_true, y_pred, binary=True), recall_score(y_true, y_pred))

        y_true, y_pred = [1, 1, 0], [0, 0, 1]
        self.assertEqual(compute_recall(y_true, y_pred, binary=True), recall_score(y_true, y_pred))

        y_true, y_pred = [1, 2, 3], [1, 2, 3]
        self.assertEqual(compute_recall(y_true, y_pred), recall_score(y_true, y_pred, average='macro'))

        y_true, y_pred = [1, 2, 3], [3, 2, 1]
        self.assertAlmostEqual(compute_recall(y_true, y_pred), recall_score(y_true, y_pred, average='macro'))

        y_true, y_pred = [0, 1, 2, 0, 1, 2, 0, 1, 2], [0, 1, 1, 1, 0, 2, 0, 2, 1]
        self.assertAlmostEqual(compute_recall(y_true, y_pred), recall_score(y_true, y_pred, average='macro'))

    def test_f1(self):
        y_true, y_pred = [0, 0, 1], [0, 0, 1]
        self.assertEqual(compute_f1(y_true, y_pred, binary=True), f1_score(y_true, y_pred))

        y_true, y_pred = [0, 0, 1], [1, 0, 1]
        self.assertAlmostEqual(compute_f1(y_true, y_pred, binary=True), f1_score(y_true, y_pred))

        y_true, y_pred = [1, 1, 0, 0], [0, 0, 1, 0]
        self.assertEqual(compute_f1(y_true, y_pred, binary=True), f1_score(y_true, y_pred))

        y_true, y_pred = [1, 2, 3], [1, 2, 3]
        self.assertEqual(compute_f1(y_true, y_pred), f1_score(y_true, y_pred, average='macro'))

        y_true, y_pred = [1, 2, 3], [3, 2, 1]
        self.assertAlmostEqual(compute_f1(y_true, y_pred), f1_score(y_true, y_pred, average='macro'))

        y_true, y_pred = [0, 1, 2, 0, 1, 2, 0, 1, 2], [0, 1, 1, 1, 0, 2, 0, 2, 1]
        self.assertAlmostEqual(compute_f1(y_true, y_pred), f1_score(y_true, y_pred, average='macro'), delta=1e-2)


if __name__ == '__main__':
    unittest.main()
