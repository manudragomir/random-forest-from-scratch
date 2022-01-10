import unittest
from utils.metrics import accuracy
from sklearn.metrics import accuracy_score


class TestingMetrics(unittest.TestCase):
    def test_accuracy(self):
        y_true, y_pred = [0, 0, 1], [0, 0, 1]
        self.assertEqual(accuracy(y_true, y_pred), accuracy_score(y_true, y_pred))

        y_true, y_pred = [0, 0, 1], [0, 0, 0]
        self.assertAlmostEqual(accuracy(y_true, y_pred), accuracy_score(y_true, y_pred))

        y_true, y_pred = [1, 1, 0], [0, 0, 1]
        self.assertEqual(accuracy(y_true, y_pred), accuracy_score(y_true, y_pred))

        y_true, y_pred = [1, 2, 3], [1, 2, 3]
        self.assertEqual(accuracy(y_true, y_pred), accuracy_score(y_true, y_pred))

        y_true, y_pred = [1, 2, 3], [3, 2, 1]
        self.assertAlmostEqual(accuracy(y_true, y_pred), accuracy_score(y_true, y_pred))


if __name__ == '__main__':
    unittest.main()
