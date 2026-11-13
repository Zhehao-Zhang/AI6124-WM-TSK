import unittest

import numpy as np

from benchmarks import make_lstm_fold, make_sequences


class SequenceTests(unittest.TestCase):
    def test_count_alignment_and_past_only(self):
        previous = np.repeat(np.arange(-3, 0)[:, None], 7, axis=1)
        current = np.repeat(np.arange(3)[:, None], 7, axis=1)
        targets = np.array([10, 20, 30])
        for length in (1, 2, 3):
            with self.subTest(length=length):
                windows, labels = make_sequences(previous, current, targets, length)
                self.assertEqual(len(windows), len(targets))
                np.testing.assert_array_equal(windows[:, -1], current)
                np.testing.assert_array_equal(labels, targets)
                for i, window in enumerate(windows):
                    expected = np.arange(i - length + 1, i + 1)
                    np.testing.assert_array_equal(window[:, 0], expected)

    def test_insufficient_history_is_rejected(self):
        with self.assertRaises(ValueError):
            make_sequences(np.empty((1, 7)), np.ones((2, 7)), np.ones(2), 3)

    def test_split_boundaries_use_immediate_history(self):
        data = {}
        for split, rows in (("train", range(5)), ("val", range(5, 8)),
                            ("test", range(8, 10))):
            data[f"X_{split}"] = np.repeat(np.array(list(rows))[:, None], 7, axis=1)
            data[f"y_{split}"] = np.array(list(rows))
        sequences = make_lstm_fold(data, 3)
        self.assertEqual(len(sequences["X_val"]), len(data["y_val"]))
        self.assertEqual(len(sequences["X_test"]), len(data["y_test"]))
        np.testing.assert_array_equal(sequences["X_val"][0, :, 0], [3, 4, 5])
        np.testing.assert_array_equal(sequences["X_test"][0, :, 0], [6, 7, 8])
        np.testing.assert_array_equal(sequences["X_test"][:, -1], data["X_test"])


if __name__ == "__main__":
    unittest.main()
