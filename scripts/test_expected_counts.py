import unittest
from check_expected_counts import CLASSES, compare_counts


class CountRules(unittest.TestCase):
    def test_five_staged_cases(self):
        expected = dict.fromkeys(CLASSES, 1)
        for counts, delta in [((1, 1, 1, 1), (0, 0, 0, 0)),
                              ((1, 0, 1, 1), (0, -1, 0, 0)),
                              ((2, 1, 1, 1), (1, 0, 0, 0)),
                              ((1, 0, 1, 2), (0, -1, 0, 1)),
                              ((0, 0, 0, 0), (-1, -1, -1, -1))]:
            with self.subTest(counts=counts):
                result = compare_counts(expected, dict(zip(CLASSES, counts)))
                self.assertEqual(result['delta'], dict(zip(CLASSES, delta)))
                self.assertTrue(result['status'].endswith('REVIEW_REQUIRED'))
                self.assertEqual('COUNT_MATCH_' in result['status'], not any(delta))

    def test_invalid_counts(self):
        expected = dict.fromkeys(CLASSES, 1)
        for value in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                compare_counts(expected, {**expected, 'drug001': value})
        with self.assertRaises(ValueError):
            compare_counts(expected, {'drug001': 1})
        with self.assertRaises(ValueError):
            compare_counts(expected, {**expected, 'unknown': 1})


if __name__ == '__main__':
    unittest.main()
