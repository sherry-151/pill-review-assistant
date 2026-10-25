"""Ensure the complete frozen confirmation batch cannot silently lose cases."""
import unittest
from evaluate_camera_confirm01 import expected_cases, validate_confirmation_cases


class ConfirmationTests(unittest.TestCase):
    def test_original_and_derived(self):
        for suffix in ('.jpg', '.png'):
            cases = expected_cases(suffix)
            self.assertEqual(len(cases), 15)
            validate_confirmation_cases(cases)

    def test_missing_rejected(self):
        cases = expected_cases('.jpg')
        cases.pop('v3c01_drug004_003.jpg')
        with self.assertRaises(ValueError):
            validate_confirmation_cases(cases)

    def test_wrong_reference_rejected(self):
        cases = expected_cases('.png')
        cases['v3c01_drug001_001.png'] = 'drug002'
        with self.assertRaises(ValueError):
            validate_confirmation_cases(cases)

    def test_extra_rejected(self):
        cases = expected_cases('.jpg')
        cases['extra.jpg'] = None
        with self.assertRaises(ValueError):
            validate_confirmation_cases(cases)


if __name__ == '__main__':
    unittest.main()
