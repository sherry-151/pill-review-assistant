"""Unit checks for fixed-camera reference manifests and result classification."""
import unittest
from evaluate_camera_minimal import CASES, assess, validate_cases


class CameraMinimalTests(unittest.TestCase):
    def test_original_manifest(self):
        validate_cases(CASES)

    def test_renamed_manifest(self):
        validate_cases({name.replace('v2-', 'v3-'): c for name, c in CASES.items()})

    def test_missing_case_rejected(self):
        with self.assertRaises(ValueError):
            validate_cases(dict(list(CASES.items())[:-1]))

    def test_duplicate_reference_rejected(self):
        cases = dict(CASES)
        cases['v2-drug001.jpg'] = 'drug002'
        with self.assertRaises(ValueError):
            validate_cases(cases)

    def test_path_rejected(self):
        cases = dict(CASES)
        cases['../drug001.jpg'] = cases.pop('v2-drug001.jpg')
        with self.assertRaises(ValueError):
            validate_cases(cases)

    def test_wrong_class_not_count_match(self):
        result = assess('drug001', [{'class_name': 'drug002'}])
        self.assertTrue(result['wrong_class_present'])
        self.assertFalse(result['counts_match_reference'])
        self.assertFalse(result['no_detection_of_present_pill'])

    def test_empty_and_missed_pill_are_distinct(self):
        self.assertTrue(assess(None, [])['counts_match_reference'])
        self.assertTrue(assess('drug004', [])['no_detection_of_present_pill'])
        self.assertFalse(assess('drug004', [])['counts_match_reference'])


if __name__ == '__main__':
    unittest.main()
