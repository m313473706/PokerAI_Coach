import unittest
from pathlib import Path

import cv2

import amount_engine as ae

ROOT = Path(__file__).resolve().parents[1]


class AmountEngineTests(unittest.TestCase):
    def test_tournament_amount_keeps_thousands_separator(self):
        display, value, inferred, valid = ae._normalize_token("22,120", "tournament_brown")
        self.assertTrue(valid)
        self.assertEqual(display, "22,120")
        self.assertEqual(value, 22120.0)
        self.assertFalse(inferred)

    def test_explicit_currency_decimal(self):
        display, value, inferred, valid = ae._normalize_token("$0.22", "green_table")
        self.assertTrue(valid)
        self.assertEqual(display, "$0.22")
        self.assertAlmostEqual(value, 0.22)
        self.assertFalse(inferred)

    def test_lost_decimal_is_candidate_not_confirmed_fact(self):
        display, value, inferred, valid = ae._normalize_token("$227", "green_table")
        self.assertTrue(valid)
        self.assertEqual(display, "$2.27")
        self.assertAlmostEqual(value, 2.27)
        self.assertTrue(inferred)

    def test_stack_region_counts_on_three_reference_frames(self):
        cases = {
            "video1_024.0s_frame1440.jpg": 7,
            "video2_000.0s_frame0.jpg": 9,
            "video3_020.0s_frame1200.jpg": 5,
        }
        for filename, expected in cases.items():
            with self.subTest(filename=filename):
                frame = cv2.imread(str(ROOT / "data" / "sample_frames" / filename))
                self.assertIsNotNone(frame)
                boxes = ae.detect_stack_amount_boxes(frame)
                self.assertEqual(len(boxes), expected)

    def test_amount_region_change_gate_ignores_identical_frame(self):
        frame = cv2.imread(str(ROOT / "data" / "sample_frames" / "video2_000.0s_frame0.jpg"))
        self.assertFalse(ae.amount_regions_changed(frame, frame.copy(), "green_table"))
        changed = frame.copy()
        changed[365:440, 590:930] = 0
        self.assertTrue(ae.amount_regions_changed(frame, changed, "green_table"))

    def test_draw_amount_observations_preserves_frame_dimensions(self):
        frame = cv2.imread(str(ROOT / "data" / "sample_frames" / "video2_000.0s_frame0.jpg"))
        annotated = ae.draw_amount_observations(frame, {
            "pot": {"bbox": [590, 365, 340, 75], "candidate": "$0.22", "confidence": 0.9,
                    "status": "单帧候选（需跨帧确认）", "region": "底池"},
            "stacks": [],
        })
        self.assertEqual(annotated.shape, frame.shape)
        self.assertFalse((annotated == frame).all())


if __name__ == "__main__":
    unittest.main()
