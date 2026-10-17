import unittest
from pathlib import Path
import cv2
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import vision_engine as ve


class VisionEngineTests(unittest.TestCase):
    def load(self, name):
        image = cv2.imread(str(ROOT / "data" / "sample_frames" / name))
        self.assertIsNotNone(image, name)
        return image

    def test_board_counts_on_six_reference_frames(self):
        expected = {
            "video1_024.0s_frame1440.jpg": 3,
            "video1_036.0s_frame2160.jpg": 5,
            "video1_054.0s_frame3240.jpg": 3,
            "video1_075.0s_frame4500.jpg": 4,
            "video2_000.0s_frame0.jpg": 4,
            "video3_020.0s_frame1200.jpg": 4,
        }
        for name, count in expected.items():
            image = self.load(name)
            boxes = ve.detect_board_boxes(image)
            standard = sum(ve._white_ratio(image, box) >= 0.57 for box in boxes)
            self.assertEqual(standard, count, name)

    def test_hidden_hero_hand_is_not_fabricated(self):
        result = ve.recognize_hero_cards(self.load("video1_024.0s_frame1440.jpg"))
        self.assertTrue(all(not card["visible_candidate"] for card in result["cards"]))
        self.assertTrue(all(card["rank"] == "待确认" for card in result["cards"]))

    def test_visible_hero_rois_are_marked_as_candidates(self):
        for name in ("video1_000.0s_frame0.jpg", "video2_000.0s_frame0.jpg", "video3_020.0s_frame1200.jpg"):
            result = ve.recognize_hero_cards(self.load(name))
            self.assertEqual(len(result["cards"]), 2)
            self.assertTrue(all(card["visible_candidate"] for card in result["cards"]), name)
        # The video1 hand is visually 9♥/7♥. The exact confidence/status may
        # vary by installed OCR backend (RapidOCR vs. Tesseract), but a single
        # frame must never be treated as final confirmation. Accept either a
        # pending card or an explicit single-frame candidate awaiting temporal
        # confirmation; forbid any auto-confirmed status.
        brown = ve.recognize_hero_cards(self.load("video1_000.0s_frame0.jpg"))
        self.assertEqual([c["rank"] for c in brown["cards"]], ["9", "7"])
        allowed_unconfirmed = {"待确认", "单帧候选（需跨帧确认）"}
        self.assertTrue(all(c["status"] in allowed_unconfirmed for c in brown["cards"]))

    def test_low_confidence_card_is_not_promoted_by_repetition(self):
        tracker = ve.TemporalCardTracker(window_size=3, min_repeats=2, min_confidence=0.65)
        card = {"index": 1, "rank": "3", "suit": "c", "confidence": 0.5, "status": "待确认"}
        result = {"cards": [dict(card)], "hero_hand_candidates": {"cards": []}}
        tracker.update(0.0, result)
        second = tracker.update(1.0, result)
        self.assertEqual(second["cards"][0]["status"], "待确认")

    def test_confident_card_stabilizes_across_frames(self):
        tracker = ve.TemporalCardTracker(window_size=3, min_repeats=2, min_confidence=0.65)
        card = {"index": 1, "rank": "K", "suit": "d", "confidence": 0.9, "status": "单帧候选"}
        result = {"cards": [dict(card)], "hero_hand_candidates": {"cards": []}}
        first = tracker.update(0.0, result)
        second = tracker.update(1.0, result)
        self.assertEqual(first["cards"][0]["status"], "单帧候选（等待下一帧）")
        self.assertTrue(second["cards"][0]["status"].startswith("跨帧稳定候选"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
