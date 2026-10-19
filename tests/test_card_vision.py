"""Synthetic contract tests; these exercise code paths, not real-card accuracy."""
import json
import unittest

import numpy as np

from card_vision import (
    CARD_LABELS,
    MIN_CLASS_MARGIN,
    MODEL_SCHEMA,
    TableStateGate,
    CardVisionError,
    decode_detections,
    inspect_table_state,
    validate_analysis_gate,
    validate_model_contract,
)


def card(candidate, zone, x, confidence=0.95, margin=0.8, source="牌面视觉模型"):
    return {
        "candidate": candidate,
        "zone": zone,
        "confidence": confidence,
        "margin": margin,
        "source": source,
        "center_normalized": [x, 0.76 if zone == "hero候选" else 0.44],
        "stable": False,
    }


def output_row(label, box=(100, 100, 200, 300), top=0.95, second=0.10):
    row = np.zeros(57, dtype=np.float32)
    row[:4] = box
    row[4] = 1.0
    row[5 + CARD_LABELS.index(label)] = top
    row[5 + (CARD_LABELS.index(label) + 1) % 52] = second
    return row.reshape(1, 57)


class ModelContractTests(unittest.TestCase):
    def metadata(self):
        return {
            "pokerai_schema": MODEL_SCHEMA,
            "class_labels": json.dumps(CARD_LABELS),
            "license": "MIT",
            "source_url": "https://example.org/model",
            "license_url": "https://example.org/license",
        }

    def test_accepts_fixed_52_class_contract(self):
        self.assertTrue(validate_model_contract(
            self.metadata(), [1, 3, 640, 640], [1, "N", 57]
        ))

    def test_rejects_unverified_or_mismatched_license_and_labels(self):
        metadata = self.metadata()
        metadata["license"] = "AGPL-3.0"
        with self.assertRaises(CardVisionError):
            validate_model_contract(metadata, [1, 3, 640, 640], [1, "N", 57])
        metadata = self.metadata()
        metadata["class_labels"] = json.dumps(list(reversed(CARD_LABELS)))
        with self.assertRaises(CardVisionError):
            validate_model_contract(metadata, [1, 3, 640, 640], [1, "N", 57])


class DecoderTests(unittest.TestCase):
    def test_decodes_rank_suit_confidence_margin_and_coordinates(self):
        boxes, values = decode_detections(
            output_row("9C"), (640, 640), 1.0, (0, 0)
        )
        self.assertEqual(len(boxes), 1)
        self.assertEqual(values[0][0], "9♣")
        self.assertAlmostEqual(values[0][1], 0.95, places=5)
        self.assertGreaterEqual(values[0][2], MIN_CLASS_MARGIN)
        self.assertEqual(boxes[0], [100, 100, 200, 300])

    def test_rejects_low_confidence_and_ambiguous_class(self):
        weak = output_row("9C", top=0.70, second=0.68)
        boxes, _ = decode_detections(weak, (640, 640), 1.0, (0, 0))
        self.assertEqual(boxes, [])

    def test_cross_class_overlapping_boxes_are_one_physical_card(self):
        rows = np.concatenate((output_row("9C"), output_row("9D")), axis=0)
        boxes, values = decode_detections(rows, (640, 640), 1.0, (0, 0))
        self.assertEqual(len(boxes), 1)
        self.assertEqual(len(values), 1)


class TableGateTests(unittest.TestCase):
    def valid_cards(self):
        return [
            card("9♣", "hero候选", 0.46),
            card("3♣", "hero候选", 0.52),
            card("3♥", "公共牌候选", 0.41),
            card("2♦", "公共牌候选", 0.50),
            card("7♣", "公共牌候选", 0.59),
        ]

    def test_requires_three_identical_complete_frames_and_rejects_stale_signature(self):
        gate = TableStateGate(required_frames=3)
        cards = self.valid_cards()
        first = gate.update([dict(item) for item in cards])
        second = gate.update([dict(item) for item in cards])
        third = gate.update([dict(item) for item in cards])
        self.assertFalse(first["通过"])
        self.assertFalse(second["通过"])
        self.assertTrue(third["通过"])
        self.assertTrue(validate_analysis_gate(cards, third)["valid"])
        changed = self.valid_cards()
        changed[-1]["candidate"] = "8♣"
        self.assertFalse(validate_analysis_gate(changed, third)["valid"])
        self.assertFalse(gate.update(changed)["通过"])

    def test_resets_on_duplicate_unknown_low_margin_or_invalid_street(self):
        gate = TableStateGate(required_frames=3)
        cards = self.valid_cards()
        cards[1]["candidate"] = "9♣"
        self.assertEqual(gate.update(cards)["状态"], "牌面冲突")
        cards = self.valid_cards()
        cards[-1]["source"] = "OCR"
        self.assertEqual(gate.update(cards)["状态"], "牌面视觉模型缺失")
        cards = self.valid_cards()
        cards[0]["margin"] = 0.02
        self.assertEqual(gate.update(cards)["状态"], "识别置信度不足")
        cards = self.valid_cards()[:4]
        self.assertEqual(gate.update(cards)["状态"], "阶段异常")


if __name__ == "__main__":
    unittest.main()
