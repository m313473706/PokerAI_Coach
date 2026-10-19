"""Synthetic contract tests; these exercise code paths, not real-card accuracy."""
import json
import unittest

import numpy as np
from PIL import Image

from card_vision import (
    CARD_LABELS,
    MIN_CLASS_MARGIN,
    MODEL_SCHEMA,
    TableStateGate,
    CardVisionError,
    classify_card_regions,
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


def probability_row(label, top=0.95, second=0.01):
    row = np.zeros((1, len(CARD_LABELS)), dtype=np.float32)
    index = CARD_LABELS.index(label)
    row[0, index] = top
    row[0, (index + 1) % len(CARD_LABELS)] = second
    row[0, (index + 2) % len(CARD_LABELS)] = 1.0 - top - second
    return row


class FakeSession:
    class Item:
        def __init__(self, name):
            self.name = name
            self.shape = [1, 3, 192, 128]

    def __init__(self, probabilities):
        self.probabilities = probabilities

    def get_inputs(self):
        return [self.Item("card_crop")]

    def get_outputs(self):
        return [self.Item("class_probabilities")]

    def run(self, output_names, feeds):
        self.assertion = (output_names, feeds)
        return [self.probabilities]


class ModelContractTests(unittest.TestCase):
    def metadata(self):
        return {
            "pokerai_schema": MODEL_SCHEMA,
            "class_labels": json.dumps(CARD_LABELS),
            "license": "CC0-1.0",
            "source_url": "https://example.org/model",
            "license_url": "https://example.org/license",
        }

    def test_accepts_fixed_52_class_crop_classifier_contract(self):
        self.assertTrue(validate_model_contract(
            self.metadata(), [1, 3, 192, 128], [1, 52]
        ))

    def test_rejects_unverified_license_and_mismatched_labels(self):
        metadata = self.metadata()
        metadata["license"] = "AGPL-3.0"
        with self.assertRaises(CardVisionError):
            validate_model_contract(metadata, [1, 3, 192, 128], [1, 52])
        metadata = self.metadata()
        metadata["class_labels"] = json.dumps(list(reversed(CARD_LABELS)))
        with self.assertRaises(CardVisionError):
            validate_model_contract(metadata, [1, 3, 192, 128], [1, 52])


class CropClassifierTests(unittest.TestCase):
    def test_classifies_card_crop_and_returns_confidence_margin(self):
        session = FakeSession(probability_row("9C"))
        image = Image.new("RGB", (100, 100), "white")
        values = classify_card_regions(session, image, [[10, 10, 90, 90]])
        self.assertEqual(len(values), 1)
        self.assertEqual(values[0][0], "9♣")
        self.assertAlmostEqual(values[0][1], 0.95, places=5)
        self.assertGreaterEqual(values[0][2], MIN_CLASS_MARGIN)

    def test_rejects_ambiguous_crop_as_unknown(self):
        session = FakeSession(probability_row("9C", top=0.50, second=0.45))
        image = Image.new("RGB", (100, 100), "white")
        values = classify_card_regions(session, image, [[10, 10, 90, 90]])
        self.assertEqual(values[0][0], "??")


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
