import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import ocr_runtime_smoke as smoke


class _FakeRapidOCR:
    def __init__(self, result):
        self.result = result

    def __call__(self, image):
        return self.result, 0.01


class OCRRuntimeSmokeTests(unittest.TestCase):
    def run_fake(self, result, strict=True):
        fake_module = types.ModuleType("rapidocr_onnxruntime")
        fake_module.RapidOCR = lambda: _FakeRapidOCR(result)
        image = np.zeros((60, 360, 3), dtype=np.uint8)
        with patch.dict(sys.modules, {"rapidocr_onnxruntime": fake_module}), \
             patch.object(smoke.cv2, "imread", return_value=image):
            return smoke.run_smoke(Path("fake_sample.png"), require_action_text=strict)

    def test_runtime_and_expected_action_text_pass(self):
        result = [([[0, 0], [350, 0], [350, 40], [0, 40]], "全押", 0.98)]
        report, code = self.run_fake(result)
        self.assertEqual(code, 0)
        self.assertTrue(report["rapidocr_imported"])
        self.assertTrue(report["engine_initialized"])
        self.assertTrue(report["inference_completed"])
        self.assertTrue(report["expected_action_detected"])

    def test_strict_mode_fails_when_label_not_read(self):
        report, code = self.run_fake([])
        self.assertEqual(code, 3)
        self.assertTrue(report["inference_completed"])
        self.assertFalse(report["expected_action_detected"])

    def test_non_strict_mode_reports_warning_but_returns_zero(self):
        report, code = self.run_fake([], strict=False)
        self.assertEqual(code, 0)
        self.assertTrue(report["inference_completed"])
        self.assertIsNotNone(report["warning"])

    def test_console_report_json_is_ascii_safe_for_windows_code_pages(self):
        # ensure_ascii=True must prevent UnicodeEncodeError on legacy Windows consoles.
        import json
        payload = json.dumps({"recognized_texts": [{"text": "全押"}], "note": "待确认"}, ensure_ascii=True)
        self.assertTrue(payload.isascii())
        self.assertIn(r"\u5168\u62bc", payload)


if __name__ == "__main__":
    unittest.main()
