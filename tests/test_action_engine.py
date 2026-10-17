import unittest
import action_engine as ax

class ActionEngineTests(unittest.TestCase):
    def test_chinese_visible_labels(self):
        cases = {"全 押": "all_in", "跟注": "call", "弃牌": "fold", "加注": "raise", "过牌": "check", "下注": "bet"}
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(ax.classify_visible_action(raw), expected)

    def test_english_visible_labels(self):
        cases = {"ALL-IN": "all_in", "Folded": "fold", "Call": "call", "Raise": "raise", "Check": "check", "Bet": "bet"}
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(ax.classify_visible_action(raw), expected)

    def test_random_text_is_not_a_poker_action(self):
        self.assertIsNone(ax.classify_visible_action("xiaomi778 $2.27"))
        self.assertIsNone(ax.classify_visible_action("每 日 $50,000"))

if __name__ == "__main__":
    unittest.main()
