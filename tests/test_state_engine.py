import unittest
import state_engine as se


def vision(count, width=1920, height=1080):
    return {"board_card_count_candidate": count, "frame_width": width, "frame_height": height}


def amounts(pot, stack=None, stack_status="单帧候选（需跨帧确认）"):
    rows = []
    if stack is not None:
        rows.append({"field": "stack", "region": "右侧", "seat_region": "右侧",
                     "bbox": [1300, 500, 75, 30], "candidate": str(stack),
                     "numeric_value": float(stack), "status": stack_status})
    return {"pot": {"field": "pot", "region": "底池", "bbox": [590,365,340,75],
                     "candidate": str(pot), "numeric_value": float(pot),
                     "confidence": 0.9, "status": "单帧候选（需跨帧确认）"}, "stacks": rows}


class StateEngineTests(unittest.TestCase):
    def test_street_change_requires_two_stable_observations(self):
        tracker = se.ContinuousStateTracker()
        tracker.update(0.0, vision(0), amounts(0.1))
        tracker.update(0.5, vision(0), amounts(0.1))
        first = tracker.update(1.0, vision(3), amounts(0.2))
        self.assertEqual(first["new_events"], [])
        second = tracker.update(1.5, vision(3), amounts(0.2))
        self.assertTrue(any(e["event_type"] == "street_change" and e["current_street"] == "翻牌" for e in second["new_events"]))

    def test_new_hand_requires_four_clear_board_samples(self):
        tracker = se.ContinuousStateTracker(new_hand_zero_repeats=4)
        tracker.update(0, vision(5), amounts(1.0))
        tracker.update(0.5, vision(5), amounts(1.0))
        tracker.update(1, vision(0), amounts(0.2))
        tracker.update(1.5, vision(0), amounts(0.2))
        tracker.update(2, vision(0), amounts(0.2))
        fourth = tracker.update(2.5, vision(0), amounts(0.2))
        events = [e for e in fourth["new_events"] if e["event_type"] == "new_hand_candidate"]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["status"], "待确认")

    def test_amount_change_requires_two_stable_observations_and_is_deduped(self):
        tracker = se.ContinuousStateTracker(amount_repeats=2)
        tracker.update(0, vision(0), amounts(0.1, 10))
        tracker.update(0.5, vision(0), amounts(0.1, 10))
        tracker.update(1, vision(0), amounts(0.2, 8))
        fourth = tracker.update(1.5, vision(0), amounts(0.2, 8))
        events = [e for e in fourth["new_events"] if e["event_type"] == "amount_change_candidate"]
        self.assertGreaterEqual(len(events), 1)
        self.assertTrue(all(e["status"] == "待确认" for e in events))
        fifth = tracker.update(2, vision(0), amounts(0.2, 8))
        self.assertFalse(any(e["event_type"] == "amount_change_candidate" for e in fifth["new_events"]))

    def test_persistent_visible_action_label_is_deduped(self):
        tracker = se.ContinuousStateTracker()
        result = {"actions": [{"action": "all_in", "region": "下方", "raw_text": "全押",
                               "confidence": 0.95, "status": "可见动作标签候选（需核对）"}]}
        first = tracker.update(0, vision(0), amounts(0.1), result)
        second = tracker.update(0.5, vision(0), amounts(0.1), result)
        self.assertTrue(any(e["event_type"] == "visible_action_label_candidate" for e in first["new_events"]))
        self.assertFalse(any(e["event_type"] == "visible_action_label_candidate" for e in second["new_events"]))


if __name__ == "__main__":
    unittest.main()
