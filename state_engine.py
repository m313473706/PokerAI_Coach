from __future__ import annotations
from collections import defaultdict
from typing import Any

STREET_BY_COUNT = {0: "翻牌前", 3: "翻牌", 4: "转牌", 5: "河牌"}


def _amount_key(field: str, item: dict, frame_width: int, frame_height: int) -> str:
    if field == "pot":
        return "pot"
    bbox = item.get("bbox", [0, 0, 0, 0])
    x, y, w, h = [int(v) for v in bbox]
    # Coarse spatial key is used only to associate nearby boxes between frames; it
    # is not a confirmed seat identity/position.
    gx = int(((x + w/2) / max(frame_width, 1)) * 12)
    gy = int(((y + h/2) / max(frame_height, 1)) * 12)
    return f"stack:{item.get('seat_region', item.get('region','unknown'))}:{gx}:{gy}"


class ContinuousStateTracker:
    """Conservative multi-frame tracker for street, amount changes, visible labels.

    Events emitted by this class are candidates, not proof of a specific poker action.
    A pot/stack delta can come from a bet, payout, blind, animation or OCR error.
    """
    def __init__(self, street_repeats: int = 2, new_hand_zero_repeats: int = 4, amount_repeats: int = 2):
        self.street_repeats = max(2, int(street_repeats))
        self.new_hand_zero_repeats = max(3, int(new_hand_zero_repeats))
        self.amount_repeats = max(2, int(amount_repeats))
        self.pending_board_count = None
        self.pending_board_repeats = 0
        self.stable_board_count = None
        self.stable_street = "待确认"
        self.amount_pending: dict[str, tuple[float, int, dict]] = {}
        self.amount_stable: dict[str, tuple[float, dict]] = {}
        self.action_last_event_ts: dict[tuple[str, str], float] = {}
        self.events: list[dict] = []
        self.hand_candidate_number = 0
        self._last_new_hand_event_ts = -99999.0

    def _event(self, timestamp: float, event_type: str, status: str, **fields) -> dict:
        event = {"timestamp_s": round(float(timestamp), 3), "event_type": event_type,
                 "status": status, **fields}
        self.events.append(event)
        return event

    def _observe_board_count(self, timestamp: float, count: int | None) -> list[dict]:
        emitted = []
        if count not in STREET_BY_COUNT:
            # Counts 1 or 2 usually appear during animation/partial recognition; don't
            # treat them as a street or a new hand.
            self.pending_board_count = None
            self.pending_board_repeats = 0
            return emitted
        if count == self.pending_board_count:
            self.pending_board_repeats += 1
        else:
            self.pending_board_count = count
            self.pending_board_repeats = 1
        required = self.new_hand_zero_repeats if count == 0 and (self.stable_board_count or 0) >= 3 else self.street_repeats
        if self.pending_board_repeats < required:
            return emitted
        if self.stable_board_count is None:
            self.stable_board_count = count
            self.stable_street = STREET_BY_COUNT[count]
            return emitted
        old_count = self.stable_board_count
        if count == old_count:
            return emitted
        old_street = STREET_BY_COUNT[old_count]
        new_street = STREET_BY_COUNT[count]
        if old_count >= 3 and count == 0:
            self.hand_candidate_number += 1
            event = self._event(timestamp, "new_hand_candidate", "待确认",
                                candidate_hand_number=self.hand_candidate_number,
                                previous_street=old_street, new_street="翻牌前",
                                note="公共牌连续多帧清空；需结合底池重置/新局动画人工确认")
            emitted.append(event)
        elif count > old_count and count in (3, 4, 5):
            event = self._event(timestamp, "street_change", "多帧稳定候选",
                                previous_street=old_street, current_street=new_street,
                                previous_board_count=old_count, current_board_count=count,
                                note="根据公共牌数量变化产生候选；动画或漏牌仍需复核")
            emitted.append(event)
        else:
            # A non-new-hand decrease in board count is suspicious; don't call it a
            # street change. It can be temporary occlusion or detection failure.
            event = self._event(timestamp, "board_count_reset_candidate", "待确认",
                                previous_street=old_street, observed_street=new_street,
                                previous_board_count=old_count, observed_board_count=count,
                                note="公共牌数量下降但不满足新一手条件")
            emitted.append(event)
        self.stable_board_count = count
        self.stable_street = new_street
        return emitted

    def _observe_amounts(self, timestamp: float, vision_result: dict, amounts: dict | None) -> list[dict]:
        emitted = []
        if not amounts:
            return emitted
        width = int(vision_result.get("frame_width", 1920))
        height = int(vision_result.get("frame_height", 1080))
        current: dict[str, dict] = {}
        pot = amounts.get("pot")
        if pot and pot.get("status", "").startswith("单帧") and pot.get("numeric_value") is not None:
            current["pot"] = pot
        for item in amounts.get("stacks", []):
            if not str(item.get("status", "")).startswith("单帧") or item.get("numeric_value") is None:
                continue
            key = _amount_key("stack", item, width, height)
            current[key] = item
        for key, item in current.items():
            val = float(item["numeric_value"])
            pending = self.amount_pending.get(key)
            if pending and abs(pending[0] - val) < 1e-6:
                repeats = pending[1] + 1
            else:
                repeats = 1
            self.amount_pending[key] = (val, repeats, item)
            if repeats < self.amount_repeats:
                continue
            previous = self.amount_stable.get(key)
            if previous and abs(previous[0] - val) > 1e-6:
                old = previous[0]
                delta = round(val - old, 4)
                region = item.get("region", item.get("seat_region", key))
                field = item.get("field", "stack")
                emitted.append(self._event(
                    timestamp, "amount_change_candidate", "待确认",
                    field=field, region=region, previous_value=old, current_value=val,
                    delta=delta, displayed_candidate=item.get("candidate", "待确认"),
                    note="金额变化可由下注、结算、盲注或动画造成；未自动映射到跟注/加注/弃牌",
                ))
            self.amount_stable[key] = (val, item)
        return emitted

    def _observe_actions(self, timestamp: float, action_result: dict | None) -> list[dict]:
        emitted = []
        if not action_result:
            return emitted
        for item in action_result.get("actions", []):
            if float(item.get("confidence", 0.0)) < 0.80:
                continue
            action = str(item.get("action", "待确认"))
            region = str(item.get("region", "未知方位"))
            key = (action, region)
            last_seen = self.action_last_event_ts.get(key, -99999.0)
            if float(timestamp) - last_seen >= 2.0:
                emitted.append(self._event(timestamp, "visible_action_label_candidate", "待确认",
                                           action_candidate=action, region=region,
                                           raw_text=item.get("raw_text", ""),
                                           confidence=float(item.get("confidence", 0.0)),
                                           note="屏幕可见文字候选；玩家行动归属/轮次合法性尚需结合座位与行动顺序核验"))
            # This timestamp is the last sighting, not only the last event. Persistent
            # labels do not create a new event every few frames.
            self.action_last_event_ts[key] = float(timestamp)
        return emitted

    def update(self, timestamp: float, vision_result: dict, amounts: dict | None = None,
               action_result: dict | None = None) -> dict[str, Any]:
        events_before = len(self.events)
        self._observe_board_count(timestamp, vision_result.get("board_card_count_candidate"))
        self._observe_amounts(timestamp, vision_result, amounts)
        self._observe_actions(timestamp, action_result)
        new_events = self.events[events_before:]
        return {
            "timestamp_s": round(float(timestamp), 3),
            "hand_number_candidate": self.hand_candidate_number,
            "street_candidate": self.stable_street,
            "board_count_candidate": vision_result.get("board_card_count_candidate"),
            "new_events": new_events,
            "event_count_total": len(self.events),
            "status": "待确认" if new_events else "跟踪中",
        }

    def export_events(self) -> list[dict]:
        return list(self.events)
