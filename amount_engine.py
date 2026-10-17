from __future__ import annotations

from pathlib import Path
from typing import Optional
import re

import cv2
import numpy as np

RANK_CURRENCY_TOKEN = re.compile(r"\$\s*[0-9][0-9,]*(?:\.[0-9]{1,2})?")
AMOUNT_TOKEN = re.compile(r"\$?\s*[0-9][0-9,]*(?:\.[0-9]{1,2})?")


def detect_stack_amount_boxes(frame: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Find blue stack-value text blocks without hard-coding every seat coordinate.

    This is a layout-adapted visual proposal, not a semantic seat detector. It may
    miss a differently coloured stack or pick up an unrelated blue text fragment.
    Such candidates are retained with low confidence/pending status.
    """
    if frame is None or frame.size == 0:
        return []
    height, width = frame.shape[:2]
    sx, sy = width / 1920.0, height / 1080.0
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Stack values are a muted cyan/blue; higher saturation filters blue-gray avatar details.
    mask = cv2.inRange(
        hsv,
        np.array([85, 65, 50], dtype=np.uint8),
        np.array([135, 255, 255], dtype=np.uint8),
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3)))
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask)
    boxes: list[tuple[int, int, int, int]] = []
    for x, y, w, h, area in stats[1:]:
        if area < int(180 * sx * sy):
            continue
        if not (int(18 * sx) <= w <= int(160 * sx)):
            continue
        # Reject short blue UI ornaments (for example dealer buttons) that often look like text.
        if not (int(20 * sy) <= h <= int(42 * sy)):
            continue
        if not (int(150 * sy) <= y <= int(1030 * sy)):
            continue
        aspect = w / max(h, 1)
        if not (0.55 <= aspect <= 5.5):
            continue
        cx, cy = x + w / 2, y + h / 2
        # Exclude blue chips/bet decorations around the pot. Player stack text is
        # normally nearer a seat panel; actual lower-center hero stacks are retained.
        center_x = width * 0.34
        if abs(cx - center_x) < 220 * sx and 400 * sy < cy < 750 * sy:
            continue
        boxes.append((int(x), int(y), int(w), int(h)))

    # Join horizontally split amounts such as "$2." + "16" into one OCR region.
    changed = True
    while changed:
        changed = False
        merged: list[tuple[int, int, int, int]] = []
        used = [False] * len(boxes)
        for i, (x, y, w, h) in enumerate(boxes):
            if used[i]:
                continue
            x0, y0, x1, y1 = x, y, x + w, y + h
            used[i] = True
            for j in range(i + 1, len(boxes)):
                if used[j]:
                    continue
                xx, yy, ww, hh = boxes[j]
                gap = max(0, max(x0, xx) - min(x1, xx + ww))
                vertical_overlap = max(0, min(y1, yy + hh) - max(y0, yy))
                center_delta = abs((y0 + y1) / 2 - (yy + yy + hh) / 2)
                if gap <= 18 * sx and center_delta <= 12 * sy and vertical_overlap >= min(y1 - y0, hh) * 0.35:
                    x0, y0, x1, y1 = min(x0, xx), min(y0, yy), max(x1, xx + ww), max(y1, yy + hh)
                    used[j] = True
                    changed = True
            merged.append((int(x0), int(y0), int(x1 - x0), int(y1 - y0)))
        boxes = merged
    boxes = sorted(set(boxes), key=lambda b: (b[1], b[0]))
    return boxes


def _extract_tokens(text: str) -> list[str]:
    if not text:
        return []
    cleaned = str(text).replace("O", "0").replace("o", "0").replace("S", "5")
    tokens = AMOUNT_TOKEN.findall(cleaned)
    # Prefer an explicit currency token when there are several OCR fragments.
    explicit = [t.strip().replace(" ", "") for t in tokens if "$" in t]
    tokens = explicit or [t.strip().replace(" ", "") for t in tokens]
    tokens = [t for t in tokens if re.search(r"\d", t)]
    if not tokens:
        return []
    tokens.sort(key=lambda t: ("$" in t, len(re.sub(r"\D", "", t))), reverse=True)
    return tokens[:3]


def _normalize_token(token: str, profile: str) -> tuple[str, Optional[float], bool, bool]:
    """Return canonical display, numeric value, inferred-format flag, and valid flag."""
    t = re.sub(r"\s+", "", token)
    has_dollar = "$" in t
    t = t.replace("$", "")
    t = t.replace("I", "1").replace("l", "1")
    # Explicit thousands separators / decimals are the least ambiguous case.
    if "." in t:
        try:
            amount = float(t.replace(",", ""))
        except ValueError:
            return token, None, False, False
        if profile == "tournament_brown" and not has_dollar:
            display = f"{int(round(amount)):,}"
        else:
            display = f"${amount:.2f}" if has_dollar or profile != "tournament_brown" else f"{amount:.2f}"
        inferred_currency = (profile != "tournament_brown" and not has_dollar)
        return display, amount, inferred_currency, True

    digits = re.sub(r"\D", "", t)
    if not digits:
        return token, None, False, False
    if "," in t:
        try:
            amount = float(digits)
        except ValueError:
            return token, None, False, False
        if has_dollar:
            display = f"${int(amount):,}"
        else:
            display = f"{int(amount):,}"
        return display, amount, False, True

    if profile == "tournament_brown":
        amount = float(int(digits))
        display = f"{int(amount):,}" if amount >= 1000 else str(int(amount))
        return display, amount, False, True

    # On the green tables, amounts below $10 are often shown with two cents
    # digits. OCR can lose the decimal point; reconstruct it only as a candidate
    # and force confirmation because "$245" can also be a genuine integer field.
    if has_dollar and len(digits) >= 3:
        amount = int(digits) / 100.0
        return f"${amount:.2f}", amount, True, True
    if has_dollar:
        amount = float(int(digits))
        return f"${int(amount)}", amount, False, True
    if len(digits) >= 3:
        amount = int(digits) / 100.0
        return f"${amount:.2f}", amount, True, True
    return f"${int(digits)}", float(int(digits)), True, True


def _run_tesseract(roi: np.ndarray, *, pot: bool = False) -> list[tuple[str, str]]:
    try:
        import pytesseract
    except Exception:
        return []
    if roi is None or roi.size == 0:
        return []
    out = []
    # A small multi-scale ensemble reduces common 2/7, 5/9 and dropped-decimal errors.
    # Disagreement is not silently resolved: the calling layer marks it 待确认.
    variants: list[tuple[str, np.ndarray, int]] = []
    for scale, psm in [(4, 7), (5, 7), (4, 8)]:
        up = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY) if up.ndim == 3 else up
        variants.append((f"gray-{scale}x-psm{psm}", gray, psm))
        if pot and scale == 4:
            binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            variants.append(("threshold-4x-psm7", binary, 7))
    for name, image, psm in variants:
        try:
            raw = pytesseract.image_to_string(
                image,
                config=f"--oem 1 --psm {psm} -c tessedit_char_whitelist=0123456789.,:$",
                timeout=2.0,
            ).strip()
            if raw:
                out.append((name, raw))
        except Exception:
            continue
    return out


def _run_rapidocr(roi: np.ndarray) -> list[tuple[str, str, float]]:
    try:
        # Reuse the process-level OCR model from the card engine if installed.
        from vision_engine import _get_rapid_engine
        engine = _get_rapid_engine()
        if engine is None:
            return []
        up = cv2.resize(roi, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        results, _elapsed = engine(up)
        out = []
        for item in results or []:
            try:
                _box, text, score = item
                if _extract_tokens(text):
                    out.append(("RapidOCR", text, float(score)))
            except Exception:
                pass
        return out
    except Exception:
        return []


def _make_amount_observation(roi: np.ndarray, bbox: tuple[int, int, int, int], profile: str,
                             *, field: str, region: str, pot: bool = False) -> dict:
    rapid = _run_rapidocr(roi)
    candidates: list[dict] = []
    for method, raw, score in rapid:
        for token in _extract_tokens(raw):
            display, numeric, inferred, valid = _normalize_token(token, profile)
            if valid:
                candidates.append({"method": method, "raw": raw, "token": token,
                                   "display": display, "numeric": numeric, "inferred": inferred,
                                   "engine_score": score})
    if not candidates:
        for method, raw in _run_tesseract(roi, pot=pot):
            tokens = _extract_tokens(raw)
            if not tokens:
                continue
            # Iterate possible tokens and keep the strongest amount-like candidate.
            for token in tokens:
                display, numeric, inferred, valid = _normalize_token(token, profile)
                if valid:
                    candidates.append({"method": method, "raw": raw, "token": token,
                                       "display": display, "numeric": numeric, "inferred": inferred,
                                       "engine_score": None})
                    break

    x, y, w, h = bbox
    base = {
        "field": field, "region": region, "bbox": [int(x), int(y), int(w), int(h)],
        "candidate": "待确认", "numeric_value": None, "confidence": 0.0,
        "status": "待确认", "raw_candidates": candidates, "note": "没有得到可用金额候选",
    }
    if not candidates:
        return base
    counts: dict[str, list[dict]] = {}
    for candidate in candidates:
        counts.setdefault(candidate["display"], []).append(candidate)
    def candidate_strength(rows: list[dict]) -> tuple[int, int, float]:
        explicit = sum(1 for row in rows if not row["inferred"])
        scores = [row["engine_score"] for row in rows if row["engine_score"] is not None]
        return (len(rows), explicit, max(scores, default=0.0))
    ordered = sorted(counts.items(), key=lambda kv: candidate_strength(kv[1]), reverse=True)
    winner, winner_rows = ordered[0]
    best_count = len(winner_rows)
    runner_count = len(ordered[1][1]) if len(ordered) > 1 else 0
    clear_majority = len(ordered) > 1 and best_count >= 2 and best_count > runner_count
    all_methods_agree = len(winner_rows) >= 2 and len(ordered) == 1
    inferred = all(row["inferred"] for row in winner_rows)
    engine_scores = [row["engine_score"] for row in winner_rows if row["engine_score"] is not None]
    if len(ordered) > 1 and not clear_majority:
        confidence = 0.48
        status = "待确认"
        note = "不同 OCR 预处理结果没有形成明确多数；保留最佳候选供人工复核"
    elif clear_majority:
        confidence = min(0.82, 0.70 + 0.05 * (best_count - runner_count))
        status = "单帧候选（OCR多数投票，仍需跨帧确认）"
        note = "多尺度 OCR 形成多数候选，但仍须跨帧或人工确认"
    elif inferred:
        confidence = 0.58 if all_methods_agree else 0.50
        status = "待确认"
        note = "小数点可能被 OCR 漏掉；已给出候选格式，但需要人工核对"
    elif engine_scores:
        confidence = max(0.0, min(1.0, float(np.mean(engine_scores))))
        status = "单帧候选（需跨帧确认）" if confidence >= 0.75 else "待确认"
        note = "RapidOCR 金额候选；内部评分不是校准概率"
    elif all_methods_agree:
        confidence = 0.90
        status = "单帧候选（需跨帧确认）"
        note = "两种图像预处理的 OCR 结果一致；仍需跨帧确认"
    else:
        confidence = 0.68
        status = "单帧候选（需跨帧确认）" if confidence >= 0.65 else "待确认"
        note = "单一 OCR 候选；仍需跨帧或人工确认"
    numeric_values = [c["numeric"] for c in winner_rows if c["numeric"] is not None]
    base.update({
        "candidate": winner,
        "numeric_value": numeric_values[0] if numeric_values else None,
        "confidence": round(float(confidence), 3), "status": status, "note": note,
    })
    return base


def _seat_region(bbox: tuple[int, int, int, int], frame_shape: tuple[int, ...]) -> str:
    x, y, w, h = bbox
    height, width = frame_shape[:2]
    sx, sy = width / 1920.0, height / 1080.0
    cx, cy = x + w / 2, y + h / 2
    if cy < 300 * sy:
        if cx < 420 * sx:
            return "上左"
        if cx < 700 * sx:
            return "上中左"
        if cx < 1030 * sx:
            return "上中/上中右"
        return "上右"
    if cy < 420 * sy and cx < 350 * sx:
        return "上左"
    if cy < 420 * sy and cx > 1150 * sx:
        return "上右"
    if cy < 650 * sy:
        return "左侧" if cx < width * 0.32 else "右侧" if cx > width * 0.62 else "中部非座位候选"
    if cy < 900 * sy:
        return "下左" if cx < width * 0.32 else "下右" if cx > width * 0.62 else "下方"
    return "下方/本位候选"


def analyze_amounts(frame: np.ndarray, profile: str = "green_table") -> dict:
    """Detect table pot and blue stack-value text blocks; betting events remain separate."""
    if frame is None or frame.size == 0:
        return {"profile": profile, "pot": None, "stacks": [], "status": "待确认",
                "note": "没有可用画面"}
    height, width = frame.shape[:2]
    sx, sy = width / 1920.0, height / 1080.0
    if profile == "tournament_brown":
        px, py, pw, ph = 560, 350, 340, 75
    else:
        px, py, pw, ph = 590, 365, 340, 75
    pot_box = (int(px * sx), int(py * sy), int(pw * sx), int(ph * sy))
    x, y, w, h = pot_box
    pot_roi = frame[max(0, y):min(height, y + h), max(0, x):min(width, x + w)]
    pot = _make_amount_observation(pot_roi, pot_box, profile, field="pot", region="底池", pot=True)

    stacks = []
    for index, bbox in enumerate(detect_stack_amount_boxes(frame), 1):
        x, y, w, h = bbox
        padding_x, padding_y = int(9 * sx), int(7 * sy)
        xa, ya = max(0, x - padding_x), max(0, y - padding_y)
        xb, yb = min(width, x + w + padding_x), min(height, y + h + padding_y)
        roi = frame[ya:yb, xa:xb]
        seat_region = _seat_region(bbox, frame.shape)
        obs = _make_amount_observation(roi, bbox, profile, field="stack", region=seat_region)
        obs["bbox"] = list(bbox)
        obs["seat_region"] = seat_region
        obs["index"] = index
        if obs["candidate"] == "待确认":
            # Retain uncertain detected regions so the GUI can show them for review.
            obs["note"] = "蓝色筹码文字区域已定位，但金额未可靠读取；" + obs["note"]
        stacks.append(obs)

    confirmedish = int(pot.get("status", "").startswith("单帧")) + sum(
        int(stack.get("status", "").startswith("单帧")) for stack in stacks
    )
    detected = 1 + len(stacks)
    return {
        "profile": profile,
        "pot": pot,
        "stacks": stacks,
        "detected_stack_regions": len(stacks),
        "single_frame_candidates": confirmedish,
        "status": "待确认",
        "note": "底池使用主题相关固定 ROI；玩家筹码先按蓝色文字区域定位。金额候选需跨帧确认。下注额/加注额尚未与行动事件绑定。",
    }


def draw_amount_observations(frame: np.ndarray, amounts: dict) -> np.ndarray:
    """Draw amount ROIs/candidates for human review; never implies confirmation."""
    if frame is None or frame.size == 0:
        return frame
    view = frame.copy()
    pot = amounts.get("pot") if isinstance(amounts, dict) else None
    observations = ([pot] if pot else []) + (amounts.get("stacks", []) if isinstance(amounts, dict) else [])
    for obs in observations:
        bbox = obs.get("bbox", [0, 0, 0, 0])
        if len(bbox) != 4:
            continue
        x, y, w, h = map(int, bbox)
        confident = str(obs.get("status", "")).startswith("单帧")
        color = (30, 180, 210) if confident else (0, 150, 255)
        cv2.rectangle(view, (x, y), (x + w, y + h), color, 2)
        label = f'{obs.get("region", obs.get("field", "金额"))}: {obs.get("candidate", "待确认")} {float(obs.get("confidence", 0.0)):.2f}'
        text_y = max(18, y - 5)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        left = min(max(0, x), max(0, view.shape[1] - tw - 8))
        cv2.rectangle(view, (left, max(0, text_y - th - 5)), (left + tw + 6, text_y + 3), color, -1)
        cv2.putText(view, label, (left + 3, text_y - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
    return view


def amount_regions_changed(previous: np.ndarray | None, current: np.ndarray | None,
                           profile: str = "green_table", *, threshold: float = 6.0) -> bool:
    """Fast pixel-change gate before invoking slower amount OCR on a new video frame.

    It watches the fixed pot ROI and detected player stack-text patches. It is a
    scheduling heuristic only; no change does not prove the whole table is unchanged.
    """
    if previous is None or current is None or previous.size == 0 or current.size == 0:
        return True
    if previous.shape != current.shape:
        return True
    height, width = current.shape[:2]
    sx, sy = width / 1920.0, height / 1080.0
    if profile == "tournament_brown":
        px, py, pw, ph = 560, 350, 340, 75
    else:
        px, py, pw, ph = 590, 365, 340, 75
    x, y, w, h = int(px*sx), int(py*sy), int(pw*sx), int(ph*sy)
    prev_pot = previous[max(0,y):min(height,y+h), max(0,x):min(width,x+w)]
    cur_pot = current[max(0,y):min(height,y+h), max(0,x):min(width,x+w)]
    if prev_pot.size == 0 or cur_pot.size == 0:
        return True
    def normalized_patch(image, bbox):
        bx, by, bw, bh = bbox
        pad_x, pad_y = int(9*sx), int(7*sy)
        crop = image[max(0,by-pad_y):min(height,by+bh+pad_y),
                     max(0,bx-pad_x):min(width,bx+bw+pad_x)]
        if crop.size == 0:
            return None
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray, (128, 48), interpolation=cv2.INTER_AREA)
    ppot = cv2.resize(cv2.cvtColor(prev_pot, cv2.COLOR_BGR2GRAY), (160, 40), interpolation=cv2.INTER_AREA)
    cpot = cv2.resize(cv2.cvtColor(cur_pot, cv2.COLOR_BGR2GRAY), (160, 40), interpolation=cv2.INTER_AREA)
    if float(cv2.absdiff(ppot, cpot).mean()) >= threshold:
        return True
    prev_boxes = detect_stack_amount_boxes(previous)
    cur_boxes = detect_stack_amount_boxes(current)
    if len(prev_boxes) != len(cur_boxes):
        return True
    unused = set(range(len(prev_boxes)))
    for box in cur_boxes:
        bx, by, bw, bh = box
        cx, cy = bx + bw/2, by + bh/2
        matches = []
        for i in unused:
            px0, py0, pw0, ph0 = prev_boxes[i]
            pcx, pcy = px0 + pw0/2, py0 + ph0/2
            dist = ((cx-pcx)**2 + (cy-pcy)**2)**0.5
            matches.append((dist, i, prev_boxes[i]))
        if not matches:
            return True
        dist, idx, old_box = min(matches, key=lambda row: row[0])
        if dist > 25*max(sx, sy):
            return True
        unused.remove(idx)
        old_patch = normalized_patch(previous, old_box)
        new_patch = normalized_patch(current, box)
        if old_patch is None or new_patch is None:
            return True
        if float(cv2.absdiff(old_patch, new_patch).mean()) >= threshold:
            return True
    return False
