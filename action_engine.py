from __future__ import annotations
import re
import shutil
from typing import Any

import cv2
import amount_engine as ae
import vision_engine as ve

# Exact visible text only. A chip/stack movement alone never turns into a poker action.
ACTION_PATTERNS = [
    ("fold", ("弃牌", "弃掉", "FOLD", "FOLDED")),
    ("check", ("过牌", "CHECK", "CHECKED")),
    ("call", ("跟注", "CALL", "CALLED")),
    ("raise", ("加注", "RAISE", "RAISED")),
    ("bet", ("下注", "BET", "BETS")),
    ("all_in", ("全押", "全下", "ALLIN", "ALL-IN", "ALL IN")),
]


def _norm_text(value: str) -> str:
    return re.sub(r"[\s:：,.，。·_\-—|丨!！?？()（）\[\]【】]", "", str(value)).upper()


def classify_visible_action(text: str) -> str | None:
    normalized = _norm_text(text)
    if not normalized:
        return None
    for action, patterns in ACTION_PATTERNS:
        for pattern in patterns:
            p = _norm_text(pattern)
            if p and p in normalized and len(normalized) <= len(p) + 4:
                return action
    return None


def _analyze_action_with_tesseract(frame) -> tuple[list[dict[str, Any]], str]:
    """Slow optional OCR fallback; requires external Tesseract and Chinese language data."""
    try:
        import pytesseract
        from pytesseract import Output
        rows = pytesseract.image_to_data(
            frame, lang="chi_sim+eng", config="--oem 1 --psm 11",
            output_type=Output.DICT,
        )
    except Exception as exc:
        binary = shutil.which("tesseract")
        detail = f"Tesseract 备用 OCR 不可用：{type(exc).__name__}: {exc}"
        if not binary:
            detail += "；未找到 tesseract 可执行文件"
        return [], detail
    height, width = frame.shape[:2]
    candidates = []
    texts = rows.get("text", [])
    for i, value in enumerate(texts):
        raw = str(value or "").strip()
        action = classify_visible_action(raw)
        if action is None:
            continue
        try:
            conf = float(rows.get("conf", [0] * len(texts))[i])
        except Exception:
            conf = 0.0
        x = int(rows.get("left", [0] * len(texts))[i]); y = int(rows.get("top", [0] * len(texts))[i])
        w = int(rows.get("width", [1] * len(texts))[i]); h = int(rows.get("height", [1] * len(texts))[i])
        bbox = [max(0, x), max(0, y), max(1, min(w, width-x)), max(1, min(h, height-y))]
        candidates.append({
            "action": action, "raw_text": raw, "bbox": bbox,
            "region": ae._seat_region(bbox, frame.shape),
            "confidence": round(max(0.0, min(1.0, conf / 100.0)), 3),
            "engine": "Tesseract 备用", "status": "可见动作标签候选（需核对）",
            "note": "Tesseract 备用输出；应在 Windows RapidOCR 打包版重新验收",
        })
    note = "Tesseract OCR 可运行但未读到动作标签" if not candidates else "Tesseract OCR 成功产生动作标签候选"
    return candidates, note


def analyze_visible_action_text(frame) -> dict[str, Any]:
    """Find explicit action-label text with RapidOCR; unknown/ambiguous labels stay pending.

    Recognition is intentionally conservative. This looks for visible on-screen labels,
    not latent poker actions, and it does not infer an action from stack/pot changes.
    """
    if frame is None or getattr(frame, "size", 0) == 0:
        return {"actions": [], "status": "待确认", "note": "没有可用画面"}
    engine = ve._get_rapid_engine()
    height, width = frame.shape[:2]
    candidates = []
    engine_name = "RapidOCR"
    result = None
    if engine is not None:
        try:
            result, _elapsed = engine(frame)
        except Exception:
            result = None
    if result is not None:
        for item in result or []:
            try:
                polygon, raw_text, score = item
                action = classify_visible_action(raw_text)
                if action is None:
                    continue
                pts = [tuple(map(float, p)) for p in polygon]
                x0 = max(0, int(min(p[0] for p in pts)))
                y0 = max(0, int(min(p[1] for p in pts)))
                x1 = min(width, int(max(p[0] for p in pts)))
                y1 = min(height, int(max(p[1] for p in pts)))
                bbox = (x0, y0, max(1, x1-x0), max(1, y1-y0))
                confidence = float(score)
                zone = ae._seat_region(bbox, frame.shape)
                candidates.append({
                    "action": action, "raw_text": str(raw_text), "bbox": list(bbox),
                    "region": zone, "confidence": round(confidence, 3), "engine": engine_name,
                    "status": "可见动作标签候选（需核对）" if confidence >= 0.80 else "待确认",
                    "note": "只识别到了画面可见文字；尚未证明该文字覆盖所有行动状态",
                })
            except Exception:
                continue
    # Optional fallback for a Python environment with Tesseract + chi_sim installed.
    # The Windows one-file build uses bundled RapidOCR as its primary runtime.
    if engine is None or result is None:
        fallback_candidates, fallback_note = _analyze_action_with_tesseract(frame)
        candidates.extend(fallback_candidates)
        engine_name = "Tesseract 备用" if fallback_candidates else "无可用 OCR 引擎/未读到标签"
        if not candidates:
            return {"actions": [], "status": "待确认", "note": fallback_note}
    # Remove duplicate OCR boxes for the same visible label.
    unique = []
    for item in sorted(candidates, key=lambda r: r["confidence"], reverse=True):
        x, y, w, h = item["bbox"]
        duplicate = any(
            old["action"] == item["action"] and old["region"] == item["region"] and
            abs((old["bbox"][0] + old["bbox"][2]/2) - (x+w/2)) < 50 and
            abs((old["bbox"][1] + old["bbox"][3]/2) - (y+h/2)) < 40
            for old in unique
        )
        if not duplicate:
            unique.append(item)
    return {
        "actions": unique,
        "status": "存在可见动作标签候选" if unique else "待确认",
        "engine": engine_name,
        "note": "未读到动作标签不等于玩家没有行动；行动类别需要与下注变化和轮次顺序联合核验",
    }


def draw_action_candidates(frame, action_result: dict) :
    if frame is None or getattr(frame, "size", 0) == 0:
        return frame
    view = frame.copy()
    for item in action_result.get("actions", []):
        x, y, w, h = item.get("bbox", [0, 0, 0, 0])
        color = (0, 170, 255) if item.get("confidence", 0) < 0.80 else (40, 210, 40)
        cv2.rectangle(view, (int(x), int(y)), (int(x+w), int(y+h)), color, 2)
        label = f"{item.get('action','?')} {item.get('confidence',0):.2f}"
        cv2.putText(view, label, (int(x), max(15, int(y)-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
    return view
