from __future__ import annotations

from dataclasses import dataclass, asdict
from collections import Counter
from pathlib import Path
from typing import Optional, Iterable
import json
import re
import shutil
import subprocess
import tempfile

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
TEMPLATE_DIR = ROOT / "templates" / "suits"
RANKS = {"2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"}
SUITS = {"h", "d", "c", "s"}
SUIT_SYMBOL = {"h": "♥", "d": "♦", "c": "♣", "s": "♠"}


@dataclass
class CardObservation:
    index: int
    bbox: tuple[int, int, int, int]
    rank: str = "待确认"
    suit: str = "待确认"
    suit_symbol: str = "?"
    rank_confidence: float = 0.0
    suit_confidence: float = 0.0
    confidence: float = 0.0
    white_ratio: float = 0.0
    status: str = "待确认"
    note: str = ""

    def to_dict(self):
        d = asdict(self)
        d["bbox"] = list(self.bbox)
        return d


def _fit_mask(mask: np.ndarray, size=(64, 64)) -> np.ndarray:
    """Normalize a binary icon mask while retaining its layout in the cropped ROI."""
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = (mask > 0).astype(np.uint8) * 255
    return cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)


def _suit_mask_from_roi(roi: np.ndarray, color_group: str) -> np.ndarray:
    """Isolate the small upper-left suit glyph and normalize its shape."""
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    if color_group == "red":
        pix = (((hsv[:, :, 0] <= 12) | (hsv[:, :, 0] >= 165)) &
               (hsv[:, :, 1] >= 70) & (hsv[:, :, 2] >= 65))
    else:
        pix = (hsv[:, :, 2] <= 100) & (hsv[:, :, 1] <= 130)
    mask = pix.astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    # The crop also contains part of the larger center suit. Keep the connected glyph
    # closest to the top-left suit position, dropping distant components.
    n, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    h, w = mask.shape[:2]
    components = []
    for i in range(1, n):
        x, y, cw, ch, area = stats[i]
        cx, cy = centroids[i]
        if area < 8 or cx > w * 0.65 or cy < h * 0.12 or cy > h * 0.95:
            continue
        components.append((int(area), int(x), int(y), int(cw), int(ch)))
    if not components:
        return np.zeros((64, 64), dtype=np.uint8)
    area, x, y, cw, ch = max(components, key=lambda item: item[0])
    shape = mask[y:y + ch, x:x + cw]
    size = 46
    scale = min(size / max(cw, 1), size / max(ch, 1))
    new_w, new_h = max(1, int(cw * scale)), max(1, int(ch * scale))
    shape = cv2.resize(shape, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    normalized = np.zeros((64, 64), dtype=np.uint8)
    sx, sy = (64 - new_w) // 2, (64 - new_h) // 2
    normalized[sy:sy + new_h, sx:sx + new_w] = shape
    return normalized


def _load_suit_templates() -> dict[str, list[np.ndarray]]:
    templates: dict[str, list[np.ndarray]] = {"h": [], "d": [], "c": [], "s": []}
    for meta_path in [TEMPLATE_DIR / "index.json"]:
        if not meta_path.exists():
            return templates
        try:
            rows = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return templates
        for row in rows:
            suit = row.get("suit")
            filename = row.get("file")
            if suit not in templates or not filename:
                continue
            im = cv2.imread(str(TEMPLATE_DIR / filename), cv2.IMREAD_GRAYSCALE)
            if im is not None:
                templates[suit].append(_fit_mask(im))
    return templates


_SUIT_TEMPLATES = _load_suit_templates()
_RAPID_ENGINE = None
_RAPID_INIT_TRIED = False
_RAPID_INIT_ERROR = None


def _get_rapid_engine():
    global _RAPID_ENGINE, _RAPID_INIT_TRIED, _RAPID_INIT_ERROR
    if _RAPID_INIT_TRIED:
        return _RAPID_ENGINE
    _RAPID_INIT_TRIED = True
    try:
        from rapidocr_onnxruntime import RapidOCR
        _RAPID_ENGINE = RapidOCR()
        _RAPID_INIT_ERROR = None
    except Exception as exc:
        _RAPID_ENGINE = None
        _RAPID_INIT_ERROR = f"{type(exc).__name__}: {exc}"
    return _RAPID_ENGINE


def get_ocr_runtime_status(initialize: bool = True) -> dict:
    """Return a user-facing OCR runtime report instead of silently hiding startup failures."""
    import importlib.metadata
    import shutil
    status = {
        "rapidocr_package_installed": False,
        "rapidocr_version": None,
        "rapidocr_engine_initialized": False,
        "rapidocr_error": None,
        "tesseract_executable": shutil.which("tesseract"),
        "pytesseract_package_installed": False,
        "tesseract_languages": [],
        "recommended_action": "",
    }
    try:
        status["rapidocr_version"] = importlib.metadata.version("rapidocr_onnxruntime")
        status["rapidocr_package_installed"] = True
    except Exception as exc:
        status["rapidocr_error"] = f"RapidOCR 包不可用：{type(exc).__name__}: {exc}"
    try:
        import pytesseract
        status["pytesseract_package_installed"] = True
        try:
            status["tesseract_languages"] = sorted(pytesseract.get_languages(config=""))
        except Exception as exc:
            status["tesseract_languages_error"] = f"{type(exc).__name__}: {exc}"
    except Exception:
        pass
    if initialize and status["rapidocr_package_installed"]:
        engine = _get_rapid_engine()
        status["rapidocr_engine_initialized"] = engine is not None
        if engine is None:
            status["rapidocr_error"] = _RAPID_INIT_ERROR or "RapidOCR 初始化失败，原因未知"
    if status["rapidocr_engine_initialized"]:
        status["recommended_action"] = "RapidOCR 已初始化；接着运行 --ocr-smoke，验证模型能实际处理测试画面。"
    elif status["tesseract_executable"] and status["pytesseract_package_installed"]:
        status["recommended_action"] = "RapidOCR 不可用；当前系统存在 Tesseract 备用通道。Windows 单文件版的主 OCR 仍应由打包的 RapidOCR 提供。"
    else:
        status["recommended_action"] = "OCR 运行依赖不可用。Windows 构建需安装 requirements.txt、收集 RapidOCR/ONNX Runtime 文件，并通过 --ocr-smoke 自检。"
    return status


def _normalize_rank(raw: str) -> str:
    value = re.sub(r"[^0-9TJQKA]", "", str(raw).upper())
    if value in {"10", "T", "IO", "I0", "1O"}:
        return "10"
    # OCR commonly produces a leading/trailing stray character around a single rank.
    if len(value) > 1:
        candidates = [v for v in (value, value[-1], value[0]) if v in RANKS or v == "T"]
        if candidates:
            value = candidates[0]
    if value == "T":
        return "10"
    return value if value in RANKS else "待确认"


def _recognize_rank_rapid(roi: np.ndarray) -> tuple[str, float, str]:
    engine = _get_rapid_engine()
    if engine is None:
        return "待确认", 0.0, "RapidOCR 未安装"
    # Upscaling matters for the small rank glyph at the top-left of each card.
    enlarged = cv2.resize(roi, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    try:
        result, _elapsed = engine(enlarged)
    except Exception as exc:
        return "待确认", 0.0, f"RapidOCR 异常: {exc}"
    candidates = []
    if result:
        for row in result:
            try:
                _box, text, score = row
                rank = _normalize_rank(text)
                if rank != "待确认":
                    candidates.append((rank, float(score)))
            except Exception:
                continue
    if not candidates:
        return "待确认", 0.0, "RapidOCR 无有效点数候选"
    candidates.sort(key=lambda x: x[1], reverse=True)
    rank, score = candidates[0]
    return rank, max(0.0, min(1.0, score)), "RapidOCR"


def _recognize_rank_tesseract(roi: np.ndarray) -> tuple[str, float, str]:
    """Fallback for development/testing environments with Tesseract installed."""
    try:
        import pytesseract
        # Calling image_to_string raises if the native executable is missing.
        enlarged = cv2.resize(roi, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY) if enlarged.ndim == 3 else enlarged
        threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        votes: list[str] = []
        for image, psm in ((gray, 10), (threshold, 10), (gray, 8), (threshold, 13)):
            raw = pytesseract.image_to_string(
                image,
                config=f"--oem 1 --psm {psm} -c tessedit_char_whitelist=0123456789TJQKA",
            )
            rank = _normalize_rank(raw)
            if rank != "待确认":
                votes.append(rank)
        if not votes:
            return "待确认", 0.0, "Tesseract 无有效点数候选"
        counts = Counter(votes)
        rank, n = counts.most_common(1)[0]
        confidence = n / 4.0
        # Rank 10 is commonly read as T and both representations normalize to 10.
        return rank, confidence, "Tesseract(备用)"
    except Exception as exc:
        return "待确认", 0.0, f"无可用 OCR 引擎: {exc}"


def recognize_rank_roi(roi: np.ndarray) -> tuple[str, float, str]:
    if roi is None or roi.size == 0:
        return "待确认", 0.0, "点数区域为空"
    # Use the bundled RapidOCR in the Windows build first, and Tesseract only as fallback.
    rank, score, engine = _recognize_rank_rapid(roi)
    if rank != "待确认" and score >= 0.50:
        return rank, score, engine
    alt_rank, alt_score, alt_engine = _recognize_rank_tesseract(roi)
    if alt_rank != "待确认" and alt_score >= 0.50:
        if rank != "待确认" and rank != alt_rank:
            # Do not silently choose between conflicting engines.
            return "待确认", max(score, alt_score) * 0.6, f"引擎结果冲突: {engine}={rank}; {alt_engine}={alt_rank}"
        return alt_rank, alt_score, alt_engine
    if rank != "待确认":
        return rank, score, engine
    return "待确认", max(score, alt_score), f"{engine}; {alt_engine}"


def recognize_rank(card_bgr: np.ndarray, bbox: tuple[int, int, int, int]) -> tuple[str, float, str]:
    x, y, w, h = bbox
    # Small top-left rank box. Avoid the large suit symbol lower in the card.
    xa = max(0, int(x + 0.02 * w))
    xb = min(card_bgr.shape[1], int(x + 0.60 * w))
    ya = max(0, int(y + 0.02 * h))
    yb = min(card_bgr.shape[0], int(y + 0.39 * h))
    return recognize_rank_roi(card_bgr[ya:yb, xa:xb])


def _white_ratio(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> float:
    x, y, w, h = bbox
    patch = frame[max(y, 0):min(frame.shape[0], y + h), max(x, 0):min(frame.shape[1], x + w)]
    if patch.size == 0:
        return 0.0
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    return float(((hsv[:, :, 2] > 180) & (hsv[:, :, 1] < 110)).mean())


def _suit_crop(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = bbox
    # The top-left suit glyph sits below the rank; keep out the large central suit.
    xa, xb = int(x + 0.01 * w), int(x + 0.65 * w)
    ya, yb = int(y + 0.32 * h), int(y + 0.66 * h)
    return frame[max(0, ya):min(frame.shape[0], yb), max(0, xa):min(frame.shape[1], xb)]


def recognize_suit(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> tuple[str, float, float, str]:
    roi = _suit_crop(frame, bbox)
    if roi.size == 0:
        return "待确认", 0.0, 0.0, "花色区域为空"
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    red_mask = (((hsv[:, :, 0] <= 12) | (hsv[:, :, 0] >= 165)) &
                (hsv[:, :, 1] >= 70) & (hsv[:, :, 2] >= 65))
    dark_mask = (hsv[:, :, 2] <= 100) & (hsv[:, :, 1] <= 130)
    red_amount = int(red_mask.sum())
    dark_amount = int(dark_mask.sum())
    if red_amount < 25 and dark_amount < 25:
        return "待确认", 0.0, 0.0, "未提取到稳定花色图形"
    color_group = "red" if red_amount > dark_amount * 1.15 else "black"
    query = _suit_mask_from_roi(roi, color_group)
    allowed = ("h", "d") if color_group == "red" else ("s", "c")
    scores: list[tuple[str, float]] = []
    for suit in allowed:
        for template in _SUIT_TEMPLATES.get(suit, []):
            score = float(cv2.matchTemplate(query, template, cv2.TM_CCOEFF_NORMED)[0, 0])
            scores.append((suit, score))
    if not scores:
        return "待确认", 0.0, 0.0, "花色模板未加载"
    # Keep the best reference score for each suit.
    best_by_suit = {s: max((score for label, score in scores if label == s), default=-1.0) for s in allowed}
    ordered = sorted(best_by_suit.items(), key=lambda item: item[1], reverse=True)
    suit, best = ordered[0]
    second = ordered[1][1]
    margin = float(best - second)
    confidence = max(0.0, min(1.0, (best * 0.7) + (max(0.0, margin) * 0.8)))
    if best < 0.62 or margin < 0.04:
        return "待确认", confidence, margin, f"花色模板不确定(top={best:.2f}, 差值={margin:.2f})"
    return suit, confidence, margin, "模板匹配"


def recognize_suit_glyph(roi: np.ndarray) -> tuple[str, float, float, str]:
    """Classify an already-cropped, mostly isolated suit glyph (used for hero cards)."""
    if roi is None or roi.size == 0:
        return "待确认", 0.0, 0.0, "花色区域为空"
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    red_mask = (((hsv[:, :, 0] <= 12) | (hsv[:, :, 0] >= 165)) &
                (hsv[:, :, 1] >= 70) & (hsv[:, :, 2] >= 65))
    dark_mask = (hsv[:, :, 2] <= 100) & (hsv[:, :, 1] <= 130)
    red_amount, dark_amount = int(red_mask.sum()), int(dark_mask.sum())
    if red_amount < 15 and dark_amount < 15:
        return "待确认", 0.0, 0.0, "未提取到稳定花色图形"
    color_group = "red" if red_amount > dark_amount * 1.15 else "black"
    pix = red_mask if color_group == "red" else dark_mask
    mask = pix.astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    n, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask)
    comps = []
    for i in range(1, n):
        xx, yy, ww, hh, area = stats[i]
        if area >= 8:
            comps.append((int(area), int(xx), int(yy), int(ww), int(hh)))
    if not comps:
        return "待确认", 0.0, 0.0, "花色图形提取失败"
    area, xx, yy, ww, hh = max(comps, key=lambda item: item[0])
    shape = mask[yy:yy + hh, xx:xx + ww]
    scale = min(46 / max(ww, 1), 46 / max(hh, 1))
    nw, nh = max(1, int(ww * scale)), max(1, int(hh * scale))
    shape = cv2.resize(shape, (nw, nh), interpolation=cv2.INTER_NEAREST)
    query = np.zeros((64, 64), dtype=np.uint8)
    sx, sy = (64 - nw) // 2, (64 - nh) // 2
    query[sy:sy + nh, sx:sx + nw] = shape
    allowed = ("h", "d") if color_group == "red" else ("s", "c")
    scores = {}
    for suit in allowed:
        values = [float(cv2.matchTemplate(query, template, cv2.TM_CCOEFF_NORMED)[0, 0])
                  for template in _SUIT_TEMPLATES.get(suit, [])]
        scores[suit] = max(values, default=-1.0)
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    suit, best = ordered[0]
    second = ordered[1][1]
    margin = float(best - second)
    confidence = max(0.0, min(1.0, (best * 0.7) + max(0.0, margin) * 0.8))
    if best < 0.62 or margin < 0.04:
        return "待确认", confidence, margin, f"花色模板不确定(top={best:.2f}, 差值={margin:.2f})"
    return suit, confidence, margin, "花色图形模板匹配"


def detect_table_profile(frame: np.ndarray) -> str:
    """Coarse visual profile selector used only for fixed hero-card ROIs."""
    h, w = frame.shape[:2]
    scale = 1.0 if w >= 1350 else w / 1350.0
    xa, xb = max(0, int(600 * scale)), min(w, int(900 * scale))
    ya, yb = max(0, int(550 * scale)), min(h, int(680 * scale))
    if xb <= xa or yb <= ya:
        return "green_table"
    hsv = cv2.cvtColor(frame[ya:yb, xa:xb], cv2.COLOR_BGR2HSV)
    mean_h = float(np.mean(hsv[:, :, 0]))
    mean_s = float(np.mean(hsv[:, :, 1]))
    return "tournament_brown" if mean_h < 35 and mean_s < 115 else "green_table"


def recognize_hero_cards(frame: np.ndarray) -> dict:
    """Prototype for the two hero cards using theme-specific top-left glyph ROIs."""
    profile = detect_table_profile(frame)
    h, w = frame.shape[:2]
    scale = 1.0 if w >= 1350 else w / 1350.0
    if profile == "tournament_brown":
        ranks = [(632, 745, 55, 65), (710, 744, 60, 65)]
        suits = [(630, 800, 52, 50), (709, 798, 55, 50)]
    else:
        ranks = [(660, 770, 60, 65), (740, 768, 60, 65)]
        suits = [(660, 812, 60, 48), (740, 808, 60, 48)]
    out = []
    for i, (rank_box, suit_box) in enumerate(zip(ranks, suits), 1):
        rx, ry, rw, rh = [int(v * scale) for v in rank_box]
        sx, sy, sw, sh = [int(v * scale) for v in suit_box]
        rank_roi = frame[max(0, ry):min(h, ry + rh), max(0, rx):min(w, rx + rw)]
        suit_roi = frame[max(0, sy):min(h, sy + sh), max(0, sx):min(w, sx + sw)]
        # The fixed bottom ROIs may overlap an avatar/action label when the user's
        # hole cards are folded or not dealt. Require card-like white background
        # in both ROIs before running OCR/template matching to avoid hallucinations.
        def white_ratio(roi):
            if roi is None or roi.size == 0:
                return 0.0
            _hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            return float(((_hsv[:, :, 2] > 180) & (_hsv[:, :, 1] < 110)).mean())
        rank_white = white_ratio(rank_roi)
        suit_white = white_ratio(suit_roi)
        visible = rank_white >= 0.42 and suit_white >= 0.36
        if not visible:
            rank, suit, rc, sc = "待确认", "待确认", 0.0, 0.0
            rank_note = f"固定手牌区域未检测到足够白底（点数区={rank_white:.2f}，花色区={suit_white:.2f}）"
            suit_note = "可能是未发牌、已弃牌或手牌被遮挡；不输出猜测"
        else:
            rank, rc, rank_note = recognize_rank_roi(rank_roi)
            suit, sc, margin, suit_note = recognize_suit_glyph(suit_roi)
        confidence = min(rc, sc) if rank != "待确认" and suit != "待确认" else 0.0
        status = "单帧候选（需跨帧确认）" if confidence >= 0.65 else "待确认"
        out.append({
            "index": i, "rank": rank, "suit": suit,
            "suit_symbol": SUIT_SYMBOL.get(suit, "?"),
            "rank_confidence": round(float(rc), 3), "suit_confidence": round(float(sc), 3),
            "confidence": round(float(confidence), 3), "status": status,
            "visible_candidate": visible,
            "rank_white_ratio": round(rank_white, 3), "suit_white_ratio": round(suit_white, 3),
            "rank_roi": [rx, ry, rw, rh], "suit_roi": [sx, sy, sw, sh],
            "note": f"{rank_note}；{suit_note}"
        })
    return {"profile": profile, "cards": out, "status": "待确认",
            "note": "底部两张牌按固定主题 ROI 识别；窗口缩放或主题变化时必须重新标定。"}


def detect_board_boxes(frame: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Locate face-up board-card-like rectangles in the central felt region."""
    if frame is None or frame.size == 0:
        return []
    height, width = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, np.array([0, 0, 180], np.uint8), np.array([180, 110, 255], np.uint8))
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    n, _labels, stats, _centroids = cv2.connectedComponentsWithStats(white)
    boxes: list[tuple[int, int, int, int]] = []
    sx, sy = width / 1920.0, height / 1080.0
    min_x, max_x = int(300 * sx), int(1320 * sx)
    min_y, max_y = int(300 * sy), int(730 * sy)
    min_w, max_w = int(62 * sx), int(190 * sx)
    min_h, max_h = int(85 * sy), int(230 * sy)
    for x, y, w, h, area in stats[1:]:
        if not (min_x <= x <= max_x and min_y <= y <= max_y):
            continue
        if not (min_w <= w <= max_w and min_h <= h <= max_h):
            continue
        aspect = w / max(1.0, h)
        if not (0.38 <= aspect <= 0.98):
            continue
        if area < int(3200 * sx * sy):
            continue
        # Chip stacks occasionally touch the lower card edge; clamp boxes to a card-like height.
        target_h = int(w / 0.70)
        if h > target_h * 1.13:
            h = min(target_h, height - y)
        boxes.append((int(x), int(y), int(w), int(h)))
    boxes.sort(key=lambda b: (b[0], b[1]))
    # Remove near-duplicates caused by adjacent high-brightness components.
    deduped: list[tuple[int, int, int, int]] = []
    for box in boxes:
        x, y, w, h = box
        duplicate = False
        for ox, oy, ow, oh in deduped:
            if abs(x - ox) < min(w, ow) * 0.45 and abs(y - oy) < 12 * sy:
                duplicate = True
                break
        if not duplicate:
            deduped.append(box)
    return deduped


def analyze_card(frame: np.ndarray, bbox: tuple[int, int, int, int], index: int) -> CardObservation:
    white_ratio = _white_ratio(frame, bbox)
    obs = CardObservation(index=index, bbox=bbox, white_ratio=white_ratio)
    if white_ratio < 0.57:
        obs.status = "待确认"
        obs.note = "明牌样式不像标准白底扑克牌（可能是特殊图标/动画）"
        return obs

    rank, rank_conf, rank_engine = recognize_rank(frame, bbox)
    suit, suit_conf, suit_margin, suit_note = recognize_suit(frame, bbox)
    obs.rank = rank
    obs.suit = suit
    obs.suit_symbol = SUIT_SYMBOL.get(suit, "?")
    obs.rank_confidence = round(float(rank_conf), 3)
    obs.suit_confidence = round(float(suit_conf), 3)
    obs.confidence = round(float(min(rank_conf, suit_conf)), 3) if rank != "待确认" and suit != "待确认" else 0.0
    notes = [rank_engine, suit_note]
    if rank == "待确认":
        notes.append("点数需要人工核对")
    if suit == "待确认":
        notes.append("花色需要人工核对")
    obs.note = "；".join(notes)
    if rank != "待确认" and suit != "待确认" and obs.confidence >= 0.65:
        obs.status = "单帧候选（需跨帧确认）"
    else:
        obs.status = "待确认"
    return obs


def analyze_frame(frame: np.ndarray) -> dict:
    boxes = detect_board_boxes(frame)
    observations = [analyze_card(frame, box, i + 1) for i, box in enumerate(boxes)]
    standard_cards = [o for o in observations if o.white_ratio >= 0.57]
    count = len(standard_cards)
    if count == 0:
        street = "翻牌前或牌面暂不可见"
    elif count == 3:
        street = "翻牌圈候选"
    elif count == 4:
        street = "转牌圈候选"
    elif count == 5:
        street = "河牌圈候选"
    else:
        street = "动画/部分可见/待确认"
    hero = recognize_hero_cards(frame)
    validation = []
    known = {}
    for card in observations:
        if card.rank in RANKS and card.suit in SUITS:
            identity = (card.rank, card.suit)
            if identity in known:
                prev_index = known[identity]
                validation.append({"type": "duplicate_board_card", "card": card.rank + SUIT_SYMBOL[card.suit],
                                  "indices": [prev_index, card.index],
                                  "message": "同一帧公共牌出现重复实体牌，相关候选需人工核验"})
                observations[prev_index - 1].status = "待确认"
                observations[prev_index - 1].note += "；重复实体牌校验未通过"
                card.status = "待确认"
                card.note += "；重复实体牌校验未通过"
            else:
                known[identity] = card.index
    for card in hero.get("cards", []):
        if card.get("rank") in RANKS and card.get("suit") in SUITS:
            identity = (card["rank"], card["suit"])
            if identity in known:
                validation.append({"type": "hero_board_collision", "card": card["rank"] + SUIT_SYMBOL[card["suit"]],
                                  "indices": [known[identity], "hero:" + str(card["index"])],
                                  "message": "英雄手牌候选与公共牌候选冲突，至少一个识别结果错误"})
                card["status"] = "待确认"
                card["confidence"] = 0.0
                card["note"] += "；与公共牌候选冲突"
                if isinstance(known[identity], int):
                    board_card = observations[known[identity] - 1]
                    board_card.status = "待确认"
                    board_card.note += "；与英雄手牌候选冲突"
            else:
                known[identity] = "hero:" + str(card["index"])
    if len(standard_cards) > 5:
        validation.append({"type": "too_many_board_cards", "count": count,
                           "message": "标准公共牌候选超过5张，牌框定位或桌面主题可能异常"})
    return {
        "frame_width": int(frame.shape[1]),
        "frame_height": int(frame.shape[0]),
        "board_card_count_candidate": count,
        "street_candidate": street,
        "hero_hand_candidates": hero,
        "cards": [o.to_dict() for o in observations],
        "validation": validation,
        "status": "待确认",
        "note": "当前阶段覆盖公共牌与可见英雄手牌候选；无可靠证据时不猜。底池/筹码/行动识别安排在后续阶段。",
    }


def draw_observations(frame: np.ndarray, result: dict) -> np.ndarray:
    view = frame.copy()
    for c in result.get("cards", []):
        x, y, w, h = c["bbox"]
        normal = c.get("rank", "待确认") != "待确认" and c.get("suit", "待确认") != "待确认"
        color = (50, 170, 60) if normal and c.get("status", "").startswith("单帧") else (0, 170, 230)
        cv2.rectangle(view, (x, y), (x + w, y + h), color, 3)
        label = f"{c.get('rank','?')}{c.get('suit_symbol','?')} {c.get('confidence',0):.2f}"
        cv2.rectangle(view, (x, max(0, y - 28)), (x + max(w, 100), y), color, -1)
        cv2.putText(view, label, (x + 3, max(15, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    for c in result.get("hero_hand_candidates", {}).get("cards", []):
        if not c.get("visible_candidate", False):
            continue
        rx, ry, rw, rh = c.get("rank_roi", [0, 0, 1, 1])
        sx, sy, sw, sh = c.get("suit_roi", [0, 0, 1, 1])
        color = (50, 170, 60) if c.get("status", "").startswith("单帧") else (0, 170, 230)
        cv2.rectangle(view, (rx, ry), (rx + rw, ry + rh), color, 2)
        cv2.rectangle(view, (sx, sy), (sx + sw, sy + sh), color, 2)
        label = f"Hero {c.get('rank','?')}{c.get('suit_symbol','?')} {c.get('confidence',0):.2f}"
        cv2.putText(view, label, (rx, max(18, ry - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return view


class TemporalCardTracker:
    """Cross-frame card voter; it deliberately does not infer bets/actions yet."""
    def __init__(self, window_size: int = 5, min_repeats: int = 2, min_confidence: float = 0.65):
        self.window_size = max(2, int(window_size))
        self.min_repeats = max(2, int(min_repeats))
        self.min_confidence = max(0.0, min(1.0, float(min_confidence)))
        self.history: list[tuple[float, dict]] = []

    def _vote_cards(self, current: list[dict], history_key: str, timestamp: float) -> list[dict]:
        stable = []
        for idx, card in enumerate(current):
            key = (card.get("rank"), card.get("suit"))
            eligible_now = (key[0] in RANKS and key[1] in SUITS and
                            float(card.get("confidence", 0.0)) >= self.min_confidence)
            votes = 0
            if eligible_now:
                for _ts, prev_result in self.history:
                    prev_cards = prev_result.get(history_key, [])
                    if idx < len(prev_cards):
                        prev = prev_cards[idx]
                        prev_key = (prev.get("rank"), prev.get("suit"))
                        if (prev_key == key and prev_key[0] in RANKS and prev_key[1] in SUITS and
                                float(prev.get("confidence", 0.0)) >= self.min_confidence):
                            votes += 1
            item = dict(card)
            item["temporal_votes"] = votes
            if eligible_now and votes >= self.min_repeats - 1:
                item["status"] = "跨帧稳定候选（仍需必要时人工确认）"
            elif not eligible_now:
                item["status"] = "待确认"
            else:
                item["status"] = "单帧候选（等待下一帧）"
            stable.append(item)
        return stable

    def update(self, timestamp: float, result: dict) -> dict:
        # Use history from earlier frames first, then append the current frame.
        stable_board = self._vote_cards(result.get("cards", []), "cards", float(timestamp))
        hero_copy = dict(result.get("hero_hand_candidates", {}))
        hero_copy["cards"] = self._vote_cards(hero_copy.get("cards", []), "hero_cards", float(timestamp))
        updated = dict(result)
        updated["cards"] = stable_board
        updated["hero_hand_candidates"] = hero_copy
        updated["sample_timestamp_s"] = round(float(timestamp), 3)
        updated["temporal_window_size"] = len(self.history) + 1
        history_result = {
            "cards": [dict(c) for c in result.get("cards", [])],
            "hero_cards": [dict(c) for c in result.get("hero_hand_candidates", {}).get("cards", [])],
        }
        self.history.append((float(timestamp), history_result))
        self.history = self.history[-self.window_size:]
        return updated
