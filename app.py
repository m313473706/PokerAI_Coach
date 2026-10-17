import ctypes
import ctypes.wintypes as wintypes
import json
import os
import queue
import re
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

IMPORT_ERROR = None
try:
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw, ImageEnhance, ImageGrab, ImageOps, ImageTk
    from rapidocr_onnxruntime import RapidOCR
except Exception as exc:
    IMPORT_ERROR = exc

from poker_engine import estimate_equity, evaluate_hand, self_check as engine_self_check

APP_TITLE = "Poker AI Coach — Windows 屏幕采集与视觉理解"
if os.name == "nt":
    try:
        ctypes.WinDLL("user32", use_last_error=True).SetProcessDPIAware()
    except Exception:
        pass
CARD_CODE = re.compile(r"(?<![A-Z0-9])(10|[2-9TJQKA])\s*([SHDC♠♥♦♣])(?![A-Z0-9])", re.I)
CARD_RANK = re.compile(r"(?<![A-Z0-9])(10|[2-9TJQKA])(?![A-Z0-9])", re.I)
MONEY_RE = re.compile(r"(?<![A-Z0-9])(?:[$€£¥]\s*)?(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)(?![A-Z0-9])")
SUIT_MAP = {"S": "♠", "H": "♥", "D": "♦", "C": "♣", "♠": "♠", "♥": "♥", "♦": "♦", "♣": "♣"}

@dataclass
class WindowInfo:
    handle: int
    title: str
    class_name: str
    bounds: tuple

def _windows_api():
    if os.name != "nt":
        raise RuntimeError("此程序只能在 Windows 上运行。")
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.IsWindowVisible.argtypes = [wintypes.HWND]
    user32.IsWindowVisible.restype = wintypes.BOOL
    user32.IsIconic.argtypes = [wintypes.HWND]
    user32.IsIconic.restype = wintypes.BOOL
    user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
    user32.GetWindowTextLengthW.restype = ctypes.c_int
    user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    user32.GetWindowTextW.restype = ctypes.c_int
    user32.GetClassNameW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    user32.GetClassNameW.restype = ctypes.c_int
    user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    user32.GetWindowRect.restype = wintypes.BOOL
    return user32

def visible_windows():
    user32 = _windows_api()
    enum_proc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    rows = []

    def visit(hwnd, _):
        if not user32.IsWindowVisible(hwnd):
            return True
        size = user32.GetWindowTextLengthW(hwnd)
        if size < 1:
            return True
        title_buffer = ctypes.create_unicode_buffer(size + 1)
        user32.GetWindowTextW(hwnd, title_buffer, size + 1)
        title = title_buffer.value.strip()
        if not title or title.startswith(APP_TITLE):
            return True
        class_buffer = ctypes.create_unicode_buffer(256)
        user32.GetClassNameW(hwnd, class_buffer, 256)
        rect = wintypes.RECT()
        if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            return True
        bounds = (rect.left, rect.top, rect.right, rect.bottom)
        if bounds[2] - bounds[0] < 120 or bounds[3] - bounds[1] < 100:
            return True
        rows.append(WindowInfo(int(hwnd), title, class_buffer.value, bounds))
        return True

    callback = enum_proc(visit)
    user32.EnumWindows(callback, 0)
    rows.sort(key=lambda row: (0 if any(word in row.title.lower() for word in ("poker", "holdem", "texas")) else 1, row.title.lower()))
    return rows

class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", wintypes.DWORD), ("biWidth", wintypes.LONG), ("biHeight", wintypes.LONG),
        ("biPlanes", wintypes.WORD), ("biBitCount", wintypes.WORD), ("biCompression", wintypes.DWORD),
        ("biSizeImage", wintypes.DWORD), ("biXPelsPerMeter", wintypes.LONG),
        ("biYPelsPerMeter", wintypes.LONG), ("biClrUsed", wintypes.DWORD), ("biClrImportant", wintypes.DWORD),
    ]

class RGBQUAD(ctypes.Structure):
    _fields_ = [("rgbBlue", ctypes.c_ubyte), ("rgbGreen", ctypes.c_ubyte), ("rgbRed", ctypes.c_ubyte), ("rgbReserved", ctypes.c_ubyte)]

class BITMAPINFO(ctypes.Structure):
    _fields_ = [("bmiHeader", BITMAPINFOHEADER), ("bmiColors", RGBQUAD * 1)]

def _capture_printwindow(handle, width, height):
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)
    hwnd = wintypes.HWND(handle)
    user32.GetWindowDC.argtypes = [wintypes.HWND]
    user32.GetWindowDC.restype = ctypes.c_void_p
    user32.ReleaseDC.argtypes = [wintypes.HWND, ctypes.c_void_p]
    user32.ReleaseDC.restype = ctypes.c_int
    user32.PrintWindow.argtypes = [wintypes.HWND, ctypes.c_void_p, wintypes.UINT]
    user32.PrintWindow.restype = wintypes.BOOL
    gdi32.CreateCompatibleDC.argtypes = [ctypes.c_void_p]
    gdi32.CreateCompatibleDC.restype = ctypes.c_void_p
    gdi32.CreateCompatibleBitmap.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    gdi32.CreateCompatibleBitmap.restype = ctypes.c_void_p
    gdi32.SelectObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    gdi32.SelectObject.restype = ctypes.c_void_p
    gdi32.DeleteObject.argtypes = [ctypes.c_void_p]
    gdi32.DeleteObject.restype = wintypes.BOOL
    gdi32.DeleteDC.argtypes = [ctypes.c_void_p]
    gdi32.DeleteDC.restype = wintypes.BOOL
    gdi32.GetDIBits.argtypes = [ctypes.c_void_p, ctypes.c_void_p, wintypes.UINT, wintypes.UINT, ctypes.c_void_p, ctypes.POINTER(BITMAPINFO), wintypes.UINT]
    gdi32.GetDIBits.restype = ctypes.c_int

    window_dc = user32.GetWindowDC(hwnd)
    if not window_dc:
        return None
    memory_dc = None
    bitmap = None
    previous = None
    try:
        memory_dc = gdi32.CreateCompatibleDC(window_dc)
        bitmap = gdi32.CreateCompatibleBitmap(window_dc, width, height)
        if not memory_dc or not bitmap:
            return None
        previous = gdi32.SelectObject(memory_dc, bitmap)
        if not user32.PrintWindow(hwnd, memory_dc, 2):
            return None
        if previous:
            gdi32.SelectObject(memory_dc, previous)
            previous = None
        info = BITMAPINFO()
        info.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        info.bmiHeader.biWidth = width
        info.bmiHeader.biHeight = -height
        info.bmiHeader.biPlanes = 1
        info.bmiHeader.biBitCount = 32
        info.bmiHeader.biCompression = 0
        info.bmiHeader.biSizeImage = width * height * 4
        pixels = ctypes.create_string_buffer(width * height * 4)
        rows = gdi32.GetDIBits(memory_dc, bitmap, 0, height, pixels, ctypes.byref(info), 0)
        if rows != height:
            return None
        return Image.frombuffer("RGB", (width, height), pixels.raw, "raw", "BGRX", 0, 1).copy()
    finally:
        if previous and memory_dc:
            gdi32.SelectObject(memory_dc, previous)
        if bitmap:
            gdi32.DeleteObject(bitmap)
        if memory_dc:
            gdi32.DeleteDC(memory_dc)
        user32.ReleaseDC(hwnd, window_dc)

def capture_window(window):
    user32 = _windows_api()
    if user32.IsIconic(window.handle):
        raise RuntimeError("选中的牌桌窗口已最小化，请先恢复窗口。")
    rect = wintypes.RECT()
    if not user32.GetWindowRect(window.handle, ctypes.byref(rect)):
        raise RuntimeError("无法读取所选窗口的位置。")
    box = (rect.left, rect.top, rect.right, rect.bottom)
    width = box[2] - box[0]
    height = box[3] - box[1]
    if width < 120 or height < 100:
        raise RuntimeError("所选窗口区域太小，无法识别。")
    # 优先请求 Windows 直接绘制目标窗口，减少教练窗口遮挡的影响；不支持时回退到屏幕区域采集。
    try:
        rendered = _capture_printwindow(window.handle, width, height)
        if rendered is not None:
            return rendered
    except Exception:
        pass
    return ImageGrab.grab(bbox=box, all_screens=True).convert("RGB")
def normalize_card_text(text):
    value = str(text).upper().replace("10", "T").replace(" ", "")
    match = CARD_CODE.search(value)
    if match:
        rank, suit = match.groups()
        return rank.replace("10", "T"), SUIT_MAP[suit.upper()]
    return None, None

def ocr_lines(engine, image):
    result, _ = engine(np.asarray(image.convert("RGB")))
    lines = []
    for row in result or []:
        try:
            polygon, text, confidence = row
            points = np.asarray(polygon, dtype=float).reshape(-1, 2)
            x1, y1 = points.min(axis=0)
            x2, y2 = points.max(axis=0)
            clean = str(text).strip()
            if clean:
                lines.append({
                    "text": clean,
                    "confidence": round(float(confidence), 3),
                    "box": [int(x1), int(y1), int(x2), int(y2)],
                    "center": [float((x1 + x2) / 2), float((y1 + y2) / 2)],
                })
        except Exception:
            continue
    return lines

def _intersection_over_union(a, b):
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])
    if right <= left or bottom <= top:
        return 0.0
    overlap = (right - left) * (bottom - top)
    area_a = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1, (b[2] - b[0]) * (b[3] - b[1]))
    return overlap / float(area_a + area_b - overlap)

def find_card_regions(image):
    rgb = np.asarray(image.convert("RGB"))
    frame_h, frame_w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    mask = cv2.threshold(gray, 205, 255, cv2.THRESH_BINARY)[1]
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    minimum_area = max(450, int(frame_w * frame_h * 0.00045))
    maximum_area = int(frame_w * frame_h * 0.055)
    candidates = []
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        area = width * height
        ratio = width / float(max(1, height))
        if area < minimum_area or area > maximum_area:
            continue
        if height < 24 or width < 14 or height > frame_h * 0.36:
            continue
        if not 0.48 <= ratio <= 0.88:
            continue
        contour_area = cv2.contourArea(contour)
        if contour_area / float(max(1, area)) < 0.55:
            continue
        candidates.append([x, y, x + width, y + height])
    candidates.sort(key=lambda box: (box[2] - box[0]) * (box[3] - box[1]), reverse=True)
    unique = []
    for box in candidates:
        if all(_intersection_over_union(box, prior) < 0.58 for prior in unique):
            unique.append(box)
    unique.sort(key=lambda box: ((box[1] + box[3]) / 2, (box[0] + box[2]) / 2))
    return unique[:18]

def read_card_corner(engine, image, box):
    x1, y1, x2, y2 = box
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    corner = image.crop((x1, y1, x1 + int(width * 0.58), y1 + int(height * 0.50)))
    corner = ImageOps.autocontrast(corner.convert("L"))
    corner = corner.resize((max(80, corner.width * 4), max(80, corner.height * 4)))
    rows = ocr_lines(engine, corner)
    combined = " ".join(row["text"] for row in rows)
    rank, suit = normalize_card_text(combined)
    if rank is None:
        match = CARD_RANK.search(combined.upper().replace("10", "T"))
        if match:
            rank = match.group(1).replace("10", "T")
    confidence = max((row["confidence"] for row in rows), default=0.0)
    identity = (rank or "?") + (suit or "?")
    return identity, confidence

def classify_zone(box, frame_size):
    width, height = frame_size
    cx = ((box[0] + box[2]) / 2) / max(1, width)
    cy = ((box[1] + box[3]) / 2) / max(1, height)
    if 0.24 <= cx <= 0.76 and cy >= 0.62:
        return "hero候选", cx, cy
    if 0.18 <= cx <= 0.82 and 0.27 <= cy < 0.62:
        return "公共牌候选", cx, cy
    return "其他牌区", cx, cy

def extract_money_candidates(lines):
    candidates = []
    for line in lines:
        text = line["text"]
        for match in MONEY_RE.finditer(text):
            value = match.group(1).replace(",", "")
            label = "底池候选" if any(word in text.lower() for word in ("pot", "底池")) else "筹码/金额候选"
            if len(text) > 40:
                continue
            candidates.append({
                "value": value,
                "label": label,
                "source_text": text,
                "confidence": line["confidence"],
                "box": line["box"],
            })
    return candidates

def extract_action_candidates(lines):
    keywords = ("fold", "check", "call", "bet", "raise", "all-in", "all in", "弃牌", "过牌", "跟注", "下注", "加注", "全下")
    found = []
    for line in lines:
        low = line["text"].lower()
        if any(word in low for word in keywords):
            found.append({"text": line["text"], "confidence": line["confidence"], "box": line["box"]})
    return found

def make_state(window, image, lines, card_boxes, card_values):
    frame_w, frame_h = image.size
    cards = []
    for box, (identity, confidence) in zip(card_boxes, card_values):
        zone, cx, cy = classify_zone(box, image.size)
        cards.append({
            "candidate": identity,
            "zone": zone,
            "confidence": round(confidence, 3),
            "box": box,
            "center_normalized": [round(cx, 3), round(cy, 3)],
            "stable": False,
            "source": "卡牌区域检测 + 卡角文字识别",
        })
    board = [card["candidate"] for card in cards if card["zone"] == "公共牌候选" and "?" not in card["candidate"]]
    hero = [card["candidate"] for card in cards if card["zone"] == "hero候选" and "?" not in card["candidate"]]
    street = {3: "翻牌", 4: "转牌", 5: "河牌"}.get(len(board), "待识别")
    return {
        "采集时间": datetime.now().astimezone().isoformat(timespec="seconds"),
        "目标窗口": {"标题": window.title, "窗口类名": window.class_name},
        "画面": {"宽度": frame_w, "高度": frame_h},
        "牌局候选": {
            "阶段": {"值": street, "状态": "根据公共牌区域候选数量推断" if street != "待识别" else "证据不足"},
            "公共牌": {"候选": board, "状态": "视觉候选，需结合目标模拟器验证"},
            "玩家手牌": {"候选": hero, "状态": "视觉候选，需结合目标模拟器验证"},
            "牌面区域": cards,
            "底池与筹码": extract_money_candidates(lines),
            "行动文字": extract_action_candidates(lines),
        },
        "画面文字": lines,
        "说明": "本程序仅输出屏幕证据和视觉候选；未可靠识别的字段不会猜测，也不会据此生成策略建议。",
    }

def build_coach_advice(state):
    table = state.get("牌局候选", {})
    cards = table.get("牌面区域", [])
    hero_rows = sorted(
        [card for card in cards if card.get("zone") == "hero候选" and card.get("stable")],
        key=lambda card: card.get("center_normalized", [0, 0])[0],
    )
    board_rows = sorted(
        [card for card in cards if card.get("zone") == "公共牌候选" and card.get("stable")],
        key=lambda card: card.get("center_normalized", [0, 0])[0],
    )
    if len(hero_rows) != 2:
        return {"状态": "等待识别", "牌型": "—", "对随机手牌胜率": "—", "底池赔率": "—",
                "建议": "等待屏幕稳定识别到两张自己的手牌。"}
    if len(board_rows) not in (0, 3, 4, 5):
        return {"状态": "等待牌面稳定", "牌型": "—", "对随机手牌胜率": "—", "底池赔率": "—",
                "建议": "公共牌数量尚未稳定，暂不生成建议。"}

    selected = hero_rows + board_rows
    if any("?" in card.get("candidate", "") or card.get("confidence", 0) < 0.45 for card in selected):
        return {"状态": "识别置信度不足", "牌型": "—", "对随机手牌胜率": "—", "底池赔率": "—",
                "建议": "牌面花色或点数尚未可靠识别；请保持牌桌清晰可见，暂不猜测行动。"}
    hole = [card["candidate"] for card in hero_rows]
    board = [card["candidate"] for card in board_rows]
    try:
        equity_result = estimate_equity(hole, board, iterations=400)
        equity = equity_result["equity"]
        hand_name = evaluate_hand(hole + board)[2] if len(board) >= 3 else "翻牌前"
    except ValueError as exc:
        return {"状态": "牌面冲突", "牌型": "—", "对随机手牌胜率": "—", "底池赔率": "—",
                "建议": "识别结果存在冲突，暂不分析：" + str(exc)}

    money = table.get("底池与筹码", [])
    pot_item = next((item for item in money if item.get("label") == "底池候选"), None)
    call_item = next((item for item in money if any(word in item.get("source_text", "").lower()
                      for word in ("call", "跟注"))), None)
    pot = call_amount = None
    try:
        if pot_item:
            pot = float(pot_item["value"])
        if call_item:
            call_amount = float(call_item["value"])
    except (TypeError, ValueError):
        pot = call_amount = None

    pot_odds = None
    if pot is not None and call_amount is not None and pot > 0 and call_amount >= 0:
        pot_odds = call_amount / (pot + call_amount) if pot + call_amount else None
    if pot_odds is not None:
        margin = equity - pot_odds
        if margin >= 0.08:
            suggestion = "当前胜率高于识别到的跟注门槛，可考虑继续；仍需留意对手范围和识别误差。"
        elif margin <= -0.08:
            suggestion = "当前胜率低于识别到的跟注门槛，弃牌方向更稳妥。"
        else:
            suggestion = "处于赔率边界，谨慎处理；请核对底池和跟注金额识别。"
        odds_text = f"{pot_odds:.1%}（跟注 {call_amount:g} / 底池 {pot:g}）"
    else:
        odds_text = "未能同时识别底池与跟注金额"
        if equity >= 0.72:
            suggestion = "对随机手牌基准胜率较高；未读到下注金额和对手范围，暂不自动建议下注尺寸。"
        elif equity <= 0.35:
            suggestion = "对随机手牌基准胜率偏低；如当前需要跟注，先核对底池赔率。"
        else:
            suggestion = "目前只给出对随机手牌的胜率基准；下注金额或对手范围不足，暂不推荐具体动作。"
    return {
        "状态": "自动分析",
        "牌型": hand_name,
        "对随机手牌胜率": f"{equity:.1%}（蒙特卡洛 {equity_result['samples']} 次）",
        "底池赔率": odds_text,
        "建议": suggestion,
        "分析依据": "仅使用连续稳定的屏幕识别结果；胜率对手模型为随机两张手牌，不等于真实对手范围。",
    }

def annotate_image(image, state):
    output = image.copy()
    draw = ImageDraw.Draw(output)
    for card in state["牌局候选"]["牌面区域"]:
        x1, y1, x2, y2 = card["box"]
        color = (30, 180, 80) if "?" not in card["candidate"] else (240, 160, 30)
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        zone_tag = {"hero候选": "HERO", "公共牌候选": "BOARD", "其他牌区": "CARD"}.get(card["zone"], "CARD")
        draw.text((x1, max(0, y1 - 18)), zone_tag + " " + card["candidate"], fill=color)
    for line in state["画面文字"]:
        x1, y1, x2, y2 = line["box"]
        if line["confidence"] >= 0.60:
            draw.rectangle((x1, y1, x2, y2), outline=(60, 120, 240), width=1)
    return output

class VisionApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1320x860")
        self.minsize(1050, 700)
        self.windows = []
        self.stop_event = threading.Event()
        self.messages = queue.Queue(maxsize=2)
        self.thread = None
        self.last_state = None
        self.last_image = None
        self.card_history = {}
        self._build()
        self.refresh_windows()
        self.after(120, self._poll_messages)
        self.protocol("WM_DELETE_WINDOW", self.close)

    def _build(self):
        top = ttk.Frame(self, padding=(12, 10))
        top.pack(fill="x")
        ttk.Label(top, text="Windows 屏幕采集与视觉理解", font=("Segoe UI", 18, "bold")).pack(side="left")
        controls = ttk.Frame(self, padding=(12, 0, 12, 10))
        controls.pack(fill="x")
        ttk.Label(controls, text="牌桌窗口").pack(side="left")
        self.window_choice = ttk.Combobox(controls, state="readonly", width=76)
        self.window_choice.pack(side="left", padx=(8, 8))
        ttk.Button(controls, text="刷新窗口", command=self.refresh_windows).pack(side="left", padx=3)
        self.start_button = ttk.Button(controls, text="开始实时识别", command=self.start_scan)
        self.start_button.pack(side="left", padx=3)
        self.stop_button = ttk.Button(controls, text="停止", command=self.stop_scan, state="disabled")
        self.stop_button.pack(side="left", padx=3)
        ttk.Button(controls, text="导出识别记录", command=self.export_record).pack(side="left", padx=3)

        panel = ttk.Panedwindow(self, orient="horizontal")
        panel.pack(fill="both", expand=True, padx=12, pady=(0, 8))
        left = ttk.Frame(panel, padding=6)
        right = ttk.Frame(panel, padding=6)
        panel.add(left, weight=3)
        panel.add(right, weight=2)
        ttk.Label(left, text="实时牌桌画面（绿色为识别到的牌区，蓝色为文字）").pack(anchor="w", pady=(0, 5))
        self.preview = ttk.Label(left, text="选择牌桌窗口后开始实时识别", anchor="center", relief="solid")
        self.preview.pack(fill="both", expand=True)
        ttk.Label(right, text="自动识别与教练建议").pack(anchor="w", pady=(0, 5))
        self.coach = ttk.Label(right, text="等待实时屏幕识别……", wraplength=430, justify="left", foreground="#154360")
        self.coach.pack(fill="x", anchor="w", pady=(0, 8))
        self.output = tk.Text(right, wrap="word", font=("Segoe UI", 10), state="disabled")
        self.output.pack(fill="both", expand=True)
        bottom = ttk.Frame(self, padding=(12, 0, 12, 10))
        bottom.pack(fill="x")
        self.status = ttk.Label(bottom, text="状态：请选择要采集的牌桌窗口")
        self.status.pack(anchor="w")

    def refresh_windows(self):
        try:
            self.windows = visible_windows()
        except Exception as exc:
            self.status.config(text=f"窗口枚举失败：{exc}")
            self.window_choice["values"] = []
            return
        values = [f"{row.title}    [{row.class_name}]    {row.bounds[2] - row.bounds[0]}×{row.bounds[3] - row.bounds[1]}" for row in self.windows]
        self.window_choice["values"] = values
        if values:
            self.window_choice.current(0)
            self.status.config(text=f"找到 {len(values)} 个可见窗口。请选择牌桌窗口，然后点击“开始实时识别”。")
        else:
            self.status.config(text="当前没有发现可采集的可见窗口。")

    def selected_window(self):
        index = self.window_choice.current()
        if index < 0 or index >= len(self.windows):
            return None
        return self.windows[index]

    def start_scan(self):
        if self.thread and self.thread.is_alive():
            return
        window = self.selected_window()
        if window is None:
            messagebox.showwarning("还没有选择窗口", "请先刷新并选择牌桌窗口。")
            return
        if IMPORT_ERROR:
            messagebox.showerror("视觉组件未能加载", str(IMPORT_ERROR))
            return
        self.stop_event.clear()
        self.card_history.clear()
        self.thread = threading.Thread(target=self._scan_loop, args=(window,), daemon=True)
        self.thread.start()
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.status.config(text="正在启动本地视觉模型；首次启动可能需要稍等。")

    def stop_scan(self):
        self.stop_event.set()
        self.start_button.config(state="disabled")
        self.stop_button.config(state="disabled")
        self.status.config(text="正在停止实时识别……")

    def _offer(self, item):
        try:
            self.messages.put_nowait(item)
        except queue.Full:
            try:
                self.messages.get_nowait()
            except queue.Empty:
                pass
            try:
                self.messages.put_nowait(item)
            except queue.Full:
                pass

    def _mark_stability(self, cards):
        for card in cards:
            cx, cy = card["center_normalized"]
            key = (card["zone"], round(cx / 0.045), round(cy / 0.06))
            history = self.card_history.setdefault(key, [])
            history.append(card["candidate"])
            del history[:-5]
            card["stable"] = card["candidate"] != "??" and history.count(card["candidate"]) >= 3
            if card["stable"]:
                card["状态"] = "连续帧稳定候选"
            else:
                card["状态"] = "单帧或未稳定候选"

    def _scan_loop(self, window):
        try:
            engine = RapidOCR()
            self._offer(("status", "本地视觉模型已加载，正在采集牌桌窗口……"))
            while not self.stop_event.is_set():
                started = time.monotonic()
                frame = capture_window(window)
                lines = ocr_lines(engine, frame)
                boxes = find_card_regions(frame)
                values = []
                for box in boxes:
                    x1, y1, x2, y2 = box
                    crop = frame.crop((x1, y1, x2, y2))
                    values.append(read_card_corner(engine, crop, (0, 0, crop.width, crop.height)))
                state = make_state(window, frame, lines, boxes, values)
                self._mark_stability(state["牌局候选"]["牌面区域"])
                state["教练分析"] = build_coach_advice(state)
                annotated = annotate_image(frame, state)
                self._offer(("frame", annotated, state))
                elapsed = time.monotonic() - started
                self.stop_event.wait(max(0.2, 1.0 - elapsed))
            self._offer(("stopped",))
        except Exception as exc:
            self._offer(("error", str(exc)))

    def _poll_messages(self):
        try:
            while True:
                item = self.messages.get_nowait()
                if item[0] == "status":
                    self.status.config(text=item[1])
                elif item[0] == "stopped":
                    self.status.config(text="已停止实时识别。")
                    self.start_button.config(state="normal")
                    self.stop_button.config(state="disabled")
                elif item[0] == "error":
                    self.status.config(text=f"识别中断：{item[1]}")
                    self.start_button.config(state="normal")
                    self.stop_button.config(state="disabled")
                    self.stop_event.set()
                elif item[0] == "frame":
                    self.last_image, self.last_state = item[1], item[2]
                    self._show_frame(self.last_image)
                    self._show_state(self.last_state)
                    self._show_coach(self.last_state.get("教练分析", {}))
                    count = len(self.last_state["画面文字"])
                    cards = len(self.last_state["牌局候选"]["牌面区域"])
                    self.status.config(text=f"实时识别中：发现 {cards} 个牌面区域、{count} 条文字。识别结果是候选值，尚未针对真实模拟器校准。")
        except queue.Empty:
            pass
        self.after(120, self._poll_messages)

    def _show_frame(self, image):
        shown = image.copy()
        shown.thumbnail((790, 720))
        photo = ImageTk.PhotoImage(shown)
        self.preview.configure(image=photo, text="")
        self.preview.image = photo

    def _show_state(self, state):
        text = json.dumps(state, ensure_ascii=False, indent=2)
        self.output.configure(state="normal")
        self.output.delete("1.0", "end")
        self.output.insert("1.0", text)
        self.output.configure(state="disabled")

    def _show_coach(self, advice):
        if not hasattr(self, "coach"):
            return
        lines = [
            "自动教练分析",
            "状态：" + str(advice.get("状态", "等待识别")),
            "牌型：" + str(advice.get("牌型", "—")),
            "对随机手牌胜率：" + str(advice.get("对随机手牌胜率", "—")),
            "底池赔率：" + str(advice.get("底池赔率", "—")),
            "建议：" + str(advice.get("建议", "等待识别牌面。")),
        ]
        if advice.get("分析依据"):
            lines.append("依据：" + advice["分析依据"])
        self.coach.configure(text="\n".join(lines))

    def export_record(self):
        if not self.last_state or not self.last_image:
            messagebox.showinfo("暂时没有识别记录", "先启动实时识别，等画面出现后再导出。")
            return
        filename = filedialog.asksaveasfilename(
            title="保存当前识别记录",
            defaultextension=".json",
            filetypes=[("识别记录 JSON", "*.json")],
            initialfile="PokerAI_视觉识别记录.json",
        )
        if not filename:
            return
        path = Path(filename)
        path.write_text(json.dumps(self.last_state, ensure_ascii=False, indent=2), encoding="utf-8")
        self.last_image.save(path.with_suffix(".png"))
        messagebox.showinfo("导出完成", f"识别记录和对应画面已保存到：\n{path}\n{path.with_suffix('.png')}")

    def close(self):
        self.stop_event.set()
        self.destroy()

def self_check():
    if IMPORT_ERROR:
        raise RuntimeError(f"视觉组件加载失败：{IMPORT_ERROR}")
    engine_self_check()
    engine = RapidOCR()
    sample = Image.new("RGB", (320, 240), "white")
    engine(np.asarray(sample))
    return 0

def main():
    if "--self-check" in sys.argv:
        try:
            return self_check()
        except Exception as exc:
            try:
                Path(os.environ.get("TEMP", "."), "PokerAI_Coach_self_check_error.txt").write_text(str(exc), encoding="utf-8")
            except Exception:
                pass
            return 1
    if IMPORT_ERROR:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("程序组件无法加载", f"程序启动失败：{IMPORT_ERROR}")
        root.destroy()
        return 1
    VisionApp().mainloop()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
