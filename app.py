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
    from PIL import Image, ImageDraw, ImageEnhance, ImageGrab, ImageTk
    from rapidocr_onnxruntime import RapidOCR
    from card_vision import (
        MODEL_SOURCE, CardVisionError, TableStateGate, classify_card_regions,
        load_card_model, validate_analysis_gate, inspect_table_state,
    )
except Exception as exc:
    IMPORT_ERROR = exc

WINDOWS_CAPTURE_IMPORT_ERROR = None
try:
    from windows_capture import WindowsCapture
except Exception as exc:
    WindowsCapture = None
    WINDOWS_CAPTURE_IMPORT_ERROR = exc

from poker_engine import estimate_equity, evaluate_hand, self_check as engine_self_check

APP_TITLE = "Poker AI Coach — Windows 屏幕采集与视觉理解"
if os.name == "nt":
    try:
        ctypes.WinDLL("user32", use_last_error=True).SetProcessDPIAware()
    except Exception:
        pass
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

class _GraphicsWindowCapture:
    """通过 Windows Graphics Capture 按窗口句柄采集硬件加速窗口。"""

    def __init__(self, handle):
        if WindowsCapture is None:
            raise RuntimeError(f"Windows 图形捕获组件未加载：{WINDOWS_CAPTURE_IMPORT_ERROR}")
        self._condition = threading.Condition()
        self._latest_image = None
        self._error = None
        self._closed = False
        self._capture = WindowsCapture(
            window_hwnd=int(handle),
            cursor_capture=False,
            draw_border=False,
            minimum_update_interval=250,
        )

        def on_frame_arrived(frame, capture_control):
            try:
                pixels = np.asarray(frame.frame_buffer).copy()
                if pixels.ndim != 3 or pixels.shape[2] != 4:
                    raise RuntimeError(f"窗口图形捕获返回异常画面尺寸：{pixels.shape}")
                rgb = cv2.cvtColor(pixels, cv2.COLOR_BGRA2RGB)
                image = Image.fromarray(rgb, "RGB")
                with self._condition:
                    self._latest_image = image
                    self._condition.notify_all()
            except Exception as exc:
                with self._condition:
                    self._error = exc
                    self._condition.notify_all()

        def on_closed():
            with self._condition:
                self._closed = True
                self._latest_image = None
                self._condition.notify_all()

        self._capture.event(on_frame_arrived)
        self._capture.event(on_closed)
        self._control = self._capture.start_free_threaded()

    def get_image(self, timeout=3.0):
        with self._condition:
            ready = self._condition.wait_for(
                lambda: self._latest_image is not None or self._error is not None or self._closed,
                timeout=max(0.0, float(timeout)),
            )
            if not ready:
                return None
            if self._error is not None:
                raise RuntimeError(f"Windows 图形捕获失败：{self._error}") from self._error
            if self._closed or self._latest_image is None:
                raise RuntimeError("目标牌桌窗口已关闭，Windows 图形捕获结束。")
            return self._latest_image.copy()

    def close(self):
        control = getattr(self, "_control", None)
        if control is not None:
            try:
                control.stop()
                control.wait()
            except Exception:
                pass

def _is_blank_capture(image):
    """检测 PrintWindow 对硬件加速窗口返回的纯黑/近乎全黑画面。"""
    pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if pixels.ndim != 3 or pixels.shape[0] == 0 or pixels.shape[1] == 0:
        return True
    luminance = (
        pixels[:, :, 0].astype(np.float32) * 0.299
        + pixels[:, :, 1].astype(np.float32) * 0.587
        + pixels[:, :, 2].astype(np.float32) * 0.114
    )
    return float(np.percentile(luminance, 99)) < 8.0 and float(np.mean(luminance > 16.0)) < 0.005

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

    # 部分模拟器使用硬件加速；PrintWindow 可能返回成功但画面全黑。
    # 只有确认窗口图像不是空白后才采信，否则回退到桌面实际截图。
    rendered = None
    try:
        rendered = _capture_printwindow(window.handle, width, height)
    except Exception:
        pass
    if rendered is not None and not _is_blank_capture(rendered):
        return rendered

    try:
        desktop = ImageGrab.grab(bbox=box, all_screens=True).convert("RGB")
    except Exception as exc:
        if rendered is not None and _is_blank_capture(rendered):
            raise RuntimeError("窗口采集返回黑屏，桌面区域截图也失败；请确认模拟器窗口可见后重试。") from exc
        raise
    if not _is_blank_capture(desktop):
        return desktop

    raise RuntimeError(
        "采集到的牌桌画面是全黑/空白：窗口绘制接口与屏幕区域截图都没有拿到画面。"
        "请将模拟器窗口恢复并置于桌面前台，确保未被遮挡，然后重新开始识别。"
    )
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

def find_unclassified_card_regions(image):
    """定位可疑牌区，但不从 OCR 字符推断牌值。"""
    boxes = find_card_regions(image)
    return boxes, [("??", 0.0) for _ in boxes]

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

def make_state(window, image, lines, card_boxes, card_values, model_available=False):
    frame_w, frame_h = image.size
    cards = []
    for box, values in zip(card_boxes, card_values):
        identity, confidence = values[0], values[1]
        margin = values[2] if len(values) > 2 else 0.0
        zone, cx, cy = classify_zone(box, image.size)
        source = MODEL_SOURCE if model_available and identity != "??" else "轮廓候选；尚无牌面视觉模型"
        cards.append({
            "candidate": identity,
            "zone": zone,
            "confidence": round(float(confidence), 3),
            "margin": round(float(margin), 3),
            "box": box,
            "center_normalized": [round(cx, 3), round(cy, 3)],
            "stable": False,
            "source": source,
        })
    board = [card["candidate"] for card in cards if card["zone"] == "公共牌候选" and "?" not in card["candidate"]]
    hero = [card["candidate"] for card in cards if card["zone"] == "hero候选" and "?" not in card["candidate"]]
    street = {3: "翻牌", 4: "转牌", 5: "河牌"}.get(len(board), "待识别")
    return {
        "采集时间": datetime.now().astimezone().isoformat(timespec="seconds"),
        "目标窗口": {"标题": window.title, "窗口类名": window.class_name},
        "画面": {"宽度": frame_w, "高度": frame_h},
        "视觉模型状态": "已加载" if model_available else "缺失",
        "牌局候选": {
            "阶段": {"值": street, "状态": "根据公共牌区域候选数量推断" if street != "待识别" else "证据不足"},
            "公共牌": {"候选": board, "状态": "视觉候选，需结合目标模拟器验证"},
            "玩家手牌": {"候选": hero, "状态": "视觉候选，需结合目标模拟器验证"},
            "牌面区域": cards,
            "底池与筹码": extract_money_candidates(lines),
            "行动文字": extract_action_candidates(lines),
        },
        "画面文字": lines,
        "说明": "只接受连续稳定且来源可信的完整牌面；本地模型或识别信息不足时不生成行动建议。",
    }


def build_coach_advice(state, gate_state=None):
    table = state.get("牌局候选", {})
    cards = table.get("牌面区域", [])
    model_status = state.get("视觉模型状态")
    if model_status is None:
        relevant = [card for card in cards if card.get("zone") in ("hero候选", "公共牌候选")]
        model_status = "已加载" if relevant and all(card.get("source") == MODEL_SOURCE for card in relevant) else "缺失"
    if model_status != "已加载":
        return {"状态": "牌面视觉模型缺失", "牌型": "—", "对随机手牌胜率": "—", "底池赔率": "—",
                "建议": "没有通过许可与模型契约检查的本地牌面模型；暂不生成行动建议。"}
    inspection = validate_analysis_gate(cards, gate_state)
    if not inspection.get("valid"):
        status = inspection.get("status", "等待识别")
        reason = inspection.get("reason") or "当前牌面未通过安全校验。"
        return {"状态": status, "牌型": "—", "对随机手牌胜率": "—", "底池赔率": "—",
                "建议": reason + " 暂不生成行动建议。"}
    hero_rows = inspection["hero"]
    board_rows = inspection["board"]
    hole = [card["candidate"] for card in hero_rows]
    board = [card["candidate"] for card in board_rows]
    try:
        equity_result = estimate_equity(hole, board, iterations=400)
        equity = equity_result["equity"]
        if len(board) >= 3:
            hand_name = evaluate_hand(hole + board)[2]
        else:
            hand_name = "翻牌前"
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
        "分析依据": "只接受连续稳定且来自牌面视觉模型的完整点数与花色；对手模型为随机两张手牌，不等于真实对手范围。",
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

    def _scan_loop(self, window):
        graphics_capture = None
        try:
            engine = RapidOCR()
            model_error = None
            try:
                card_model = load_card_model(required=False)
            except CardVisionError as exc:
                card_model = None
                model_error = str(exc)
            table_gate = TableStateGate(required_frames=3)
            try:
                graphics_capture = _GraphicsWindowCapture(window.handle)
            except Exception as exc:
                self._offer(("status", f"目标窗口图形捕获不可用，将尝试兼容采集：{exc}"))
            if card_model is None:
                detail = f"；模型拒绝原因：{model_error}" if model_error else ""
                self._offer(("status", "本地牌面模型不可用，暂只读取金额与操作文字，等待合规模型。" + detail))
            else:
                self._offer(("status", "本地 CPU 牌面模型已加载；正在采集并等待整手牌连续三帧一致。"))
            while not self.stop_event.is_set():
                started = time.monotonic()
                frame = None
                capture_method = None
                if graphics_capture is not None:
                    try:
                        frame = graphics_capture.get_image(timeout=2.5)
                        if frame is not None and _is_blank_capture(frame):
                            graphics_capture.close()
                            graphics_capture = None
                            frame = None
                            self._offer(("status", "目标窗口图形捕获返回黑屏/空白，切换到窗口绘制 / 桌面区域回退采集。"))
                        elif frame is not None:
                            capture_method = "Windows 图形捕获（目标窗口）"
                    except Exception as exc:
                        graphics_capture.close()
                        graphics_capture = None
                        self._offer(("status", f"目标窗口图形捕获中断，改用兼容采集：{exc}"))
                if frame is None:
                    frame = capture_window(window)
                    capture_method = "窗口绘制 / 桌面区域回退"
                lines = ocr_lines(engine, frame)
                boxes = find_card_regions(frame)
                if card_model is not None:
                    values = classify_card_regions(card_model, frame, boxes)
                else:
                    values = [("??", 0.0, 0.0) for _ in boxes]
                state = make_state(window, frame, lines, boxes, values, model_available=card_model is not None)
                state["采集方式"] = capture_method
                gate_state = table_gate.update(state["牌局候选"]["牌面区域"])
                state["牌局候选"]["帧间校验"] = gate_state
                state["教练分析"] = build_coach_advice(state, gate_state)
                annotated = annotate_image(frame, state)
                self._offer(("frame", annotated, state))
                elapsed = time.monotonic() - started
                self.stop_event.wait(max(0.2, 1.0 - elapsed))
            self._offer(("stopped",))
        except Exception as exc:
            self._offer(("error", str(exc)))
        finally:
            if graphics_capture is not None:
                graphics_capture.close()

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
                    source = self.last_state.get("采集方式", "采集方式未知")
                    self.status.config(text=f"实时识别中：{source}；发现 {cards} 个牌面区域、{count} 条文字。识别结果是候选值，尚未针对真实模拟器校准。")
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
    require_model = "--require-model" in sys.argv
    packaged = bool(getattr(sys, "_MEIPASS", None))
    source_model = Path("models/playing_cards.onnx").is_file()
    if require_model or packaged or source_model:
        load_card_model(required=True)
    if WindowsCapture is None:
        raise RuntimeError(f"Windows 图形捕获组件加载失败：{WINDOWS_CAPTURE_IMPORT_ERROR}")
    blank = Image.new("RGB", (96, 72), (0, 0, 0))
    table = Image.new("RGB", (96, 72), (18, 54, 38))
    ImageDraw.Draw(table).rounded_rectangle((30, 18, 66, 64), radius=4, fill=(238, 238, 230))
    if not _is_blank_capture(blank) or _is_blank_capture(table):
        raise RuntimeError("窗口空白画面检测自检失败")
    engine_self_check()
    unsafe_state = {
        "牌局候选": {
            "牌面区域": [
                {"candidate": "9♣", "zone": "hero候选", "stable": True, "confidence": 0.95, "source": "OCR"},
                {"candidate": "3♣", "zone": "hero候选", "stable": True, "confidence": 0.93, "source": "OCR"},
            ],
            "底池与筹码": [],
        }
    }
    if build_coach_advice(unsafe_state).get("状态") != "牌面视觉模型缺失":
        raise RuntimeError("OCR 牌值被误用于生成行动建议")

    duplicate_cards = [
        {"candidate": "9♣", "zone": "hero候选", "stable": True, "confidence": 0.95, "margin": 0.8, "source": MODEL_SOURCE, "center_normalized": [0.46, 0.76]},
        {"candidate": "9♣", "zone": "hero候选", "stable": True, "confidence": 0.93, "margin": 0.7, "source": MODEL_SOURCE, "center_normalized": [0.52, 0.76]},
    ]
    if inspect_table_state(duplicate_cards).get("status") != "牌面冲突":
        raise RuntimeError("重复牌未阻止行动建议")

    stable_cards = [
        {"candidate": "9♣", "zone": "hero候选", "confidence": 0.95, "margin": 0.8, "source": MODEL_SOURCE, "center_normalized": [0.46, 0.76]},
        {"candidate": "3♣", "zone": "hero候选", "confidence": 0.93, "margin": 0.7, "source": MODEL_SOURCE, "center_normalized": [0.52, 0.76]},
        {"candidate": "3♥", "zone": "公共牌候选", "confidence": 0.94, "margin": 0.8, "source": MODEL_SOURCE, "center_normalized": [0.41, 0.44]},
        {"candidate": "2♦", "zone": "公共牌候选", "confidence": 0.93, "margin": 0.7, "source": MODEL_SOURCE, "center_normalized": [0.50, 0.44]},
        {"candidate": "7♣", "zone": "公共牌候选", "confidence": 0.92, "margin": 0.7, "source": MODEL_SOURCE, "center_normalized": [0.59, 0.44]},
    ]
    gate = TableStateGate(required_frames=3)
    gate_state = None
    for _ in range(3):
        gate_state = gate.update([dict(card) for card in stable_cards])
    if not gate_state.get("通过") or not validate_analysis_gate(stable_cards, gate_state).get("valid"):
        raise RuntimeError("整手牌连续帧门控自检失败")
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
