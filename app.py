from __future__ import annotations
import csv, json, os, sys, time, threading, ctypes
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
from PIL import Image, ImageTk, ImageGrab
import vision_engine as ve
import amount_engine as ae
import action_engine as axe
import state_engine as se
from advisor_dialog import open_advisor

ROOT = Path(__file__).resolve().parent
# In a PyInstaller one-file build, __file__ points inside a temporary extraction
# directory. Store user exports outside that temporary directory so they persist.
if getattr(sys, "frozen", False):
    OUT = Path.home() / "Documents" / "PokerAI_Recognition_Output"
else:
    OUT = ROOT / "output"
OUT.mkdir(parents=True, exist_ok=True)


def _enum_windows():
    if os.name != "nt":
        return []
    from ctypes import wintypes
    user32 = ctypes.windll.user32
    callback_t = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
    windows = []
    def get_text(hwnd):
        n = user32.GetWindowTextLengthW(hwnd)
        buf = ctypes.create_unicode_buffer(n + 1)
        user32.GetWindowTextW(hwnd, buf, n + 1)
        return buf.value.strip()
    def callback(hwnd, _):
        if user32.IsWindowVisible(hwnd):
            title = get_text(hwnd)
            if title:
                clsbuf = ctypes.create_unicode_buffer(256)
                user32.GetClassNameW(hwnd, clsbuf, 256)
                rect = wintypes.RECT()
                if user32.GetWindowRect(hwnd, ctypes.byref(rect)):
                    w, h = rect.right - rect.left, rect.bottom - rect.top
                    if w >= 400 and h >= 300:
                        windows.append({"hwnd": int(hwnd), "title": title, "class": clsbuf.value,
                                        "rect": [rect.left, rect.top, rect.right, rect.bottom]})
        return True
    user32.EnumWindows(callback_t(callback), 0)
    windows.sort(key=lambda row: (0 if "poker" in row["title"].lower() else 1, row["title"].lower()))
    return windows


class App:
    def __init__(self, root):
        self.root = root
        root.title("Poker AI Coach - Windows")
        root.geometry("1280x860")
        root.minsize(1050, 700)
        self.capture_obj = None
        self.video_path = None
        self.video_fps = 0.0
        self.video_total = 0
        self.frame_index = 0
        self.current_frame = None
        self.current_result = None
        self.current_time = 0.0
        self.preview_tk = None
        self.win_rows = []
        self.monitoring = False
        self.temporal_tracker = ve.TemporalCardTracker()
        self.state_tracker = se.ContinuousStateTracker()
        self.last_capture = None

        top = ttk.Frame(root, padding=(10, 8)); top.pack(fill="x")
        ttk.Label(top, text="Poker AI Coach：牌桌识别与局面分析", font=("Microsoft YaHei UI", 16, "bold")).pack(side="left")
        ttk.Label(top, text="识别结果为候选，核对后再分析", foreground="#376b3b").pack(side="left", padx=18)

        actions = ttk.Frame(root, padding=(10, 0, 10, 8)); actions.pack(fill="x")
        ttk.Button(actions, text="打开视频", command=self.open_video).pack(side="left", padx=3)
        ttk.Button(actions, text="上一秒", command=lambda: self.jump_seconds(-1)).pack(side="left", padx=3)
        ttk.Button(actions, text="下一秒", command=lambda: self.jump_seconds(1)).pack(side="left", padx=3)
        ttk.Button(actions, text="分析当前帧", command=self.analyze_current).pack(side="left", padx=3)
        ttk.Button(actions, text="分析已核对局面", command=self.open_advisor).pack(side="left", padx=3)
        ttk.Button(actions, text="检查 OCR 运行环境", command=self.check_ocr_runtime).pack(side="left", padx=3)
        ttk.Button(actions, text="整段视频每3秒抽样", command=self.sample_video).pack(side="left", padx=3)
        ttk.Button(actions, text="连续状态扫描（每0.5秒）", command=self.scan_continuous_state).pack(side="left", padx=3)
        ttk.Button(actions, text="保存当前结果", command=self.save_current).pack(side="left", padx=3)
        ttk.Button(actions, text="刷新模拟器窗口", command=self.refresh_windows).pack(side="right", padx=3)
        ttk.Button(actions, text="抓取选中窗口", command=self.capture_selected_window).pack(side="right", padx=3)

        winbar = ttk.Frame(root, padding=(10, 0, 10, 8)); winbar.pack(fill="x")
        ttk.Label(winbar, text="实时窗口：").pack(side="left")
        self.win_combo = ttk.Combobox(winbar, state="readonly", width=75)
        self.win_combo.pack(side="left", padx=5, fill="x", expand=True)
        self.window_status = ttk.Label(winbar, text="当前为离线视频识别模式")
        self.window_status.pack(side="right", padx=5)

        main = ttk.Panedwindow(root, orient="horizontal")
        main.pack(fill="both", expand=True, padx=10, pady=4)
        left = ttk.Frame(main); right = ttk.Frame(main)
        main.add(left, weight=3); main.add(right, weight=2)

        self.preview = ttk.Label(left, text="打开一段扑克模拟器视频，或抓取模拟器窗口", anchor="center")
        self.preview.pack(fill="both", expand=True)
        self.seek_label = ttk.Label(left, text="未载入视频", anchor="w")
        self.seek_label.pack(fill="x", pady=3)
        self.seek = ttk.Scale(left, from_=0, to=100, orient="horizontal", command=self.on_seek)
        self.seek.pack(fill="x", pady=2)

        ttk.Label(right, text="识别候选结果", font=("Microsoft YaHei UI", 12, "bold")).pack(anchor="w", pady=(0, 4))
        cols = ("idx", "rank", "suit", "conf", "white", "status")
        self.tree = ttk.Treeview(right, columns=cols, show="headings", height=12)
        headers = {"idx":"序号", "rank":"点数", "suit":"花色", "conf":"置信度", "white":"白底比例", "status":"状态"}
        widths = {"idx":45,"rank":65,"suit":55,"conf":70,"white":80,"status":200}
        for col in cols:
            self.tree.heading(col, text=headers[col]); self.tree.column(col, width=widths[col], anchor="center")
        self.tree.pack(fill="x", pady=(0, 10))

        self.summary = tk.Text(right, wrap="word", height=18, font=("Consolas", 10))
        self.summary.pack(fill="both", expand=True)
        self.summary.insert("1.0", "当前识别公共牌、可见英雄手牌、底池与玩家筹码候选，并提供每0.5秒的离线连续追踪。牌街变化、底池/筹码数值变化和可见行动文字都只是事件候选；不能仅凭金额变化断言跟注/加注/弃牌。\n动作 OCR 依赖 RapidOCR；低置信度、模型不可用或不同 OCR 结果冲突时保持‘待确认’。")

        self.status = ttk.Label(root, text="就绪：可进行牌面/金额识别和0.5秒离线状态扫描；位置映射与实际行动准确率仍待验证。", anchor="w")
        self.status.pack(fill="x", padx=10, pady=7)
        self.refresh_windows()

    def refresh_windows(self):
        self.win_rows = _enum_windows()
        values = [f"{i}: {x['title']} [{x['class']}] {x['rect'][2]-x['rect'][0]}×{x['rect'][3]-x['rect'][1]}"
                  for i, x in enumerate(self.win_rows)]
        self.win_combo["values"] = values
        if values:
            self.win_combo.current(0)
            self.window_status.config(text=f"找到 {len(values)} 个窗口")
        else:
            self.window_status.config(text="未发现 Windows 窗口 / 当前系统不支持窗口枚举")

    def check_ocr_runtime(self):
        """Show the installed OCR engine state and a real probe on the current frame."""
        try:
            diag = ve.get_ocr_runtime_status(initialize=True)
            action_probe = axe.analyze_visible_action_text(self.current_frame) if self.current_frame is not None else None
            report = {"runtime": diag, "current_frame_action_probe": action_probe}
            self.summary.delete("1.0", "end")
            self.summary.insert("1.0", json.dumps(report, ensure_ascii=False, indent=2))
            if diag.get("rapidocr_engine_initialized"):
                message = "RapidOCR 已初始化。右侧显示当前帧推理输出；仍需用 Windows 自检报告确认 EXE 内模型可加载。"
            elif diag.get("tesseract_executable"):
                message = "RapidOCR 不可用，但检测到了 Tesseract 备用程序。Windows 单文件版是否包含 OCR，仍以 --ocr-smoke 报告为准。"
            else:
                message = "OCR 运行依赖不可用。请按 README 重新构建 Windows 版本并运行 OCR 自检。"
            messagebox.showinfo("OCR 运行环境检查", message)
        except Exception as exc:
            messagebox.showerror("OCR 运行环境检查失败", f"{type(exc).__name__}: {exc}")

    def open_video(self):
        path = filedialog.askopenfilename(title="选择扑克模拟器录屏",
            filetypes=[("视频文件", "*.mp4 *.mkv *.avi *.mov"), ("所有文件", "*.*")])
        if not path: return
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            messagebox.showerror("打开失败", "无法打开该视频文件。")
            return
        if self.capture_obj is not None:
            self.capture_obj.release()
        self.capture_obj = cap
        self.video_path = path
        self.video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.video_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.frame_index = 0
        self.temporal_tracker = ve.TemporalCardTracker()
        self.read_frame(0)
        self.status.config(text=f"已打开：{Path(path).name} | {self.video_total} 帧 | {self.video_fps:.2f} FPS")

    def read_frame(self, frame_index: int):
        if self.capture_obj is None:
            return
        frame_index = max(0, min(int(frame_index), max(0, self.video_total - 1)))
        self.capture_obj.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = self.capture_obj.read()
        if not ok:
            return
        self.frame_index = frame_index
        self.current_frame = frame
        self.current_time = frame_index / max(self.video_fps, 1e-6)
        self.seek.configure(to=max(1, self.video_total - 1))
        self.seek.set(frame_index)
        self.seek_label.config(text=f"{Path(self.video_path).name} | {self.current_time:.2f} 秒 | 帧 {frame_index}/{self.video_total}")
        self.show_frame(frame)

    def jump_seconds(self, seconds):
        if self.capture_obj is None: return
        self.read_frame(self.frame_index + int(seconds * self.video_fps))

    def on_seek(self, value):
        if self.capture_obj is None: return
        try: target = int(float(value))
        except Exception: return
        if abs(target - self.frame_index) > max(2, int(self.video_fps * 0.6)):
            self.read_frame(target)

    def analyze_current(self):
        if self.current_frame is None:
            messagebox.showinfo("提示", "请先打开视频或抓取模拟器窗口。")
            return
        try:
            result = ve.analyze_frame(self.current_frame)
            amount_profile = result.get("hero_hand_candidates", {}).get("profile", ve.detect_table_profile(self.current_frame))
            result["amounts"] = ae.analyze_amounts(self.current_frame, amount_profile)
            result["visible_action_candidates"] = axe.analyze_visible_action_text(self.current_frame)
            result["source_video"] = self.video_path or "窗口截图"
            result["timestamp_s"] = round(self.current_time, 3)
            self.current_result = result
            self._show_result(result)
            annotated = ve.draw_observations(self.current_frame, result)
            annotated = ae.draw_amount_observations(annotated, result.get("amounts", {}))
            annotated = axe.draw_action_candidates(annotated, result.get("visible_action_candidates", {}))
            self.show_frame(annotated)
            self.status.config(text=f"分析完成：识别到 {result['board_card_count_candidate']} 张标准明牌候选；结果仍需跨帧/人工确认。")
        except Exception as exc:
            messagebox.showerror("识别失败", str(exc))

    def open_advisor(self):
        if self.current_result is None and self.current_frame is not None:
            self.analyze_current()
        open_advisor(self.root, self.current_result)

    def sample_video(self):
        if not self.video_path:
            messagebox.showinfo("提示", "请先打开一段视频。")
            return
        interval = 3.0
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            messagebox.showerror("错误", "视频无法重新打开。")
            return
        fps = cap.get(cv2.CAP_PROP_FPS) or self.video_fps or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or self.video_total)
        dur = total / max(fps, 1e-6)
        out_csv = OUT / (Path(self.video_path).stem + "_card_candidates.csv")
        out_json = OUT / (Path(self.video_path).stem + "_card_candidates.json")
        out_amount_csv = OUT / (Path(self.video_path).stem + "_amount_candidates.csv")
        out_amount_json = OUT / (Path(self.video_path).stem + "_amount_candidates.json")
        rows, amount_rows, frame_logs, amount_logs = [], [], [], []
        tracker = ve.TemporalCardTracker(window_size=5, min_repeats=2)
        self.status.config(text="正在按3秒间隔扫描视频…耗时取决于视频长度。")
        self.root.update_idletasks()
        timestamp = 0.0
        try:
            while timestamp <= dur:
                cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
                ok, frame = cap.read()
                if not ok: break
                self.status.config(text=f"正在分析 {timestamp:.0f}s / {dur:.0f}s…")
                self.root.update_idletasks()
                result = ve.analyze_frame(frame)
                amount_profile = result.get("hero_hand_candidates", {}).get("profile", ve.detect_table_profile(frame))
                result["amounts"] = ae.analyze_amounts(frame, amount_profile)
                result["visible_action_candidates"] = axe.analyze_visible_action_text(frame)
                result["timestamp_s"] = round(timestamp, 3)
                result["source_video"] = self.video_path
                stable = tracker.update(timestamp, result)
                frame_logs.append(stable)
                amount_logs.append({"timestamp_s": round(timestamp, 3), "source_video": self.video_path, **result.get("amounts", {})})
                amounts = result.get("amounts", {})
                if amounts.get("pot"):
                    amount_rows.append({"timestamp_s": round(timestamp,3), "field":"pot", "region":"底池", "bbox":json.dumps(amounts["pot"].get("bbox", [])), "candidate":amounts["pot"].get("candidate"), "numeric_value":amounts["pot"].get("numeric_value"), "confidence":amounts["pot"].get("confidence"), "status":amounts["pot"].get("status"), "raw_candidates":json.dumps(amounts["pot"].get("raw_candidates", []), ensure_ascii=False), "note":amounts["pot"].get("note")})
                for amount in amounts.get("stacks", []):
                    amount_rows.append({"timestamp_s":round(timestamp,3), "field":"stack", "region":amount.get("seat_region", amount.get("region", "待确认")), "bbox":json.dumps(amount.get("bbox", [])), "candidate":amount.get("candidate"), "numeric_value":amount.get("numeric_value"), "confidence":amount.get("confidence"), "status":amount.get("status"), "raw_candidates":json.dumps(amount.get("raw_candidates", []), ensure_ascii=False), "note":amount.get("note")})
                for card in stable.get("cards", []):
                    x, y, w, h = card["bbox"]
                    rows.append({
                        "timestamp_s": round(timestamp,3), "card_scope": "公共牌", "card_index": card["index"],
                        "x":x,"y":y,"width":w,"height":h,
                        "rank_candidate":card["rank"],"suit_candidate":card["suit_symbol"],
                        "rank_confidence":card["rank_confidence"],"suit_confidence":card["suit_confidence"],
                        "confidence":card["confidence"],"temporal_votes":card.get("temporal_votes",0),
                        "status":card["status"],"note":card["note"]
                    })
                for card in stable.get("hero_hand_candidates", {}).get("cards", []):
                    # The crop coordinates are used as a pointer to the hero-card region;
                    # this is not a full card bounding box.
                    roi = card.get("rank_roi", [0, 0, 0, 0])
                    rows.append({
                        "timestamp_s": round(timestamp,3), "card_scope": "英雄手牌", "card_index": card["index"],
                        "x":roi[0],"y":roi[1],"width":roi[2],"height":roi[3],
                        "rank_candidate":card["rank"],"suit_candidate":card["suit_symbol"],
                        "rank_confidence":card["rank_confidence"],"suit_confidence":card["suit_confidence"],
                        "confidence":card["confidence"],"temporal_votes":card.get("temporal_votes",0),
                        "status":card["status"],"note":card["note"]
                    })
                timestamp += interval
        except Exception as exc:
            messagebox.showerror("抽样失败", str(exc))
            return
        finally:
            cap.release()
        if rows:
            with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys())); writer.writeheader(); writer.writerows(rows)
        out_json.write_text(json.dumps(frame_logs, ensure_ascii=False, indent=2), encoding="utf-8")
        out_amount_json.write_text(json.dumps(amount_logs, ensure_ascii=False, indent=2), encoding="utf-8")
        if amount_rows:
            with out_amount_csv.open("w", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(amount_rows[0].keys())); writer.writeheader(); writer.writerows(amount_rows)
        self.status.config(text=f"抽样完成：{len(frame_logs)} 个时间点，牌面 {len(rows)} 条、金额 {len(amount_rows)} 条候选。结果保存到 {OUT}")
        messagebox.showinfo("抽样完成", f"完成 {len(frame_logs)} 个时间点。\n牌面候选：{len(rows)} 条\n金额候选：{len(amount_rows)} 条\n牌面 CSV：{out_csv}\n金额 CSV：{out_amount_csv}\n金额 JSON：{out_amount_json}\n注意：抽样日志不等于准确率报告；下注事件仍未完整追踪。")

    def scan_continuous_state(self):
        """High-frequency offline scan; amount OCR is invoked only after monitored ROIs change."""
        if not self.video_path:
            messagebox.showinfo("提示", "请先打开一段扑克模拟器视频，再运行连续状态扫描。")
            return
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            messagebox.showerror("错误", "视频无法重新打开。")
            return
        fps = cap.get(cv2.CAP_PROP_FPS) or self.video_fps or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or self.video_total)
        duration = total / max(fps, 1e-6)
        interval = 0.5
        tracker = se.ContinuousStateTracker()
        card_tracker = ve.TemporalCardTracker(window_size=5, min_repeats=2)
        previous_frame = None
        cached_amounts = None
        cached_profile = None
        trace = []
        self.status.config(text="正在连续状态扫描：每0.5秒采样；OCR区域变化时才重算金额。此模式用于离线验证，长视频需要等待。")
        self.root.update_idletasks()
        timestamp = 0.0
        try:
            while timestamp <= duration:
                cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
                ok, frame = cap.read()
                if not ok:
                    break
                result = ve.analyze_frame(frame)
                result = card_tracker.update(timestamp, result)
                profile = result.get("hero_hand_candidates", {}).get("profile", ve.detect_table_profile(frame))
                if (cached_amounts is None or profile != cached_profile or
                        ae.amount_regions_changed(previous_frame, frame, profile)):
                    cached_amounts = ae.analyze_amounts(frame, profile)
                    cached_profile = profile
                    amount_refreshed = True
                else:
                    amount_refreshed = False
                actions = axe.analyze_visible_action_text(frame)
                state = tracker.update(timestamp, result, cached_amounts, actions)
                trace.append({
                    "timestamp_s": round(timestamp, 3),
                    "source_video": self.video_path,
                    "board_card_count_candidate": result.get("board_card_count_candidate"),
                    "street_candidate": state.get("street_candidate"),
                    "amounts_refreshed": amount_refreshed,
                    "pot_candidate": (cached_amounts.get("pot") or {}).get("candidate") if cached_amounts else None,
                    "stack_candidate_count": len((cached_amounts or {}).get("stacks", [])),
                    "action_candidates": (actions or {}).get("actions", []),
                    "new_events": state.get("new_events", []),
                })
                previous_frame = frame.copy()
                if int(timestamp * 2) % 5 == 0:
                    self.status.config(text=f"连续扫描：{timestamp:.1f}s / {duration:.1f}s；累计事件 {state['event_count_total']}；金额本帧重算：{'是' if amount_refreshed else '否'}")
                    self.root.update_idletasks()
                timestamp += interval
        except Exception as exc:
            messagebox.showerror("连续扫描失败", str(exc))
            return
        finally:
            cap.release()
        stem = Path(self.video_path).stem
        trace_json = OUT / f"{stem}_continuous_state_trace.json"
        events_csv = OUT / f"{stem}_state_events.csv"
        trace_json.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
        events = tracker.export_events()
        standard_fields = ["timestamp_s", "event_type", "status", "field", "region", "action_candidate",
                           "previous_street", "current_street", "previous_value", "current_value", "delta",
                           "raw_text", "confidence", "note"]
        extra = sorted({k for event in events for k in event.keys()} - set(standard_fields))
        fields = standard_fields + extra
        with events_csv.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader(); writer.writerows(events)
        counts = {}
        for event in events:
            counts[event.get("event_type", "unknown")] = counts.get(event.get("event_type", "unknown"), 0) + 1
        self.status.config(text=f"连续扫描完成：{len(trace)}个时间点，事件候选{len(events)}条。结果保存到 {OUT}")
        messagebox.showinfo("连续扫描完成",
            f"扫描时间点：{len(trace)}（约每0.5秒）\n事件候选：{len(events)}\n事件类型：{json.dumps(counts, ensure_ascii=False)}\n\nJSON：{trace_json}\nCSV：{events_csv}\n\n注意：金额变化不等于特定行动；可见动作标签与新一手仍须人工确认。")

    def save_current(self):
        if self.current_result is None:
            self.analyze_current()
            if self.current_result is None: return
        suggested = f"poker_state_{int(self.current_time*1000)}ms.json"
        path = filedialog.asksaveasfilename(title="保存当前帧识别结果", initialfile=suggested,
                    defaultextension=".json", filetypes=[("JSON", "*.json")])
        if not path: return
        Path(path).write_text(json.dumps(self.current_result, ensure_ascii=False, indent=2), encoding="utf-8")
        self.status.config(text=f"已保存识别 JSON：{path}")

    def capture_selected_window(self):
        if os.name != "nt":
            messagebox.showinfo("不支持", "自动窗口抓取只在 Windows 运行；当前可以使用打开视频功能验证。")
            return
        idx = self.win_combo.current()
        if idx < 0 or idx >= len(self.win_rows):
            messagebox.showwarning("提示", "请先选中一个模拟器窗口。")
            return
        row = self.win_rows[idx]
        l, t, r, b = row["rect"]
        try:
            im = ImageGrab.grab(bbox=(l, t, r, b), all_screens=True)
            frame = cv2.cvtColor(np_from_pil(im), cv2.COLOR_RGB2BGR)
        except Exception as exc:
            messagebox.showerror("抓取失败", str(exc))
            return
        self.current_frame = frame
        self.current_time = 0.0
        self.video_path = None
        self.current_result = None
        self.show_frame(frame)
        self.seek_label.config(text=f"窗口截图：{row['title']} | {frame.shape[1]}×{frame.shape[0]}")
        self.status.config(text=f"已抓取窗口：{row['title']}；点击‘分析当前帧’开始识别。")

    def show_frame(self, frame):
        if frame is None: return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        max_w, max_h = 760, 620
        pil.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
        self.preview_tk = ImageTk.PhotoImage(pil)
        self.preview.configure(image=self.preview_tk, text="")

    def _show_result(self, result):
        for item in self.tree.get_children(): self.tree.delete(item)
        for c in result.get("cards", []):
            self.tree.insert("", "end", values=(c["index"], c["rank"], c["suit_symbol"],
                f"{c['confidence']:.2f}", f"{c['white_ratio']:.2f}", c["status"]))
        brief = {
            "时间(秒)":result.get("timestamp_s",0),
            "公共牌数量候选":result.get("board_card_count_candidate"),
            "牌街候选":result.get("street_candidate"),
            "整体状态":result.get("status"),
            "公共牌候选":result.get("cards"),
            "英雄手牌候选":result.get("hero_hand_candidates"),
            "底池金额候选":(result.get("amounts") or {}).get("pot"),
            "玩家筹码候选":(result.get("amounts") or {}).get("stacks"),
            "金额识别说明":(result.get("amounts") or {}).get("note"),
            "可见行动标签候选":result.get("visible_action_candidates", {}),
            "一致性校验":result.get("validation", []),
            "连续追踪说明": "可通过连续状态扫描生成牌街、底池/筹码变化和可见动作标签候选；所有事件仍需核对",
            "说明":result.get("note")
        }
        self.summary.delete("1.0", "end")
        self.summary.insert("1.0", json.dumps(brief, ensure_ascii=False, indent=2))


def np_from_pil(im):
    import numpy as np
    return np.array(im)


def main():
    if "--ocr-smoke" in sys.argv:
        from ocr_runtime_smoke import main as ocr_smoke_main
        sys.exit(ocr_smoke_main())
    root = tk.Tk()
    try:
        style = ttk.Style()
        if "vista" in style.theme_names(): style.theme_use("vista")
    except Exception:
        pass
    App(root)
    root.mainloop()

if __name__ == "__main__":
    main()
