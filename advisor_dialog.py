"""Manual-confirmation bridge from visual candidates to the poker advisor."""
import tkinter as tk
from tkinter import ttk, messagebox

from poker_core import GameState, SituationAnalyzer, parse_cards


RANKS = set("23456789TJQKA")
SUITS = set("cdhs")


def _card_text(card):
    rank = str(card.get("rank", "")).upper()
    suit = str(card.get("suit", "")).lower()
    if rank in RANKS and suit in SUITS:
        return rank + suit
    return None


def _candidate_inputs(recognition_result):
    result = recognition_result or {}
    hero_rows = (result.get("hero_hand_candidates") or {}).get("cards") or []
    hero = [_card_text(card) for card in hero_rows]
    hero = [card for card in hero if card]

    board_rows = result.get("cards") or []
    board = [_card_text(card) for card in board_rows]
    board = [card for card in board if card]

    lines = ["视觉识别结果仅作候选，请对照左侧画面核对并修正后再分析。"]
    if hero_rows:
        lines.append("手牌候选状态：" + "；".join(
            f"{row.get('status', '待确认')}（置信度 {float(row.get('confidence', 0) or 0):.2f}）"
            for row in hero_rows
        ))
    if board_rows:
        lines.append("公共牌候选状态：" + "；".join(
            f"第{row.get('index', '?')}张 {row.get('status', '待确认')}"
            for row in board_rows
        ))
    if not hero and not board:
        lines.append("当前没有可用的牌面候选，请手动输入已核实的牌。")
    return " ".join(hero), " ".join(board), "\n".join(lines)


class AdvisorDialog:
    def __init__(self, parent, recognition_result=None):
        self.window = tk.Toplevel(parent)
        self.window.title("Poker AI Coach - 局面分析")
        self.window.geometry("900x700")
        self.window.minsize(760, 600)
        self.analyzer = SituationAnalyzer()
        hero_default, board_default, note = _candidate_inputs(recognition_result)

        outer = ttk.Frame(self.window, padding=12)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="先核对局面，再查看训练分析", font=("Microsoft YaHei UI", 14, "bold")).pack(anchor="w")
        ttk.Label(outer, text=note, wraplength=850, justify="left").pack(anchor="w", pady=(5, 10))

        fields = ttk.Frame(outer)
        fields.pack(fill="x")
        self.values = {}
        entries = [
            ("Hero 手牌", hero_default, 16),
            ("公共牌", board_default, 22),
            ("底池 BB", "", 12),
            ("跟注 BB", "", 12),
            ("有效筹码 BB", "", 12),
        ]
        for index, (label, default, width) in enumerate(entries):
            row, col = divmod(index, 3)
            ttk.Label(fields, text=label).grid(row=row * 2, column=col, sticky="w", padx=5, pady=(4, 0))
            value = tk.StringVar(value=default)
            self.values[label] = value
            ttk.Entry(fields, textvariable=value, width=width).grid(row=row * 2 + 1, column=col, sticky="ew", padx=5, pady=(0, 5))
        for col in range(3):
            fields.columnconfigure(col, weight=1)

        selectors = ttk.Frame(outer)
        selectors.pack(fill="x", pady=(4, 8))
        ttk.Label(selectors, text="对手位置").pack(side="left")
        self.position = tk.StringVar(value="BTN")
        ttk.Combobox(selectors, textvariable=self.position, values=["UTG", "HJ", "CO", "BTN", "SB", "BB"], state="readonly", width=10).pack(side="left", padx=(5, 18))
        ttk.Label(selectors, text="对手行动").pack(side="left")
        self.action = tk.StringVar(value="无")
        ttk.Combobox(selectors, textvariable=self.action, values=["无", "BET", "RAISE", "3BET", "ALL-IN"], state="readonly", width=10).pack(side="left", padx=5)
        ttk.Button(selectors, text="分析已核对局面", command=self.analyze).pack(side="right")

        self.output = tk.Text(outer, wrap="word", font=("Microsoft YaHei UI", 10))
        self.output.pack(fill="both", expand=True)
        self.output.insert("1.0", "金额单位均为 BB。至少输入 3 张公共牌才能计算。\n识别候选需要人工核对；建议基于简化训练范围和近似 EV，不等同于 GTO 求解器。")

    def analyze(self):
        try:
            hero = parse_cards(self.values["Hero 手牌"].get())
            board = parse_cards(self.values["公共牌"].get())
            pot = float(self.values["底池 BB"].get())
            call = float(self.values["跟注 BB"].get())
            effective_stack = float(self.values["有效筹码 BB"].get())
            state = GameState(hero, board, pot, call, effective_stack, self.position.get(), self.action.get())
            result = self.analyzer.analyze(state)
            best = max(result.actions, key=lambda row: row.ev)
            action_lines = "\n".join(f"  {row.action}：{row.ev:+.2f} BB" for row in result.actions)
            outs = "、".join(result.outs_cards) if result.outs_cards else "无明显改善牌"
            assumptions = "\n".join("  • " + item for item in result.assumptions)
            report = (
                f"【当前牌力】\n{result.hand_strength}\n\n"
                f"【对手范围】\n位置：{state.villain_position}；行动：{state.villain_action}\n"
                f"剩余估算组合：{result.range_count}\n\n"
                f"【胜率与赔率】\nHero vs Range：{result.equity * 100:.2f}%\n"
                f"底池赔率：{result.pot_odds * 100:.2f}%\n"
                f"跟注盈亏平衡胜率：{result.breakeven_equity * 100:.2f}%\n"
                f"SPR：{result.spr:.2f}\n跟注 EV：{result.call_ev:+.2f} BB\n\n"
                f"【补牌】\n估算 Outs：{result.outs}\n补牌候选：{outs}\n\n"
                f"【牌面结构】\n{result.texture}\n\n"
                f"【简化 EV 对比】\n{action_lines}\n\n"
                f"最高估算 EV：{best.action}（{best.ev:+.2f} BB）\n\n"
                f"【模型假设】\n{assumptions}\n\n"
                "此结果用于训练参考。牌面候选、金额、对手范围与建议都需要结合实际牌局核对。"
            )
            self.output.delete("1.0", "end")
            self.output.insert("1.0", report)
        except Exception as exc:
            messagebox.showerror("无法分析", str(exc), parent=self.window)


def open_advisor(parent, recognition_result=None):
    return AdvisorDialog(parent, recognition_result)
