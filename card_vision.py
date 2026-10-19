"""本地 ONNX 扑克牌分类适配层。

契约 pokerai-card-classifier-v1：
- 输入：float32 RGB NCHW，[1, 3, 192, 128]，像素范围 [0, 1]。
- 输出：float32 [1, 52]，按 CARD_LABELS 排列的 softmax 概率。
- 元数据记录分类顺序、CC0 权重许可和固定训练素材来源。
"""
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

RANKS = ("2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A")
SUITS = ("C", "D", "H", "S")
CARD_LABELS = tuple(rank + suit for rank in RANKS for suit in SUITS)
SUIT_SYMBOLS = {"C": "♣", "D": "♦", "H": "♥", "S": "♠"}
MODEL_SOURCE = "牌面视觉模型"
MODEL_SCHEMA = "pokerai-card-classifier-v1"
INPUT_HEIGHT = 192
INPUT_WIDTH = 128
OUTPUT_CLASSES = 52
MIN_CONFIDENCE = 0.70
MIN_CLASS_MARGIN = 0.15
RECOGNIZED_LICENSES = {"MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "CC0-1.0", "Unlicense"}
CARD_CODE_PATTERN = __import__("re").compile(r"^(10|[2-9TJQKA])[♠♥♦♣]$")


class CardVisionError(RuntimeError):
    """模型文件、元数据或推理输出不符合本地适配契约。"""


def _model_path():
    candidates = []
    bundle_root = getattr(sys, "_MEIPASS", None)
    if bundle_root:
        candidates.append(Path(bundle_root) / "models" / "playing_cards.onnx")
    candidates.append(Path(__file__).resolve().parent / "models" / "playing_cards.onnx")
    return next((path for path in candidates if path.is_file()), candidates[0])


def validate_model_contract(metadata, input_shape, output_shape):
    """检查模型输入、52 类顺序、许可证和数据来源。"""
    if metadata.get("pokerai_schema") != MODEL_SCHEMA:
        raise CardVisionError("ONNX 模型缺少受支持的 pokerai_schema 元数据。")
    try:
        labels = metadata.get("class_labels", "")
        labels = json.loads(labels) if isinstance(labels, str) else labels
    except (TypeError, ValueError) as exc:
        raise CardVisionError("ONNX 模型的 class_labels 元数据不是有效 JSON。") from exc
    if tuple(labels or ()) != CARD_LABELS:
        raise CardVisionError("ONNX 模型的 52 类顺序与 Poker AI Coach 契约不一致。")
    if metadata.get("license") not in RECOGNIZED_LICENSES:
        raise CardVisionError("ONNX 模型没有可接受的明确 SPDX 分发许可。")
    for key in ("source_url", "license_url"):
        if not str(metadata.get(key, "")).startswith(("https://", "http://")):
            raise CardVisionError(f"ONNX 模型缺少可核实的 {key} 来源链接。")
    if len(input_shape) != 4:
        raise CardVisionError("模型输入必须是 [1, 3, 192, 128] NCHW 张量。")
    for actual, expected, label in zip(input_shape, (1, 3, INPUT_HEIGHT, INPUT_WIDTH),
                                       ("batch", "channels", "height", "width")):
        if isinstance(actual, int) and actual > 0 and actual != expected:
            raise CardVisionError(f"模型 {label} 维度应为 {expected}，实际为 {actual}。")
    if len(output_shape) != 2 or (isinstance(output_shape[-1], int) and output_shape[-1] != OUTPUT_CLASSES):
        raise CardVisionError("模型输出必须是 [1, 52] 分类概率张量。")
    if isinstance(output_shape[0], int) and output_shape[0] != 1:
        raise CardVisionError("模型输出 batch 维度必须为 1。")
    return True


def load_card_model(model_path=None, required=False):
    """加载本地 CPU ONNX 分类器；模型缺失时可安全等待。"""
    path = Path(model_path) if model_path else _model_path()
    if not path.is_file():
        if required:
            raise CardVisionError(f"找不到本地牌面模型：{path}")
        return None
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        if len(session.get_inputs()) != 1 or len(session.get_outputs()) != 1:
            raise CardVisionError("牌面分类器必须恰有一个输入和一个输出。")
        metadata = session.get_modelmeta().custom_metadata_map or {}
        validate_model_contract(
            metadata,
            session.get_inputs()[0].shape,
            session.get_outputs()[0].shape,
        )
        return session
    except CardVisionError:
        raise
    except Exception as exc:
        raise CardVisionError(f"无法加载本地牌面 ONNX 分类器：{exc}") from exc


def card_label_to_unicode(label):
    label = str(label).upper()
    if len(label) < 2 or label[-1] not in SUIT_SYMBOLS:
        raise CardVisionError(f"未知牌面类别：{label}")
    rank = label[:-1]
    if rank not in RANKS:
        raise CardVisionError(f"未知牌点类别：{label}")
    return rank + SUIT_SYMBOLS[label[-1]]


def classify_card_regions(session, image, boxes, confidence_threshold=MIN_CONFIDENCE,
                          margin_threshold=MIN_CLASS_MARGIN):
    """对轮廓定位出的单张牌框逐张分类；低置信度输出未知牌。"""
    if session is None:
        raise CardVisionError("牌面视觉模型未配置。")
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    values = []
    for box in boxes:
        if len(box) != 4:
            values.append(("??", 0.0, 0.0))
            continue
        x1, y1, x2, y2 = map(int, box)
        if x2 <= x1 or y2 <= y1:
            values.append(("??", 0.0, 0.0))
            continue
        crop = image.crop((x1, y1, x2, y2)).convert("RGB")
        crop = crop.resize((INPUT_WIDTH, INPUT_HEIGHT), Image.Resampling.BILINEAR)
        tensor = np.asarray(crop, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
        try:
            output = session.run([output_name], {input_name: tensor})[0]
        except Exception as exc:
            raise CardVisionError(f"本地牌面分类器推理失败：{exc}") from exc
        scores = np.asarray(output, dtype=np.float32)
        if scores.shape != (1, OUTPUT_CLASSES):
            raise CardVisionError(f"模型输出应为 [1, 52]，实际为 {scores.shape}。")
        scores = scores[0]
        if not np.isfinite(scores).all() or np.any(scores < 0) or np.any(scores > 1):
            raise CardVisionError("模型输出不是有限的分类概率。")
        if not math.isclose(float(scores.sum()), 1.0, rel_tol=0.05, abs_tol=0.05):
            raise CardVisionError("模型分类概率总和异常。")
        order = np.argsort(scores)
        class_index = int(order[-1])
        confidence = float(scores[class_index])
        margin = confidence - float(scores[int(order[-2])])
        identity = card_label_to_unicode(CARD_LABELS[class_index])
        if confidence < confidence_threshold or margin < margin_threshold:
            identity = "??"
        values.append((identity, confidence, margin))
    return values


def inspect_table_state(cards, confidence_threshold=MIN_CONFIDENCE, margin_threshold=MIN_CLASS_MARGIN):
    """检查牌点来源、置信度、张数、牌局阶段与重复牌。"""
    relevant = [card for card in cards if card.get("zone") in ("hero候选", "公共牌候选")]
    if not relevant:
        return {"valid": False, "status": "等待识别", "reason": "还没有找到手牌或公共牌候选。"}
    if any(card.get("source") != MODEL_SOURCE for card in relevant):
        return {"valid": False, "status": "牌面视觉模型缺失", "reason": "牌值没有来自本地视觉模型。"}
    hero = sorted(
        [card for card in relevant if card.get("zone") == "hero候选"],
        key=lambda card: card.get("center_normalized", [0, 0])[0],
    )
    board = sorted(
        [card for card in relevant if card.get("zone") == "公共牌候选"],
        key=lambda card: card.get("center_normalized", [0, 0])[0],
    )
    if len(hero) != 2:
        return {"valid": False, "status": "等待识别", "reason": "自己的手牌必须恰好两张。"}
    if len(board) not in (0, 3, 4, 5):
        return {"valid": False, "status": "阶段异常", "reason": "公共牌只能是 0、3、4 或 5 张。"}
    selected = hero + board
    for card in selected:
        candidate = str(card.get("candidate", ""))
        if not CARD_CODE_PATTERN.fullmatch(candidate):
            return {"valid": False, "status": "识别置信度不足", "reason": "存在未知或不完整牌值。"}
    codes = [card["candidate"] for card in selected]
    if len(set(codes)) != len(codes):
        return {"valid": False, "status": "牌面冲突", "reason": "同一副牌中检测到重复牌。"}
    for card in selected:
        try:
            confidence = float(card.get("confidence", 0.0))
            margin = float(card.get("margin", 0.0))
        except (TypeError, ValueError):
            confidence, margin = 0.0, 0.0
        if not math.isfinite(confidence) or confidence < confidence_threshold:
            return {"valid": False, "status": "识别置信度不足", "reason": "至少一张牌的置信度不足。"}
        if not math.isfinite(margin) or margin < margin_threshold:
            return {"valid": False, "status": "识别置信度不足", "reason": "至少一张牌与次高类别的差距不足。"}
    street = {0: "翻牌前", 3: "翻牌", 4: "转牌", 5: "河牌"}[len(board)]
    signature = (tuple(card["candidate"] for card in hero),
                 tuple(card["candidate"] for card in board))
    return {"valid": True, "status": "等待牌面稳定", "reason": "", "street": street,
            "hero": hero, "board": board, "signature": signature}


def _signature_as_lists(signature):
    if not isinstance(signature, (tuple, list)) or len(signature) != 2:
        return None
    if not all(isinstance(part, (tuple, list)) for part in signature):
        return None
    return [list(signature[0]), list(signature[1])]


class TableStateGate:
    """只有同一副完整、合法的牌连续出现指定帧数后才允许分析。"""

    def __init__(self, required_frames=3):
        self.required_frames = max(2, int(required_frames))
        self.reset()

    def reset(self):
        self._signature = None
        self._consecutive_frames = 0

    def update(self, cards):
        inspection = inspect_table_state(cards)
        for card in cards:
            card["stable"] = False
            card["状态"] = "单帧或未稳定候选"
        if not inspection["valid"]:
            self.reset()
            return {
                "通过": False, "帧数": 0, "阶段": inspection.get("street", "待识别"),
                "原因": inspection["reason"], "状态": inspection["status"], "签名": None,
            }
        signature = inspection["signature"]
        if signature == self._signature:
            self._consecutive_frames += 1
        else:
            self._signature = signature
            self._consecutive_frames = 1
        ready = self._consecutive_frames >= self.required_frames
        for card in inspection["hero"] + inspection["board"]:
            card["stable"] = ready
            card["状态"] = "连续帧稳定" if ready else "单帧或未稳定候选"
        return {
            "通过": ready,
            "帧数": self._consecutive_frames,
            "阶段": inspection["street"],
            "原因": "" if ready else "等待同一组牌面连续通过帧间校验。",
            "状态": "通过" if ready else "等待牌面稳定",
            "签名": _signature_as_lists(signature),
        }


def validate_analysis_gate(cards, gate_state):
    inspection = inspect_table_state(cards)
    if not inspection["valid"]:
        return inspection
    if not isinstance(gate_state, dict) or gate_state.get("通过") is not True:
        return {"valid": False, "status": "等待牌面稳定", "reason": "尚未通过连续帧校验。"}
    if _signature_as_lists(inspection["signature"]) != gate_state.get("签名"):
        return {"valid": False, "status": "等待牌面稳定", "reason": "当前牌面与已校验帧不一致。"}
    return inspection
