"""本地 ONNX 扑克牌检测适配层。

模型契约 pokerai-card-detector-v1：
- 输入：float32 RGB NCHW，形状 [1, 3, 640, 640]，像素范围 [0, 1]，等比例缩放并以 114 填充。
- 输出：float32 [1, N, 57] 或 [N, 57]；每行是输入画布像素坐标 x1,y1,x2,y2、objectness、52 个 class probability。
- ONNX 自定义元数据必须包含 schema、class_labels、license、source_url、license_url。
"""
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

RANKS = ("2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A")
SUITS = ("C", "D", "H", "S")
CARD_LABELS = tuple(rank + suit for rank in RANKS for suit in SUITS)
SUIT_SYMBOLS = {"C": "♣", "D": "♦", "H": "♥", "S": "♠"}
MODEL_SOURCE = "牌面视觉模型"
MODEL_SCHEMA = "pokerai-card-detector-v1"
INPUT_SIZE = 640
OUTPUT_COLUMNS = 57
MIN_CONFIDENCE = 0.70
MIN_CLASS_MARGIN = 0.15
NMS_IOU_THRESHOLD = 0.35
RECOGNIZED_LICENSES = {
    "MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "CC0-1.0", "Unlicense"
}
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
    """验证权重来源记录、固定类别顺序和 ONNX 张量形状。"""
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
        raise CardVisionError("模型输入必须是 [1, 3, 640, 640] NCHW 张量。")
    for actual, expected, label in zip(input_shape, (1, 3, INPUT_SIZE, INPUT_SIZE), ("batch", "channels", "height", "width")):
        if isinstance(actual, int) and actual > 0 and actual != expected:
            raise CardVisionError(f"模型 {label} 维度应为 {expected}，实际为 {actual}。")
    if len(output_shape) not in (2, 3):
        raise CardVisionError("模型输出必须是 [N, 57] 或 [1, N, 57]。")
    if len(output_shape) == 3 and isinstance(output_shape[0], int) and output_shape[0] != 1:
        raise CardVisionError("模型输出 batch 维度必须为 1。")
    if isinstance(output_shape[-1], int) and output_shape[-1] != OUTPUT_COLUMNS:
        raise CardVisionError(f"模型每行必须包含 {OUTPUT_COLUMNS} 个值。")
    return True


def load_card_model(model_path=None, required=False):
    """加载本地 CPU ONNX 模型；模型缺失时可返回 None，让应用安全等待。"""
    path = Path(model_path) if model_path else _model_path()
    if not path.is_file():
        if required:
            raise CardVisionError(f"找不到本地牌面模型：{path}")
        return None
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        if len(session.get_inputs()) != 1 or len(session.get_outputs()) != 1:
            raise CardVisionError("牌面模型必须恰有一个输入和一个输出。")
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
        raise CardVisionError(f"无法加载本地牌面 ONNX 模型：{exc}") from exc


def card_label_to_unicode(label):
    label = str(label).upper()
    if len(label) < 2 or label[-1] not in SUIT_SYMBOLS:
        raise CardVisionError(f"未知牌面类别：{label}")
    rank = label[:-1]
    if rank not in RANKS:
        raise CardVisionError(f"未知牌点类别：{label}")
    return rank + SUIT_SYMBOLS[label[-1]]


def intersection_over_union(a, b):
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    if right <= left or bottom <= top:
        return 0.0
    overlap = (right - left) * (bottom - top)
    area_a = max(1.0, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1.0, (b[2] - b[0]) * (b[3] - b[1]))
    return overlap / (area_a + area_b - overlap)


def decode_detections(raw_output, image_size, scale, padding, confidence_threshold=MIN_CONFIDENCE,
                      margin_threshold=MIN_CLASS_MARGIN, nms_threshold=NMS_IOU_THRESHOLD):
    """将固定版 ONNX 检测输出变成图像坐标牌框；分数不足的框直接丢弃。"""
    rows = np.asarray(raw_output, dtype=np.float32)
    if rows.ndim == 3:
        if rows.shape[0] != 1:
            raise CardVisionError("模型只支持单帧 batch=1 推理。")
        rows = rows[0]
    if rows.ndim != 2 or rows.shape[1] != OUTPUT_COLUMNS:
        raise CardVisionError(f"模型输出应为 [N, {OUTPUT_COLUMNS}]，实际为 {rows.shape}。")
    image_width, image_height = image_size
    pad_x, pad_y = padding
    candidates = []
    for row in rows:
        if not np.isfinite(row).all():
            continue
        x1, y1, x2, y2 = map(float, row[:4])
        objectness = float(np.clip(row[4], 0.0, 1.0))
        class_scores = np.clip(row[5:], 0.0, 1.0)
        if class_scores.size != len(CARD_LABELS):
            continue
        order = np.argsort(class_scores)
        class_index = int(order[-1])
        top_score = float(class_scores[class_index])
        second_score = float(class_scores[int(order[-2])])
        confidence = objectness * top_score
        margin = top_score - second_score
        if confidence < confidence_threshold or margin < margin_threshold:
            continue
        if x2 <= x1 or y2 <= y1:
            continue
        box = [
            float(np.clip((x1 - pad_x) / scale, 0, image_width)),
            float(np.clip((y1 - pad_y) / scale, 0, image_height)),
            float(np.clip((x2 - pad_x) / scale, 0, image_width)),
            float(np.clip((y2 - pad_y) / scale, 0, image_height)),
        ]
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        candidates.append({
            "box": [int(round(value)) for value in box],
            "candidate": card_label_to_unicode(CARD_LABELS[class_index]),
            "confidence": confidence,
            "margin": margin,
            "class_id": class_index,
            "source": MODEL_SOURCE,
        })

    candidates.sort(key=lambda item: item["confidence"], reverse=True)
    kept = []
    for candidate in candidates:
        if any(
            candidate["class_id"] == prior["class_id"]
            and intersection_over_union(candidate["box"], prior["box"]) >= nms_threshold
            for prior in kept
        ):
            continue
        kept.append(candidate)
    kept.sort(key=lambda item: ((item["box"][0] + item["box"][2]) / 2,
                                (item["box"][1] + item["box"][3]) / 2))
    boxes = [item["box"] for item in kept]
    values = [(item["candidate"], item["confidence"], item["margin"]) for item in kept]
    return boxes, values


def detect_cards(session, image):
    """在单张桌面帧上做等比例 letterbox 和本地 ONNX 推理。"""
    if session is None:
        raise CardVisionError("牌面视觉模型未配置。")
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    image_height, image_width = rgb.shape[:2]
    if image_width < 1 or image_height < 1:
        raise CardVisionError("不能识别空白图像。")
    scale = min(INPUT_SIZE / image_width, INPUT_SIZE / image_height)
    resized_width = max(1, int(round(image_width * scale)))
    resized_height = max(1, int(round(image_height * scale)))
    resized = cv2.resize(rgb, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    pad_x = (INPUT_SIZE - resized_width) // 2
    pad_y = (INPUT_SIZE - resized_height) // 2
    canvas = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.uint8)
    canvas[pad_y:pad_y + resized_height, pad_x:pad_x + resized_width] = resized
    tensor = canvas.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    try:
        output = session.run(
            [session.get_outputs()[0].name],
            {session.get_inputs()[0].name: tensor},
        )[0]
    except Exception as exc:
        raise CardVisionError(f"本地牌面模型推理失败：{exc}") from exc
    return decode_detections(
        output, (image_width, image_height), scale, (pad_x, pad_y)
    )


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
