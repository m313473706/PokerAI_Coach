#!/usr/bin/env python3
"""Replay labeled simulator screenshots against the contour locator and local classifier."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

from PIL import Image
from app import find_card_regions
from card_vision import CARD_LABELS, CardVisionError, classify_card_regions, load_card_model


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    intersection = (x2 - x1) * (y2 - y1)
    area_a = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1, (b[2] - b[0]) * (b[3] - b[1]))
    return intersection / float(area_a + area_b - intersection)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def label_code(value):
    value = str(value).strip().upper().replace(" ", "")
    for symbol, code in (("♣", "C"), ("♦", "D"), ("♥", "H"), ("♠", "S")):
        value = value.replace(symbol, code)
    if value not in CARD_LABELS:
        raise ValueError(f"未知牌值：{value}")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="JSON 文件：screenshots 列表及每张牌的 label、box")
    parser.add_argument("--model", default="models/playing_cards.onnx")
    parser.add_argument("--report", default="models/real_replay_report.json")
    parser.add_argument("--min-iou", type=float, default=0.50)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    screenshots = manifest.get("screenshots")
    if not isinstance(screenshots, list) or not screenshots:
        raise ValueError("manifest 的 screenshots 必须是非空数组。")
    model_path = Path(args.model).resolve()
    session = load_card_model(model_path, required=True)
    expected_classes, rows = set(), []
    total = correct = misses = extras = low_confidence = 0

    for shot in screenshots:
        image_name = shot["image"]
        image_path = (manifest_path.parent / image_name).resolve()
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        detections = find_card_regions(image)
        used = set()
        cards = shot.get("cards", [])
        has_boxes = [isinstance(card, dict) and "box" in card for card in cards]
        if any(has_boxes) and not all(has_boxes):
            raise ValueError(f"{image_name} 的 cards 不能混用带框与不带框格式。")

        if not all(has_boxes):
            # 简化格式：只写牌值，按定位器返回的从上到下、从左到右顺序对应。
            labels = [label_code(card if isinstance(card, str) else card["label"]) for card in cards]
            expected_classes.update(labels)
            total += len(labels)
            paired = min(len(labels), len(detections))
            misses += len(labels) - paired
            extras += len(detections) - paired
            for expected, detected_box in zip(labels, detections):
                prediction, confidence, margin = classify_card_regions(
                    session, image, [detected_box]
                )[0]
                predicted = prediction
                for symbol, code in (("♣", "C"), ("♦", "D"), ("♥", "H"), ("♠", "S")):
                    predicted = predicted.replace(symbol, code)
                is_correct = predicted == expected
                correct += int(is_correct)
                low_confidence += int(prediction == "??")
                rows.append({"image": image_name, "expected": expected, "predicted": predicted,
                             "confidence": confidence, "margin": margin, "correct": is_correct})
            continue

        for card in cards:
            expected = label_code(card["label"])
            expected_classes.add(expected)
            box = [int(v) for v in card["box"]]
            if len(box) != 4 or box[2] <= box[0] or box[3] <= box[1]:
                raise ValueError(f"{image_name} 的无效标注框：{box}")
            choices = sorted(
                ((iou(box, found), index) for index, found in enumerate(detections) if index not in used),
                reverse=True,
            )
            overlap, index = choices[0] if choices else (0.0, -1)
            total += 1
            if index < 0 or overlap < args.min_iou:
                misses += 1
                rows.append({"image": image_name, "expected": expected, "predicted": "NO_DETECTION",
                             "iou": overlap, "correct": False})
                continue
            used.add(index)
            prediction, confidence, margin = classify_card_regions(
                session, image, [detections[index]]
            )[0]
            predicted = prediction
            for symbol, code in (("♣", "C"), ("♦", "D"), ("♥", "H"), ("♠", "S")):
                predicted = predicted.replace(symbol, code)
            is_correct = predicted == expected
            correct += int(is_correct)
            low_confidence += int(prediction == "??")
            rows.append({"image": image_name, "expected": expected, "predicted": predicted,
                         "confidence": confidence, "margin": margin, "iou": overlap,
                         "correct": is_correct})
        extras += len(detections) - len(used)

    failed_predictions = total - correct - misses
    failures = misses + extras + failed_predictions
    status = "passed" if total >= 52 and len(expected_classes) == 52 and failures == 0 else "failed"
    report = {
        "schema": "pokerai-real-replay-v1",
        "status": status,
        "model": model_path.name,
        "model_sha256": sha256(model_path),
        "manifest_sha256": sha256(manifest_path),
        "total_cards": total,
        "correct_cards": correct,
        "distinct_expected_classes": len(expected_classes),
        "expected_classes": sorted(expected_classes),
        "screenshot_count": len(screenshots),
        "detection_misses": misses,
        "spurious_detections": extras,
        "low_confidence_count": low_confidence,
        "failures": [row for row in rows if not row["correct"]],
        "results": rows,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "status", "model_sha256", "total_cards", "correct_cards",
        "distinct_expected_classes", "screenshot_count", "detection_misses",
        "spurious_detections", "low_confidence_count",
    )}, ensure_ascii=False))
    return 0 if status == "passed" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (CardVisionError, OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"回放失败：{exc}", file=sys.stderr)
        raise SystemExit(2)
