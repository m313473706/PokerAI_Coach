#!/usr/bin/env python3
"""Train a small 52-class card-crop classifier from CC0 card artwork."""
import argparse
import hashlib
import json
import random
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import onnx

RANKS = ("2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A")
SUITS = ("C", "D", "H", "S")
SUIT_FILES = {"clubs": "C", "diamonds": "D", "hearts": "H", "spades": "S"}
CARD_LABELS = tuple(rank + suit for rank in RANKS for suit in SUITS)
HEIGHT, WIDTH = 192, 128
DATASET_REPO = "https://github.com/safe-acid/Playing-cards-assets"
DATASET_COMMIT = "ae925833a28f587ec6abed561822889eb044a18c"
DATASET_SOURCE = f"{DATASET_REPO}/tree/{DATASET_COMMIT}"
DATASET_LICENSE = f"{DATASET_REPO}/blob/{DATASET_COMMIT}/LICENSE"
MODEL_LICENSE_URL = "https://creativecommons.org/publicdomain/zero/1.0/legalcode"


def parse_card_path(path):
    match = re.fullmatch(r"(A|[2-9]|10|J|Q|K)_+(clubs|diamonds|hearts|spades)", path.stem, re.IGNORECASE)
    if not match:
        return None
    rank, suit = match.groups()
    rank = rank.upper()
    suit = SUIT_FILES[suit.lower()]
    label = rank + suit
    return CARD_LABELS.index(label) if label in CARD_LABELS else None


def read_assets(asset_dir):
    by_label = {}
    for path in Path(asset_dir).glob("*.png"):
        label_id = parse_card_path(path)
        if label_id is None:
            continue
        with Image.open(path) as source:
            rgba = source.convert("RGBA")
            background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            background.alpha_composite(rgba)
            by_label[label_id] = background.convert("RGB")
    missing = [CARD_LABELS[i] for i in range(len(CARD_LABELS)) if i not in by_label]
    if missing:
        raise RuntimeError("CC0 牌面素材没有覆盖全部 52 类：" + ", ".join(missing))
    if len(by_label) != 52:
        raise RuntimeError(f"素材中应恰好有 52 类，实际为 {len(by_label)} 类。")
    return [by_label[i] for i in range(52)]


def make_example(source, rng):
    image = source.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)
    scale = rng.uniform(0.86, 1.04)
    scaled = image.resize((max(1, round(WIDTH * scale)), max(1, round(HEIGHT * scale))),
                          Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (WIDTH, HEIGHT), (rng.randint(238, 255),) * 3)
    offset_x = (WIDTH - scaled.width) // 2 + rng.randint(-4, 4)
    offset_y = (HEIGHT - scaled.height) // 2 + rng.randint(-5, 5)
    canvas.paste(scaled, (offset_x, offset_y))
    canvas = canvas.rotate(rng.uniform(-12, 12), resample=Image.Resampling.BICUBIC,
                           fillcolor=(rng.randint(238, 255),) * 3)
    canvas = ImageEnhance.Brightness(canvas).enhance(rng.uniform(0.78, 1.18))
    canvas = ImageEnhance.Contrast(canvas).enhance(rng.uniform(0.82, 1.22))
    canvas = ImageEnhance.Color(canvas).enhance(rng.uniform(0.80, 1.20))
    if rng.random() < 0.18:
        canvas = canvas.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.2, 0.8)))
    return canvas


class SyntheticCardCrops(Dataset):
    def __init__(self, cards, samples_per_class, training):
        self.cards = cards
        self.samples_per_class = samples_per_class
        self.training = training

    def __len__(self):
        return len(self.cards) * self.samples_per_class

    def __getitem__(self, index):
        label = index // self.samples_per_class
        rng = random if self.training else random.Random(4171 + index)
        image = make_example(self.cards[label], rng)
        array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
        return torch.from_numpy(array), label


class CardNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 24, 3, padding=1), nn.BatchNorm2d(24), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(24, 48, 3, padding=1), nn.BatchNorm2d(48), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(48, 80, 3, padding=1), nn.BatchNorm2d(80), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(80, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.15), nn.Linear(128, 52))

    def forward(self, image):
        return self.classifier(self.features(image))


class ProbabilityOutput(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        return torch.softmax(self.model(image), dim=1)


def evaluate(model, loader, device):
    model.eval()
    total = correct = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            predictions = model(images).argmax(dim=1)
            correct += int((predictions == labels).sum().item())
            total += int(labels.shape[0])
    return correct / max(1, total)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets-dir", required=True)
    parser.add_argument("--output", default="models/playing_cards.onnx")
    parser.add_argument("--report", default="models/card_model_training.json")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--samples-per-class", type=int, default=64)
    parser.add_argument("--validation-per-class", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20261019)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    device = torch.device("cpu")
    cards = read_assets(args.assets_dir)
    train_loader = DataLoader(
        SyntheticCardCrops(cards, args.samples_per_class, True),
        batch_size=args.batch_size, shuffle=True, num_workers=0,
    )
    validation_loader = DataLoader(
        SyntheticCardCrops(cards, args.validation_per_class, False),
        batch_size=args.batch_size, shuffle=False, num_workers=0,
    )

    model = CardNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0005)
    loss_function = nn.CrossEntropyLoss()
    best_accuracy = -1.0
    best_state = None
    history = []
    for epoch in range(args.epochs):
        model.train()
        loss_sum = 0.0
        seen = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = loss_function(logits, labels)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * int(labels.shape[0])
            seen += int(labels.shape[0])
        accuracy = evaluate(model, validation_loader, device)
        history.append({
            "epoch": epoch + 1,
            "train_loss": loss_sum / max(1, seen),
            "synthetic_validation_accuracy": accuracy,
        })
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(f"epoch {epoch + 1}/{args.epochs}: loss={history[-1]['train_loss']:.4f} synthetic_val={accuracy:.4f}",
              flush=True)

    if best_state is None:
        raise RuntimeError("模型训练未产生可导出的检查点。")
    model.load_state_dict(best_state)
    model.eval()
    output_path = Path(args.output)
    report_path = Path(args.report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    wrapped = ProbabilityOutput(model).eval()
    dummy = torch.zeros((1, 3, HEIGHT, WIDTH), dtype=torch.float32)
    torch.onnx.export(
        wrapped, dummy, str(output_path),
        input_names=["card_crop"], output_names=["class_probabilities"],
        opset_version=17, do_constant_folding=True, dynamo=False,
    )
    graph = onnx.load(str(output_path))
    onnx.checker.check_model(graph)
    metadata = {
        "pokerai_schema": "pokerai-card-classifier-v1",
        "class_labels": json.dumps(CARD_LABELS, separators=(",", ":")),
        "license": "CC0-1.0",
        "source_url": DATASET_SOURCE,
        "license_url": MODEL_LICENSE_URL,
        "training_data_license": "CC0-1.0",
        "training_data_license_url": DATASET_LICENSE,
        "training_method": "custom_cnn_random_initialization",
        "target_replay_status": "not_run",
    }
    del graph.metadata_props[:]
    for key, value in metadata.items():
        item = graph.metadata_props.add()
        item.key = key
        item.value = value
    onnx.save(graph, str(output_path))

    model_hash = hashlib.sha256(output_path.read_bytes()).hexdigest()
    report = {
        "model": output_path.name,
        "sha256": model_hash,
        "license": metadata["license"],
        "source_url": DATASET_SOURCE,
        "license_url": MODEL_LICENSE_URL,
        "training_data_license_url": DATASET_LICENSE,
        "classes": list(CARD_LABELS),
        "architecture": "CardNet, initialized from random weights",
        "epochs": args.epochs,
        "samples_per_class_per_epoch": args.samples_per_class,
        "synthetic_validation_accuracy": best_accuracy,
        "target_screenshot_replay": "not_run",
        "history": history,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {output_path} sha256={model_hash}", flush=True)
    print(f"synthetic validation accuracy={best_accuracy:.4f}; target screenshot replay not run", flush=True)


if __name__ == "__main__":
    main()
