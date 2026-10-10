"""Crop Pascal VOC face annotations; split original images before cropping."""
import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

from common import LABELS


def read_records(raw):
    annotations, images = raw / "annotations", raw / "images"
    if not annotations.is_dir() or not images.is_dir():
        raise ValueError(f"Can thu muc {images} va {annotations}. Xem README.md.")
    records = []
    for xml in sorted(annotations.glob("*.xml")):
        root = ET.parse(xml).getroot()
        name = root.findtext("filename", "").strip()
        # Some Pascal VOC exports omit filename; resolve matching stem instead.
        candidates = [images / Path(name).name] if name else sorted(images.glob(xml.stem + ".*"))
        if len(candidates) != 1 or not candidates[0].is_file():
            raise ValueError(f"Khong tim thay anh duy nhat cho {xml.name}")
        path = candidates[0]
        with Image.open(path) as source:
            image = source.convert("RGB")
            width, height = image.size
            digest = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
        boxes = []
        for obj in root.findall("object"):
            label = obj.findtext("name", "").strip()
            if label not in LABELS:
                raise ValueError(f"Nhan khong ho tro {label!r}: {xml.name}")
            box = obj.find("bndbox")
            if box is None:
                raise ValueError(f"Thieu bndbox: {xml.name}")
            # Pascal VOC coordinates are 1-based, inclusive; Pillow uses 0-based bounds.
            x1 = max(0, min(width, math.floor(float(box.findtext("xmin"))) - 1))
            y1 = max(0, min(height, math.floor(float(box.findtext("ymin"))) - 1))
            x2 = max(0, min(width, math.ceil(float(box.findtext("xmax")))))
            y2 = max(0, min(height, math.ceil(float(box.findtext("ymax")))))
            if x2 <= x1 or y2 <= y1:
                raise ValueError(f"Box khong hop le: {xml.name}, {label}")
            boxes.append((label, (x1, y1, x2, y2)))
        if boxes:
            records.append({"path": path, "hash": digest, "boxes": boxes})
    if not records:
        raise ValueError("Khong co annotation hop le.")
    return records


def choose_split(records, seed):
    # Identical decoded images stay together, even if their filenames differ.
    groups = sorted({r["hash"] for r in records})
    counts = {g: np.zeros(len(LABELS), dtype=int) for g in groups}
    for record in records:
        for label, _ in record["boxes"]:
            counts[record["hash"]][LABELS.index(label)] += 1
    n = len(groups)
    sizes = [int(n * 0.70), int(n * 0.15)]
    sizes.append(n - sum(sizes))
    if min(sizes) < 1:
        raise ValueError("Qua it anh goc de chia train/val/test.")
    matrix = np.stack([counts[g] for g in groups])
    if np.any((matrix > 0).sum(axis=0) < 3):
        raise ValueError("Moi nhan can xuat hien trong it nhat 3 nhom anh goc khac nhau.")
    totals = matrix.sum(axis=0)
    rng = np.random.default_rng(seed)
    best, best_score = None, float("inf")
    # Select by label counts only, never by model performance or image features.
    for _ in range(500):
        order = rng.permutation(n)
        parts = np.split(order, np.cumsum(sizes)[:-1])
        split_counts = np.stack([matrix[p].sum(axis=0) for p in parts])
        if np.any(split_counts == 0):
            continue
        score = float(np.square(split_counts / totals - np.array([0.70, 0.15, 0.15])[:, None]).sum())
        if score < best_score:
            best, best_score = parts, score
    if best is None:
        raise ValueError("Khong chia duoc ca 3 nhan vao moi tap. Can them anh cho nhan it mau.")
    return {groups[i]: split for split, part in zip(["train", "val", "test"], best) for i in part}


def prepare(raw, out, seed=42, padding=0.18):
    if not 0 <= padding <= 0.5:
        raise ValueError("Padding phai tu 0 den 0.5.")
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"{out} khong trong. Dung --out voi thu muc moi de giu ban cu.")
    records = read_records(raw)
    assignments = choose_split(records, seed)
    for split in ["train", "val", "test"]:
        for label in LABELS:
            (out / split / label).mkdir(parents=True, exist_ok=True)
    manifest, counts = [], {s: Counter() for s in ["train", "val", "test"]}
    for record_index, record in enumerate(records):
        split = assignments[record["hash"]]
        with Image.open(record["path"]) as source:
            image = source.convert("RGB")
            width, height = image.size
            for face_index, (label, box) in enumerate(record["boxes"]):
                x1, y1, x2, y2 = box
                side = min(max(x2 - x1, y2 - y1) * (1 + 2 * padding), width, height)
                left = max(0, min(width - side, (x1 + x2 - side) / 2))
                top = max(0, min(height - side, (y1 + y2 - side) / 2))
                bounds = (int(left), int(top), min(width, math.ceil(left + side)), min(height, math.ceil(top + side)))
                name = f"{record_index:05d}_{record['path'].stem}_face{face_index}.png"
                dest = out / split / label / name
                image.crop(bounds).save(dest)
                counts[split][label] += 1
                manifest.append({"crop": dest.relative_to(out).as_posix(), "source": record["path"].name,
                                 "source_hash": record["hash"], "split": split, "label": label,
                                 "box": box, "crop_bounds": bounds})
    empty = [f"{split}/{label}" for split in counts for label in LABELS if counts[split][label] == 0]
    if empty:
        raise ValueError(f"Khong co mau cho: {', '.join(empty)}. Can them anh cho nhan it mau.")
    summary = {"seed": seed, "padding": padding, "labels": LABELS,
               "source_images": len(records), "unique_source_groups": len(assignments),
               "counts": {s: dict(c) for s, c in counts.items()}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("data/processed"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--padding", type=float, default=0.18)
    args = parser.parse_args()
    try:
        prepare(args.raw, args.out, args.seed, args.padding)
    except (ValueError, OSError, ET.ParseError) as error:
        parser.exit(1, f"Loi: {error}\n")
