#!/usr/bin/env python3
"""Weakly-labelled real C&D data: obstacle boxes -> rubble, other content -> ground.

Obstacle classes (Brick/Concrete/Reinforcement/timber) become class 1; the
Plastic Bottle class is intentionally treated as background so the model learns
that not every man-made object protruding above ground is a low obstacle.
"""
from __future__ import annotations

import argparse
import ast
import json
import random
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CAD = ROOT / "data/real/cad/AI waste Sorting_2.v5i.yolov11"
OBSTACLE = {"Brick", "Concrete", "Reinforcement", "timber"}


def classes(path: Path):
    for line in path.read_text(errors="ignore").splitlines():
        if line.strip().startswith("names:"):
            return list(ast.literal_eval(line.split(":", 1)[1].strip()))
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data/real/cad_weak"))
    ap.add_argument("--limit", type=int, default=300)
    args = ap.parse_args()
    names = classes(CAD / "data.yaml")
    keep = [i for i, n in enumerate(names) if n in OBSTACLE]
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    files = sorted((CAD / "train/images").glob("*.jpg")) + sorted((CAD / "train/images").glob("*.png"))
    random.seed(3)
    random.shuffle(files)
    saved = 0
    for f in files:
        if saved >= args.limit:
            break
        lp = CAD / "train/labels" / (f.stem + ".txt")
        if not lp.exists():
            continue
        img = cv2.imread(str(f))
        if img is None:
            continue
        h, w = img.shape[:2]
        mask = np.zeros((h, w), np.uint8)
        hit = False
        for line in lp.read_text(errors="ignore").splitlines():
            p = line.split()
            if len(p) < 5:
                continue
            c = int(float(p[0]))
            if c not in keep:
                continue
            cx, cy, bw, bh = [float(x) for x in p[1:5]]
            x0 = max(0, int((cx - bw / 2) * w)); x1 = min(w, int((cx + bw / 2) * w))
            y0 = max(0, int((cy - bh / 2) * h)); y1 = min(h, int((cy + bh / 2) * h))
            if x1 > x0 and y1 > y0:
                mask[y0:y1, x0:x1] = 1
                hit = True
        if not hit:
            continue
        name = f.stem
        cv2.imwrite(str(out / "images" / (name + ".png")), img)
        cv2.imwrite(str(out / "masks" / (name + ".png")), mask)
        saved += 1
    (out / "meta.json").write_text(json.dumps({"images": saved, "classes": names}, indent=1))
    print(json.dumps({"saved": saved, "out": str(out)}))


if __name__ == "__main__":
    main()
