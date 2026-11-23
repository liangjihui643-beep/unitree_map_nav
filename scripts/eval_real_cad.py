#!/usr/bin/env python3
"""Material-level response of the low-obstacle segmenter on real C&D waste photos.

The C&D dataset (Zenodo 17447590, CC-BY-4.0) is conveyor close-ups with YOLO
boxes for Timber/Brick/Concrete/etc.  This script measures, per class, the
fraction of annotated objects that contain predicted rubble pixels, plus the
predicted-rubble fraction outside all boxes (background false positives).
Geometry/height are not evaluated: real photos have no calibrated camera here.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

from go2_rescue.config import Config
from go2_rescue.low_obstacle import LowObstacleDetector

ROOT = Path(__file__).resolve().parents[1]
CAD = ROOT / "data/real/cad/AI waste Sorting_2.v5i.yolov11"



def load_yaml_classes(path: Path):
    import ast
    for line in path.read_text(errors="ignore").splitlines():
        s = line.strip()
        if s.startswith("names:"):
            return list(ast.literal_eval(s.split(":", 1)[1].strip()))
    return []


def boxes_for(label_path: Path):
    out = []
    for line in label_path.read_text(errors="ignore").splitlines():
        p = line.split()
        if len(p) >= 5:
            out.append((int(float(p[0])), *[float(x) for x in p[1:5]]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", nargs="+", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--per-class", type=int, default=40)
    ap.add_argument("--out", default="/tmp/eval_real_cad.json")
    args = ap.parse_args()
    classes = load_yaml_classes(CAD / "data.yaml")
    dets = {Path(o).stem: LowObstacleDetector(o, Config().camera_matrix,
                                              Config().body_from_camera,
                                              Config().base_height) for o in args.onnx}
    img_dir = CAD / args.split / "images"
    lbl_dir = CAD / args.split / "labels"
    files = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    random.seed(0)
    random.shuffle(files)
    per = {c: [] for c in classes}
    for f in files:
        lp = lbl_dir / (f.stem + ".txt")
        if not lp.exists():
            continue
        bs = boxes_for(lp)
        if not bs:
            continue
        cls = classes[bs[0][0]] if bs[0][0] < len(classes) else "?"
        if cls in per and len(per[cls]) < args.per_class:
            per[cls].append((f, bs))
    results = {}
    for tag, det in dets.items():
        stats = {}
        for cls, rows in per.items():
            if not rows:
                continue
            hits, total, in_frac, out_frac = 0, 0, [], []
            for f, bs in rows:
                img = cv2.imread(str(f))
                if img is None:
                    continue
                prob = det.segment(img)
                pred = (prob[1] > 0.6).astype(np.uint8)
                h, w = pred.shape
                covered = np.zeros_like(pred)
                for _, cx, cy, bw, bh in bs:
                    x0 = max(0, int((cx - bw / 2) * w)); x1 = min(w, int((cx + bw / 2) * w))
                    y0 = max(0, int((cy - bh / 2) * h)); y1 = min(h, int((cy + bh / 2) * h))
                    if x1 <= x0 or y1 <= y0:
                        continue
                    box = pred[y0:y1, x0:x1]
                    covered[y0:y1, x0:x1] = 1
                    total += 1
                    frac = float(box.mean())
                    in_frac.append(frac)
                    hits += int(frac > 0.15)
                outside = pred[covered == 0]
                if len(outside):
                    out_frac.append(float(outside.mean()))
            stats[cls] = {
                "objects": total,
                "hit_rate": round(hits / max(1, total), 3),
                "mean_in_box_fraction": round(float(np.mean(in_frac)), 3) if in_frac else None,
                "mean_outside_fraction": round(float(np.mean(out_frac)), 3) if out_frac else None,
            }
        results[tag] = stats
    Path(args.out).write_text(json.dumps(results, indent=1))
    print(json.dumps(results, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
