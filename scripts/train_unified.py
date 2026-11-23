#!/usr/bin/env python3
"""Unified model experiment (v7): freeze the segmentation backbone, adapt the head
on mixed simulation + real C&D data, then evaluate all three domains.

Rationale: v6 (balanced full fine-tune) preserved real-domain selectivity but
degraded simulation precision. Freezing the backbone keeps the simulation
features while the head learns real-domain appearance.

Run with system Python (torch):
    /usr/local/bin/python3.12 scripts/train_unified.py
"""
from __future__ import annotations

import argparse
import glob
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


class Mix(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        ip, mp = self.items[i]
        img = cv2.cvtColor(cv2.imread(ip), cv2.COLOR_BGR2RGB)
        msk = np.clip(cv2.imread(mp, 0), 0, 2).astype(np.uint8)
        if random.random() < 0.5:
            img, msk = img[:, ::-1].copy(), msk[:, ::-1].copy()
        if random.random() < 0.4:
            img = np.clip(img.astype(np.float32) * random.uniform(0.75, 1.25), 0, 255).astype(np.uint8)
        img = cv2.resize(img, (480, 320))
        msk = cv2.resize(msk, (480, 320), interpolation=cv2.INTER_NEAREST)
        x = (img.astype(np.float32) / 255.0 - MEAN) / STD
        w = 2.0 if "mujoco_clutter" in ip else 1.0
        return (torch.from_numpy(x).permute(2, 0, 1),
                torch.from_numpy(msk.astype(np.int64)), torch.tensor(w))


def gather(real_limit=300, clutter_repeat=1):
    items = []
    real = ROOT / "data/real/cad_weak"
    for ip in sorted((real / "images").glob("*.png"))[:real_limit]:
        mp = real / "masks" / ip.name
        if mp.exists():
            items.append((str(ip), str(mp)))
    for meta_dir in ("data/mujoco_rubble3", "data/mujoco_clutter", "data/mujoco_rubble_far"):
        base = ROOT / meta_dir
        meta = json.loads((base / "meta.json").read_text())
        repeat = clutter_repeat if "clutter" in meta_dir else 1
        for _ in range(repeat):
            for name in meta.get("train", []):
                items.append((str(base / "images" / (name + ".png")),
                              str(base / "masks" / (name + ".png"))))
    for ip in sorted(glob.glob("/home/ros/rgb_depth_dataset/images/*.png")):
        mp = ip.replace("/images/", "/masks/")
        if Path(mp).exists():
            items.append((ip, mp))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default=str(ROOT / "models/seg_low_obstacle_v4.pt"))
    ap.add_argument("--out", default=str(ROOT / "models/seg_low_obstacle_v9.pt"))
    ap.add_argument("--onnx", default=str(ROOT / "models/seg_low_obstacle_v9.onnx"))
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--real-limit", type=int, default=300)
    ap.add_argument("--clutter-repeat", type=int, default=1)
    ap.add_argument("--freeze-backbone", type=int, default=1)
    args = ap.parse_args()
    items = gather(args.real_limit, args.clutter_repeat)
    random.seed(4)
    random.shuffle(items)
    print("frames", len(items), "freeze_backbone", bool(args.freeze_backbone), flush=True)
    dl = DataLoader(Mix(items), batch_size=4, shuffle=True, num_workers=0)
    model = torchvision.models.segmentation.lraspp_mobilenet_v3_large(
        weights=None, weights_backbone=None, num_classes=3)
    model.load_state_dict(torch.load(args.init, map_location="cpu"))
    if args.freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad = False
    model.to("cpu")
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    lossf = nn.CrossEntropyLoss(weight=torch.tensor([0.4, 2.5, 1.0]), reduction="none")
    best = 1e9
    for ep in range(args.epochs):
        model.train()
        total = 0.0
        for x, y, w in dl:
            opt.zero_grad()
            out = model(x)["out"]
            per = lossf(out, y).mean(dim=(1, 2))
            loss = (per * w).mean()
            loss.backward()
            opt.step()
            total += loss.item()
        avg = total / max(1, len(dl))
        print("EP %02d loss %.4f" % (ep, avg), flush=True)
        if avg < best:
            best = avg
            torch.save(model.state_dict(), args.out)
    import hashlib
    # Reload the best-loss checkpoint so the exported ONNX matches the PT file.
    model.load_state_dict(torch.load(args.out, map_location="cpu"))
    model.eval()
    torch.onnx.export(model, torch.randn(1, 3, 320, 480), args.onnx,
                      input_names=["image"], output_names=["logits"], opset_version=17, dynamo=False)
    sha_pt = hashlib.sha256(open(args.out, "rb").read()).hexdigest()
    sha_onnx = hashlib.sha256(open(args.onnx, "rb").read()).hexdigest()
    sidecar = {"pt": args.out, "onnx": args.onnx, "sha256_pt": sha_pt,
               "sha256_onnx": sha_onnx, "best_loss": round(best, 4), "frames": len(items)}
    io.open(args.onnx + ".json", "w", encoding="utf-8").write(json.dumps(sidecar, indent=1))
    print(json.dumps(sidecar))


if __name__ == "__main__":
    main()
