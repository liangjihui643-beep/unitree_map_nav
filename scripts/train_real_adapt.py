#!/usr/bin/env python3
"""Real-domain adaptation: mix weakly-labelled C&D photos with simulation data.

Run with system Python (torch):
    /usr/local/bin/python3.12 scripts/train_real_adapt.py
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
        return torch.from_numpy(x).permute(2, 0, 1), torch.from_numpy(msk.astype(np.int64))


def gather(limit=300):
    items = []
    real = ROOT / "data/real/cad_weak"
    for ip in sorted((real / "images").glob("*.png"))[:limit]:
        mp = real / "masks" / ip.name
        if mp.exists():
            items.append((str(ip), str(mp)))
    for meta_dir in ("data/mujoco_rubble3", "data/mujoco_clutter", "data/mujoco_rubble_far"):
        base = ROOT / meta_dir
        meta = json.loads((base / "meta.json").read_text())
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
    ap.add_argument("--out", default=str(ROOT / "models/seg_low_obstacle_v6.pt"))
    ap.add_argument("--onnx", default=str(ROOT / "models/seg_low_obstacle_v6.onnx"))
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--real-limit", type=int, default=150)
    ap.add_argument("--lr", type=float, default=5e-5)
    args = ap.parse_args()
    items = gather(args.real_limit)
    random.seed(2)
    random.shuffle(items)
    print("frames", len(items))
    dl = DataLoader(Mix(items), batch_size=4, shuffle=True, num_workers=0)
    model = torchvision.models.segmentation.lraspp_mobilenet_v3_large(
        weights=None, weights_backbone=None, num_classes=3)
    model.load_state_dict(torch.load(args.init, map_location="cpu"))
    model.to("cpu")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    lossf = nn.CrossEntropyLoss(weight=torch.tensor([0.4, 2.5, 1.0]))
    best = 1e9
    for ep in range(args.epochs):
        model.train()
        total = 0.0
        for x, y in dl:
            opt.zero_grad()
            out = model(x)["out"]
            loss = lossf(out, y)
            loss.backward()
            opt.step()
            total += loss.item()
        avg = total / max(1, len(dl))
        print("EP %02d loss %.4f" % (ep, avg))
        if avg < best:
            best = avg
            torch.save(model.state_dict(), args.out)
    model.eval()
    torch.onnx.export(model, torch.randn(1, 3, 320, 480), args.onnx,
                      input_names=["image"], output_names=["logits"], opset_version=17, dynamo=False)
    print(json.dumps({"best": round(best, 4), "frames": len(items), "onnx": args.onnx}))


if __name__ == "__main__":
    main()
