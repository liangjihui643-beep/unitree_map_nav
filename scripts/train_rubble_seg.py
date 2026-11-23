#!/usr/bin/env python3
"""Fine-tune the low-obstacle segmentation model with MuJoCo rubble labels.

Run with the system Python that has torch (not the repo venv):
    /usr/local/bin/python3.12 scripts/train_rubble_seg.py
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

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
ROOT = Path(__file__).resolve().parents[1]


class RubbleData(Dataset):
    def __init__(self, items, train=True):
        self.items, self.train = items, train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        ip, mp = self.items[i]
        img = cv2.cvtColor(cv2.imread(ip), cv2.COLOR_BGR2RGB)
        msk = cv2.imread(mp, 0)
        msk = np.clip(msk, 0, 2).astype(np.uint8)
        if self.train:
            if random.random() < 0.5:
                img, msk = img[:, ::-1].copy(), msk[:, ::-1].copy()
            if random.random() < 0.3:
                img = np.clip(img.astype(np.float32) * random.uniform(0.8, 1.2), 0, 255).astype(np.uint8)
        img = cv2.resize(img, (480, 320))
        msk = cv2.resize(msk, (480, 320), interpolation=cv2.INTER_NEAREST)
        x = (img.astype(np.float32) / 255.0 - MEAN) / STD
        return torch.from_numpy(x).permute(2, 0, 1), torch.from_numpy(msk.astype(np.int64))


def build_items():
    items = []
    mu = json.loads((ROOT / "data/mujoco_rubble3/meta.json").read_text())
    for name in mu["train"]:
        items.append((str(ROOT / "data/mujoco_rubble3/images" / (name + ".png")),
                      str(ROOT / "data/mujoco_rubble3/masks" / (name + ".png"))))
    cl = json.loads((ROOT / "data/mujoco_clutter/meta.json").read_text())
    for name in cl["train"]:
        items.append((str(ROOT / "data/mujoco_clutter/images" / (name + ".png")),
                      str(ROOT / "data/mujoco_clutter/masks" / (name + ".png"))))
    # Gazebo labelled data keeps the old domain alive
    gz = "/home/ros/rgb_depth_dataset"
    for ip in sorted(glob.glob(gz + "/images/*.png")):
        mp = ip.replace("/images/", "/masks/")
        m = cv2.imread(mp, 0)
        if m is not None and (m > 0).mean() > 0.005 and len(items) < 80:
            items.append((ip, mp))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default=str(ROOT / "models/seg_low_obstacle_v2.pt"))
    ap.add_argument("--out", default=str(ROOT / "models/seg_low_obstacle_v4.pt"))
    ap.add_argument("--onnx", default=str(ROOT / "models/seg_low_obstacle_v4.onnx"))
    ap.add_argument("--epochs", type=int, default=12)
    args = ap.parse_args()

    items = build_items()
    random.seed(1)
    random.shuffle(items)
    print("frames", len(items))
    dl = DataLoader(RubbleData(items), batch_size=4, shuffle=True, num_workers=0)
    model = torchvision.models.segmentation.lraspp_mobilenet_v3_large(
        weights=None, weights_backbone=None, num_classes=3)
    model.load_state_dict(torch.load(args.init, map_location="cpu"))
    device = "cpu"
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    lossf = nn.CrossEntropyLoss(weight=torch.tensor([0.5, 3.0, 1.0]).to(device))
    best = 1e9
    for ep in range(args.epochs):
        model.train()
        total = 0.0
        for x, y in dl:
            x, y = x.to(device), y.to(device)
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
    print(json.dumps({"best_loss": round(best, 4), "pt": args.out, "onnx": args.onnx,
                      "frames": len(items)}))


if __name__ == "__main__":
    main()
