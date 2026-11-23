"""Fit a frozen-backbone LRASPP candidate and score the original and new ONNX."""
import argparse
import hashlib
import json
from pathlib import Path
import random

import cv2
import numpy as np
import onnxruntime as ort
import torch
from torch.utils.data import DataLoader
import torchvision

from train_rubble_seg import RubbleData


def split(root):
    folds = {"train": [], "val": []}
    for folder in ("mujoco_rubble3", "mujoco_clutter"):
        directory = root / "data" / folder
        meta = json.loads((directory / "meta.json").read_text())
        for name, field in (("train", "train"), ("val", "test")):
            for stem in sorted(meta[field]):
                folds[name].append((str(directory / "images" / (stem + ".png")),
                                    str(directory / "masks" / (stem + ".png"))))
    for name, pairs in folds.items():
        if not pairs or not all(Path(p).is_file() for pair in pairs for p in pair):
            raise ValueError(f"Missing {name} data")
    if {p[0] for p in folds["train"]} & {p[0] for p in folds["val"]}:
        raise ValueError("Train/validation overlap")
    return folds


def score(path, items):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    session = ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])
    rows = []
    for (image, _), (x, label) in zip(items, RubbleData(items, train=False)):
        logits = session.run(None, {session.get_inputs()[0].name: x.numpy()[None]})[0][0]
        exp = np.exp(logits - logits.max(axis=0, keepdims=True))
        pred = exp[1] / exp.sum(axis=0) >= .60
        gt = label.numpy() == 1
        rows.append({"frame": Path(image).stem, "tp": int((pred & gt).sum()),
                     "fp": int((pred & ~gt).sum()), "fn": int((~pred & gt).sum())})
    tp, fp, fn = (sum(row[k] for row in rows) for k in ("tp", "fp", "fn"))
    return {"precision": tp/max(1, tp+fp), "recall": tp/max(1, tp+fn),
            "iou": tp/max(1, tp+fp+fn), "samples": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True, exist_ok=False)
    baseline = root / "models/seg_low_obstacle_v10.onnx"
    init = baseline.with_suffix(".pt")
    folds = split(root)
    seed = 26
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torchvision.models.segmentation.lraspp_mobilenet_v3_large(
        weights=None, weights_backbone=None, num_classes=3)
    model.load_state_dict(torch.load(init, map_location="cpu", weights_only=True))
    model.backbone.requires_grad_(False)
    model.to(device).train()
    model.backbone.eval()
    loader = DataLoader(RubbleData(folds["train"]), batch_size=4, shuffle=True)
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=8e-5)
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor([1., 2., 1.25], device=device))
    print("device", device, {k: len(v) for k, v in folds.items()}, flush=True)
    losses = []
    for _ in range(args.epochs):
        values = []
        for x, y in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(x.to(device))["out"], y.to(device))
            loss.backward()
            optimizer.step()
            values.append(float(loss.detach()))
        losses.append(float(np.mean(values)))
        print("epoch", len(losses), "loss", losses[-1], flush=True)
    model.cpu().eval()
    torch.save(model.state_dict(), args.output / "rubble_candidate.pt")
    candidate = args.output / "rubble_candidate.onnx"
    torch.onnx.export(model, torch.zeros(1, 3, 320, 480), str(candidate), opset_version=17,
                      input_names=["image"], output_names=["logits"], dynamo=False)
    results = {}
    for path in (baseline, candidate):
        results[path.name] = score(path, folds["val"])
        results[path.name]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (args.output / "training.json").write_text(json.dumps({
        "seed": seed, "epochs": args.epochs, "device": device, "lr": 8e-5,
        "initial_weights_sha256": hashlib.sha256(init.read_bytes()).hexdigest(),
        "losses": losses, "split": {k: [tuple(str(Path(p).relative_to(root)) for p in pair)
                                     for pair in v] for k, v in folds.items()},
        "validation": results,
        "validation_scope": "grouped existing data, held out from this fine tune; initial weights exposure unknown",
    }, indent=2))
    print({model: {metric: stats[metric] for metric in ("precision", "recall", "iou")}
           for model, stats in results.items()}, flush=True)


if __name__ == "__main__":
    main()
