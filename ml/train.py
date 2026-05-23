from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SignClipDataset
from model import build_model

ROOT = Path(__file__).resolve().parent.parent
ML_ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(ML_ROOT / "data" / "manifest.json"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--model-version", default="v1")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    sign_ids = sorted(set(manifest.get("sign_ids", [])) or {c["sign_id"] for c in manifest["clips"]})
    label_to_idx = {s: i for i, s in enumerate(sign_ids)}

    train_ds = SignClipDataset(manifest_path, "train", label_to_idx)
    val_ds = SignClipDataset(manifest_path, "val", label_to_idx)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=len(sign_ids)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    ckpt_dir = ML_ROOT / "checkpoints" / args.model_version
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()
            total_loss += loss.item()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        acc = correct / max(total, 1)
        print(f"Epoch {epoch+1} loss={total_loss/len(train_loader):.4f} val_acc={acc:.4f}")
        if acc >= best_acc:
            best_acc = acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "label_to_idx": label_to_idx,
                    "sign_ids": sign_ids,
                    "model_version": args.model_version,
                    "val_accuracy": acc,
                },
                ckpt_dir / "best.pt",
            )

    meta = {
        "model_version": args.model_version,
        "num_classes": len(sign_ids),
        "sign_ids": sign_ids,
        "val_accuracy": best_acc,
        "pretrained": False,
    }
    exports = ML_ROOT / "exports"
    exports.mkdir(parents=True, exist_ok=True)
    with open(exports / "model_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Best val acc: {best_acc:.4f} -> {ckpt_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
