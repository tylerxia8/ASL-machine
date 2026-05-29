from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from landmark_dataset import HandLandmarkDataset, HAND_FEATURES, NUM_FRAMES
from landmark_model import build_landmark_model

ROOT = Path(__file__).resolve().parent.parent
ML_ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(ML_ROOT / "data" / "manifest.json"))
    parser.add_argument("--feature-dir", default=str(ML_ROOT / "data" / "hand_landmarks"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--model-version", default="hand-landmarks-v1")
    parser.add_argument("--early-stop-patience", type=int, default=8)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.02)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--min-feature-coverage", type=float, default=0.20)
    parser.add_argument("--landmark-noise-std", type=float, default=0.01)
    parser.add_argument("--landmark-dropout-prob", type=float, default=0.05)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    sign_ids = sorted(set(manifest.get("sign_ids", [])) or {c["sign_id"] for c in manifest["clips"]})
    label_to_idx = {s: i for i, s in enumerate(sign_ids)}

    train_ds = HandLandmarkDataset(
        manifest_path,
        "train",
        label_to_idx,
        Path(args.feature_dir),
        min_coverage=args.min_feature_coverage,
        noise_std=args.landmark_noise_std,
        dropout_prob=args.landmark_dropout_prob,
    )
    val_ds = HandLandmarkDataset(manifest_path, "val", label_to_idx, Path(args.feature_dir), augment=False)
    if not train_ds or not val_ds:
        raise SystemExit(
            "No usable hand-landmark features for training/validation "
            f"({len(train_ds)} train, {len(val_ds)} val)."
        )

    train_labels = [label_to_idx[row["sign_id"]] for row in train_ds.items]
    class_counts = Counter(train_labels)
    sample_weights = [1.0 / max(class_counts[label], 1) for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ds), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"Dataset: {len(train_ds)} train, {len(val_ds)} val, {len(sign_ids)} classes")
    dropped = train_ds.missing_features_count + val_ds.missing_features_count
    if dropped:
        print(
            "Dropped clips without extracted hand-landmark features: "
            f"{dropped} ({train_ds.missing_features_count} train, {val_ds.missing_features_count} val)"
        )
    if train_ds.low_coverage_count:
        print(
            "Dropped low-coverage training clips: "
            f"{train_ds.low_coverage_count} below {args.min_feature_coverage:.0%} hand-tracking coverage"
        )
    print(f"Input: hand_landmarks ({HAND_FEATURES} features x {NUM_FRAMES} frames)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_landmark_model(num_classes=len(sign_ids)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {n_params/1e6:.2f}M ({n_params:,})")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(opt, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    ckpt_dir = ML_ROOT / "checkpoints" / args.model_version
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0
    best_distinct_preds = 0
    epochs_since_improvement = 0
    val_acc_history: list[float] = []

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} lr={opt.param_groups[0]['lr']:.2e}"):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            opt.step()
            total_loss += loss.item()
        scheduler.step()

        model.eval()
        correct = total = 0
        pred_counts: dict[int, int] = {}
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(1)
                correct += (pred == y).sum().item()
                total += y.numel()
                for p in pred.tolist():
                    pred_counts[p] = pred_counts.get(p, 0) + 1
        acc = correct / max(total, 1)
        distinct_preds = len(pred_counts)
        val_acc_history.append(acc)
        print(
            f"Epoch {epoch + 1}: loss={total_loss / max(len(train_loader), 1):.4f} "
            f"val_acc={acc:.4f} distinct_preds={distinct_preds}/{len(sign_ids)}"
        )

        improved = acc > best_acc
        tied_but_more_diverse = acc == best_acc and distinct_preds > best_distinct_preds
        if improved or tied_but_more_diverse:
            best_acc = acc
            best_distinct_preds = distinct_preds
            epochs_since_improvement = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "label_to_idx": label_to_idx,
                    "sign_ids": sign_ids,
                    "model_version": args.model_version,
                    "model_size": "hand_landmark_tcn",
                    "input_type": "hand_landmarks",
                    "num_frames": NUM_FRAMES,
                    "n_features": HAND_FEATURES,
                    "val_accuracy": acc,
                    "val_distinct_preds": distinct_preds,
                    "val_acc_history": val_acc_history,
                    "learning_rate": args.lr,
                    "weight_decay": args.weight_decay,
                    "label_smoothing": args.label_smoothing,
                    "max_grad_norm": args.max_grad_norm,
                    "pretrained_detector": "mediapipe_hands",
                    "min_feature_coverage": args.min_feature_coverage,
                    "landmark_noise_std": args.landmark_noise_std,
                    "landmark_dropout_prob": args.landmark_dropout_prob,
                },
                ckpt_dir / "best.pt",
            )
        else:
            epochs_since_improvement += 1
            if args.early_stop_patience > 0 and epochs_since_improvement >= args.early_stop_patience:
                print(f"Early stopping at epoch {epoch + 1}: best={best_acc:.4f}")
                break

    exports = ML_ROOT / "exports"
    exports.mkdir(parents=True, exist_ok=True)
    with open(exports / "model_meta.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_version": args.model_version,
                "model_size": "hand_landmark_tcn",
                "input_type": "hand_landmarks",
                "num_classes": len(sign_ids),
                "sign_ids": sign_ids,
                "val_accuracy": best_acc,
                "val_distinct_preds": best_distinct_preds,
                "val_acc_history": val_acc_history,
                "n_features": HAND_FEATURES,
                "num_frames": NUM_FRAMES,
                "params": n_params,
                "pretrained_detector": "mediapipe_hands",
                "min_feature_coverage": args.min_feature_coverage,
                "landmark_noise_std": args.landmark_noise_std,
                "landmark_dropout_prob": args.landmark_dropout_prob,
            },
            f,
            indent=2,
        )
    print(f"Best val acc: {best_acc:.4f} -> {ckpt_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
