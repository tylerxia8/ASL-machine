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
    parser.add_argument("--model-size", choices=["default", "small", "frame", "tcn", "motion_tcn"], default="default",
                        help="'small' (~720K params) for low-data regimes (<2k clips); "
                             "'frame' (~160K params) for frame-wise temporal pooling experiments; "
                             "'tcn' (~220K params) for frame-wise + temporal Conv1d experiments; "
                             "'motion_tcn' (~390K params) for dual-stream RGB + frame-difference temporal experiments; "
                             "'default' (~3.57M params) otherwise.")
    parser.add_argument("--early-stop-patience", type=int, default=4,
                        help="Stop early if val_acc doesn't improve for this many epochs. "
                             "Set to 0 to disable.")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="Adam weight decay — helps fight mode collapse.")
    parser.add_argument("--label-smoothing", type=float, default=0.05,
                        help="Small cross-entropy label smoothing to reduce early overconfidence.")
    parser.add_argument("--max-grad-norm", type=float, default=1.0,
                        help="Clip gradient norm after backprop. Set to 0 to disable.")
    parser.add_argument("--balanced-sampling", action="store_true", default=True,
                        help="Use WeightedRandomSampler so every batch has roughly uniform class "
                             "representation. Defends against mode collapse when class counts "
                             "are unequal (some Wave 1 signs cap at 100, others at 1).")
    parser.add_argument("--no-balanced-sampling", dest="balanced_sampling", action="store_false")
    parser.add_argument("--preprocess", choices=["center_crop", "letterbox"], default="center_crop",
                        help="Video resize/preprocess mode used to create the training clips. "
                             "Stored in checkpoint/export metadata for browser inference.")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    sign_ids = sorted(set(manifest.get("sign_ids", [])) or {c["sign_id"] for c in manifest["clips"]})
    label_to_idx = {s: i for i, s in enumerate(sign_ids)}

    train_ds = SignClipDataset(manifest_path, "train", label_to_idx)
    val_ds = SignClipDataset(manifest_path, "val", label_to_idx)

    # Class-balanced sampling: when class counts are unequal (Sem-Lex coverage
    # ranges from 1 to ~600 per sign in the wave1 roster), uniform per-clip
    # sampling produces batches dominated by the over-represented classes,
    # which has historically driven mode collapse in v5/v6/v7. Computing
    # per-sample weights = 1/N_class makes every class equally likely to
    # appear in each batch, regardless of total class count.
    if args.balanced_sampling:
        train_labels = [label_to_idx[row["sign_id"]] for row in train_ds.items]
        class_counts = Counter(train_labels)
        n_classes = len(sign_ids)
        # Show the imbalance before training so it's clear in CI logs.
        cnts = [class_counts.get(i, 0) for i in range(n_classes)]
        print(f"Train class counts: min={min(cnts)} max={max(cnts)} "
              f"mean={sum(cnts)/max(n_classes,1):.1f} zeros={sum(1 for c in cnts if c == 0)}")
        # Per-sample weights — inverse class frequency. Zero-count classes get
        # weight 0 (they have no samples anyway; the sampler never picks them).
        sample_weights = [1.0 / max(class_counts[lbl], 1) for lbl in train_labels]
        # num_samples = len(train_ds) keeps each "epoch" roughly the same length
        # as before, but with class-balanced composition rather than count-weighted.
        sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(train_ds), replacement=True
        )
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=0
        )
        print(f"Using WeightedRandomSampler (class-balanced batches)")
    else:
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0
        )
        print("Using uniform per-clip shuffle (no class balancing)")
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"Dataset: {len(train_ds)} train, {len(val_ds)} val, {len(sign_ids)} classes")
    print(f"Model: {args.model_size} (build_model size='{args.model_size}')")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=len(sign_ids), size=args.model_size).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {n_params/1e6:.2f}M  ({n_params:,})")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Cosine annealing decays LR from args.lr down to 0 over args.epochs.
    # Smooth decay helps avoid the late-epoch mode collapse observed in v5/v6.
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
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} lr={opt.param_groups[0]['lr']:.2e}"):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            opt.step()
            total_loss += loss.item()
        scheduler.step()

        # Val accuracy + class-diversity probe: count distinct classes predicted.
        # A model in mode collapse predicts only 1-2 classes, even on a diverse val set.
        model.eval()
        correct = total = 0
        pred_class_counts: dict[int, int] = {}
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
                for p in pred.tolist():
                    pred_class_counts[p] = pred_class_counts.get(p, 0) + 1
        acc = correct / max(total, 1)
        n_distinct_preds = len(pred_class_counts)
        val_acc_history.append(acc)
        avg_loss = total_loss / max(len(train_loader), 1)
        collapse_flag = " ⚠ MODE-COLLAPSE" if n_distinct_preds <= 3 and total >= 10 else ""
        print(f"Epoch {epoch+1}: loss={avg_loss:.4f} val_acc={acc:.4f} "
              f"distinct_preds={n_distinct_preds}/{len(sign_ids)}{collapse_flag}")

        improved = acc > best_acc
        tied_but_more_diverse = acc == best_acc and n_distinct_preds > best_distinct_preds
        if improved or tied_but_more_diverse:
            best_acc = acc
            best_distinct_preds = n_distinct_preds
            epochs_since_improvement = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "label_to_idx": label_to_idx,
                    "sign_ids": sign_ids,
                    "model_version": args.model_version,
                    "model_size": args.model_size,
                    "val_accuracy": acc,
                    "val_distinct_preds": n_distinct_preds,
                    "val_acc_history": val_acc_history,
                    "learning_rate": args.lr,
                    "weight_decay": args.weight_decay,
                    "label_smoothing": args.label_smoothing,
                    "max_grad_norm": args.max_grad_norm,
                    "balanced_sampling": args.balanced_sampling,
                    "preprocess": args.preprocess,
                },
                ckpt_dir / "best.pt",
            )
        else:
            epochs_since_improvement += 1
            if args.early_stop_patience > 0 and epochs_since_improvement >= args.early_stop_patience:
                print(f"Early stopping at epoch {epoch+1}: "
                      f"val_acc hasn't improved for {epochs_since_improvement} epochs "
                      f"(best={best_acc:.4f}).")
                break

    meta = {
        "model_version": args.model_version,
        "model_size": args.model_size,
        "num_classes": len(sign_ids),
        "sign_ids": sign_ids,
        "val_accuracy": best_acc,
        "val_acc_history": val_acc_history,
        "val_distinct_preds": best_distinct_preds,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "label_smoothing": args.label_smoothing,
        "max_grad_norm": args.max_grad_norm,
        "balanced_sampling": args.balanced_sampling,
        "preprocess": args.preprocess,
        "params": n_params,
        "pretrained": False,
    }
    exports = ML_ROOT / "exports"
    exports.mkdir(parents=True, exist_ok=True)
    with open(exports / "model_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Best val acc: {best_acc:.4f} -> {ckpt_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
