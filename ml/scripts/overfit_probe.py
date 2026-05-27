"""Tiny memorization probe for the ASL training stack.

This is a diagnostic, not a pilot metric. It answers one narrow question:
can the current model/training code memorize a few real clips per sign when
augmentation and signer-disjoint generalization are removed?

If this fails on real Sem-Lex clips, inspect data/label alignment before
running more full CI training jobs. If it passes but validation stays poor,
the blocker is generalization/modeling rather than basic pipeline wiring.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml"))

from dataset import SignClipDataset  # noqa: E402
from model import build_model  # noqa: E402


def _disable_dropout(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            module.p = 0.0


def _choose_subset(manifest: dict, split: str, samples_per_class: int,
                   max_classes: int | None) -> tuple[list[str], list[int]]:
    rows = [row for row in manifest["clips"] if row["split"] == split]
    by_sign: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        by_sign[row["sign_id"]].append(idx)

    sign_ids = [
        sign_id for sign_id in sorted(by_sign)
        if len(by_sign[sign_id]) >= samples_per_class
    ]
    if max_classes is not None:
        sign_ids = sign_ids[:max_classes]
    if not sign_ids:
        raise SystemExit(
            f"No classes in split={split!r} have at least {samples_per_class} clips."
        )

    indices: list[int] = []
    for sign_id in sign_ids:
        indices.extend(by_sign[sign_id][:samples_per_class])
    return sign_ids, indices


def _accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct / max(total, 1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(ROOT / "ml" / "data" / "manifest.json"))
    parser.add_argument("--split", default="train")
    parser.add_argument("--samples-per-class", type=int, default=3)
    parser.add_argument("--max-classes", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--model-size", choices=["default", "small", "frame"], default="small")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--pass-threshold", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--disable-dropout", action="store_true",
                        help="Set Dropout probabilities to 0 for a purer memorization check.")
    parser.add_argument("--no-fail", action="store_true",
                        help="Always exit 0; useful when recording diagnostics in CI logs.")
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    manifest_path = Path(args.manifest)
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    sign_ids, subset_indices = _choose_subset(
        manifest, args.split, args.samples_per_class, args.max_classes
    )
    label_to_idx = {sign_id: i for i, sign_id in enumerate(sign_ids)}
    base_ds = SignClipDataset(manifest_path, args.split, label_to_idx, augment=False)
    ds = Subset(base_ds, subset_indices)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    eval_loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=len(sign_ids), size=args.model_size).to(device)
    if args.disable_dropout:
        _disable_dropout(model)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(
        f"Overfit probe: {len(ds)} clips, {len(sign_ids)} classes, "
        f"{args.samples_per_class} clips/class, model={args.model_size}, device={device}"
    )
    print(f"Signs: {', '.join(sign_ids)}")

    last_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            opt.step()
            total_loss += loss.item()
        last_acc = _accuracy(model, eval_loader, device)
        print(
            f"Epoch {epoch:02d}: loss={total_loss / max(len(loader), 1):.4f} "
            f"memorization_acc={last_acc:.3f}"
        )
        if last_acc >= args.pass_threshold:
            break

    passed = last_acc >= args.pass_threshold
    print(
        f"OVERFIT_PROBE {'PASS' if passed else 'FAIL'}: "
        f"memorization_acc={last_acc:.3f}, threshold={args.pass_threshold:.3f}"
    )
    return 0 if passed or args.no_fail else 1


if __name__ == "__main__":
    raise SystemExit(main())
