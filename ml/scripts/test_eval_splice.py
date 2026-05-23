"""Smoke test: eval.py's splice logic preserves narrative around AUTO-METRICS markers."""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except AttributeError:
    pass

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml"))

from eval import AUTO_END, AUTO_START, _build_auto_block, _splice  # noqa: E402


def fake_classification_report() -> dict:
    return {
        "hello": {"precision": 0.95, "recall": 0.90, "f1-score": 0.92, "support": 20},
        "goodbye": {"precision": 0.80, "recall": 0.85, "f1-score": 0.82, "support": 20},
        "macro avg": {"precision": 0.88, "recall": 0.88, "f1-score": 0.87, "support": 40},
        "accuracy": 0.88,
    }


def main() -> int:
    import numpy as np

    sign_ids = ["hello", "goodbye"]
    ckpt = {"model_version": "test-v0", "val_accuracy": 0.91}
    cm = np.array([[18, 2], [3, 17]])

    auto_block = _build_auto_block(ckpt, sign_ids, 0.875, fake_classification_report(), cm)
    assert AUTO_START in auto_block
    assert AUTO_END in auto_block
    assert "hello" in auto_block
    assert "test-v0" in auto_block
    print(f"[1/4] _build_auto_block OK ({len(auto_block.splitlines())} lines)")

    narrative_before = (
        "# Validation Report — Wave 1\n\n"
        "## Pilot scope\n\nHand-written content here.\n\n"
        f"{AUTO_START}\n\n_old metrics that should be replaced_\n\n{AUTO_END}\n\n"
        "## Known limitations\n\nMore hand-written content.\n"
    )
    spliced = _splice(narrative_before, auto_block)
    assert "Hand-written content here." in spliced, "lost prefix narrative"
    assert "More hand-written content." in spliced, "lost suffix narrative"
    assert "_old metrics that should be replaced_" not in spliced, "old metrics survived"
    assert "test-v0" in spliced, "new metrics not inserted"
    print("[2/4] _splice preserves narrative when markers exist OK")

    # Running it again should be idempotent (still preserves narrative).
    spliced_twice = _splice(spliced, auto_block)
    assert spliced_twice == spliced, "splice is not idempotent"
    print("[3/4] _splice is idempotent OK")

    # If there are no markers, the auto block should be appended.
    no_markers = "# Validation Report\n\nLegacy file with no markers.\n"
    spliced_appended = _splice(no_markers, auto_block)
    assert "Legacy file with no markers." in spliced_appended
    assert AUTO_START in spliced_appended
    print("[4/4] _splice appends auto block when no markers OK")

    print("\nSPLICE PASS — eval.py auto-block insertion preserves narrative.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
