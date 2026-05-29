from __future__ import annotations

import argparse
import json
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parents[1]
ROOT = ML_ROOT.parent

MANUAL_CONFUSIONS = {
    ("who", "where"): "WHO happens close to the chin. WHERE is an upright index finger shaking in neutral space.",
    ("who", "deaf"): "WHO stays near the chin in a small circle or tap. DEAF has two clear touch points near ear and mouth.",
    ("deaf", "where"): "DEAF needs two face touch points. WHERE is a small side-to-side index-finger shake away from the face.",
    ("deaf", "who"): "DEAF should travel between ear and mouth. WHO stays close to the chin.",
    ("where", "who"): "WHERE uses an upright index finger shaking in neutral space. WHO is closer to the chin.",
    ("where", "deaf"): "WHERE should not touch the face. Keep the finger upright and shake it in neutral space.",
    ("two", "three"): "TWO uses index and middle fingers. THREE uses thumb, index, and middle.",
    ("three", "two"): "THREE includes the thumb. TWO keeps the thumb closed.",
    ("four", "five"): "FOUR has the thumb tucked. FIVE spreads the thumb out.",
    ("five", "four"): "FIVE spreads all fingers including the thumb. FOUR tucks the thumb.",
    ("please", "sorry"): "PLEASE uses an open flat hand on the chest. SORRY uses a closed fist.",
    ("sorry", "please"): "SORRY uses a closed fist. PLEASE uses an open flat hand.",
    ("no", "yes"): "NO is a quick finger-close. YES is a fist nodding at the wrist.",
    ("yes", "no"): "YES is a fist nod. NO closes index and middle fingers to the thumb.",
    ("nice", "goodbye"): "NICE uses both hands with a palm slide. GOODBYE is a one-hand finger wave.",
    ("how", "help"): "HOW starts with curved hands together and rotates. HELP stacks an A-hand on a flat palm and lifts.",
}


def thresholds_for(f1: float, support: int) -> tuple[float, float]:
    if support < 5 or f1 < 0.5:
        return 0.95, 0.82
    if f1 < 0.7:
        return 0.93, 0.78
    if f1 < 0.85:
        return 0.90, 0.70
    return 0.86, 0.62


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", default=str(ML_ROOT / "exports" / "eval_metrics.json"))
    parser.add_argument("--out", default=str(ML_ROOT / "exports" / "recognition_calibration.json"))
    args = parser.parse_args()

    metrics = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    sign_ids = metrics["sign_ids"]
    report = metrics.get("report", {})
    cm = metrics.get("confusion_matrix", [])

    thresholds = {}
    for sign in sign_ids:
        row = report.get(sign, {})
        f1 = float(row.get("f1-score", 0.0))
        support = int(row.get("support", 0))
        pass_threshold, retry_threshold = thresholds_for(f1, support)
        thresholds[sign] = {
            "passThreshold": pass_threshold,
            "retryThreshold": retry_threshold,
            "f1": f1,
            "support": support,
        }

    confusions = {}
    for i, true_label in enumerate(sign_ids):
        pairs = []
        for j, predicted_label in enumerate(sign_ids):
            count = int(cm[i][j]) if i < len(cm) and j < len(cm[i]) else 0
            if i != j and count > 0:
                pairs.append((count, predicted_label))
        pairs.sort(reverse=True)
        for count, predicted_label in pairs[:3]:
            key = f"{true_label}->{predicted_label}"
            message = MANUAL_CONFUSIONS.get(
                (true_label, predicted_label),
                f"The model confused this with {predicted_label}. Emphasize the handshape, location, and movement difference before trying again.",
            )
            confusions[key] = {"count": count, "message": message}

    out = {
        "model_version": metrics.get("model_version"),
        "accuracy": metrics.get("accuracy"),
        "thresholds": thresholds,
        "confusions": confusions,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {out_path} ({len(thresholds)} thresholds, {len(confusions)} confusions)")


if __name__ == "__main__":
    main()
