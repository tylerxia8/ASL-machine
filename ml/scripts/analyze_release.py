"""Post-training analysis: fetch a Release's eval_metrics.json and produce a
focused report covering accuracy, confusable-pair behavior, and weak signs.

Run after a `Train Wave 1` workflow completes. Writes a markdown summary
suitable for pasting into VALIDATION_REPORT.md narrative sections.

Usage:
    # Latest release of the configured repo
    python ml/scripts/analyze_release.py
    # Specific tag
    python ml/scripts/analyze_release.py --tag wave1-semlex-full-v1
    # Specific repo (defaults to tylerxia8/ASL-machine)
    python ml/scripts/analyze_release.py --repo someone/their-fork
    # Local file (no download — useful right after a local eval.py run)
    python ml/scripts/analyze_release.py --metrics-file ml/exports/eval_metrics.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPO = "tylerxia8/ASL-machine"
DOCS_DIR = ROOT / "docs"

# Hand-curated pairs that VALIDATION_REPORT.md flagged as visually-similar
# within the 25-sign Wave 1 roster. Tracking model behavior on these directly
# tells us whether the model learned the distinguishing feature.
CONFUSABLE_PAIRS: list[tuple[str, str]] = [
    ("please", "sorry"),    # both chest circles; differ only in handshape
    ("what", "where"),      # both shaking; differ in handshape
    ("four", "five"),       # only thumb position differs
    ("one", "two"),         # static handshapes; finger precision
    ("two", "three"),       # adjacent static handshapes
    ("eat", "drink"),       # both at mouth (drink not in trained set; included for awareness)
    ("hello", "goodbye"),   # both head-level waves/salutes
]


def _fetch_latest_metrics(repo: str, tag: str | None) -> dict:
    """Download eval_metrics.json from a GitHub Release (latest or a specific tag)."""
    api_base = f"https://api.github.com/repos/{repo}/releases"
    api_url = f"{api_base}/tags/{tag}" if tag else f"{api_base}/latest"
    req = urllib.request.Request(api_url, headers={"Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req) as r:
        release = json.loads(r.read())
    print(f"Release: {release['tag_name']} (published {release['published_at']})")
    asset = next((a for a in release["assets"] if a["name"] == "eval_metrics.json"), None)
    if not asset:
        raise SystemExit(f"Release {release['tag_name']} has no eval_metrics.json asset.")
    print(f"Downloading {asset['name']} ({asset['size']/1024:.1f} KB)...")
    with urllib.request.urlopen(asset["browser_download_url"]) as r:
        return json.loads(r.read())


def _confusion(cm: list[list[int]], sign_ids: list[str], a: str, b: str) -> dict:
    """How often did the model predict `a` as `b` and vice-versa?"""
    out = {"a": a, "b": b, "a_in_data": a in sign_ids, "b_in_data": b in sign_ids}
    if not (a in sign_ids and b in sign_ids):
        return out
    ia, ib = sign_ids.index(a), sign_ids.index(b)
    support_a = sum(cm[ia])
    support_b = sum(cm[ib])
    correct_a = cm[ia][ia]
    correct_b = cm[ib][ib]
    a_as_b = cm[ia][ib]
    b_as_a = cm[ib][ia]
    out.update({
        "support_a": support_a, "support_b": support_b,
        "correct_a": correct_a, "correct_b": correct_b,
        "a_as_b": a_as_b, "b_as_a": b_as_a,
        "recall_a": (correct_a / support_a) if support_a else 0.0,
        "recall_b": (correct_b / support_b) if support_b else 0.0,
        "confusion_rate": ((a_as_b + b_as_a) / (support_a + support_b)) if (support_a + support_b) else 0.0,
    })
    return out


def _weak_signs(report: dict, sign_ids: list[str], top_n: int = 5) -> list[dict]:
    """Per-class metrics sorted from worst f1 to best, with support > 0."""
    rows = []
    for s in sign_ids:
        r = report.get(s)
        if not r or r["support"] == 0:
            continue
        rows.append({"sign": s, **r})
    rows.sort(key=lambda r: r["f1-score"])
    return rows


def _format_report(metrics: dict) -> str:
    accuracy = metrics.get("accuracy", 0.0)
    report = metrics.get("report", {})
    sign_ids = metrics.get("sign_ids", [])
    cm = metrics.get("confusion_matrix", [])
    model_version = metrics.get("model_version", "unknown")

    lines: list[str] = []
    lines.append(f"# Post-training analysis — `{model_version}`\n")
    lines.append(f"**Overall test accuracy:** {accuracy:.2%}  ")
    macro = report.get("macro avg", {})
    lines.append(f"**Macro F1:** {macro.get('f1-score', 0):.3f}  ")
    lines.append(f"**Total test clips:** {int(macro.get('support', 0))}  ")
    lines.append(f"**Classes with any test data:** "
                 f"{sum(1 for s in sign_ids if report.get(s, {}).get('support', 0) > 0)} / {len(sign_ids)}")
    lines.append("")

    # Weak/missing classes
    lines.append("## Weakest 5 classes by F1 (with non-empty test set)")
    lines.append("")
    weak = _weak_signs(report, sign_ids, top_n=5)[:5]
    if weak:
        lines.append("| Sign | Precision | Recall | F1 | Support |")
        lines.append("|---|---|---|---|---|")
        for r in weak:
            lines.append(f"| `{r['sign']}` | {r['precision']:.2f} | {r['recall']:.2f} | "
                         f"{r['f1-score']:.2f} | {int(r['support'])} |")
    else:
        lines.append("_All classes have zero test support — eval is uninterpretable._")
    lines.append("")

    # Classes with zero support — flagged separately as "no signal to evaluate"
    no_support = [s for s in sign_ids if report.get(s, {}).get("support", 0) == 0]
    if no_support:
        lines.append(f"## {len(no_support)} class(es) with zero test data")
        lines.append("")
        lines.append("These signs are in the trained roster but the signer-disjoint test split "
                     "happens to include no clips of them. Accuracy is unknown until the test "
                     "set grows or `--signer-disjoint` is rebalanced.")
        lines.append("")
        lines.append(", ".join(f"`{s}`" for s in no_support))
        lines.append("")

    # Confusable-pair analysis
    if cm and sign_ids:
        lines.append("## Confusable pairs (known a-priori from sign linguistics)")
        lines.append("")
        lines.append("| Pair | Recall A | Recall B | A↔B confusions | A+B confusion rate |")
        lines.append("|---|---|---|---|---|")
        for a, b in CONFUSABLE_PAIRS:
            c = _confusion(cm, sign_ids, a, b)
            if not (c.get("a_in_data") and c.get("b_in_data")):
                lines.append(f"| `{a}` vs `{b}` | _not in trained set_ | | | |")
                continue
            lines.append(
                f"| `{a}` vs `{b}` "
                f"| {c['recall_a']:.2f} (n={c['support_a']}) "
                f"| {c['recall_b']:.2f} (n={c['support_b']}) "
                f"| {c['a_as_b']}/{c['b_as_a']} "
                f"| {c['confusion_rate']:.2%} |"
            )
        lines.append("")

    # Top off-diagonal confusions (any pair, not just curated)
    if cm and sign_ids:
        lines.append("## Top 10 most-confused pairs (any, sorted by count)")
        lines.append("")
        confusions = []
        for i, s_true in enumerate(sign_ids):
            for j, s_pred in enumerate(sign_ids):
                if i != j and cm[i][j] > 0:
                    confusions.append((cm[i][j], s_true, s_pred))
        confusions.sort(reverse=True)
        if confusions:
            lines.append("| True → | Predicted | Count |")
            lines.append("|---|---|---|")
            for count, t, p in confusions[:10]:
                lines.append(f"| `{t}` | `{p}` | {count} |")
        else:
            lines.append("_No off-diagonal confusions (or empty test set)._")
        lines.append("")

    # Interpretation
    lines.append("## Interpretation")
    lines.append("")
    if accuracy < 0.20:
        lines.append("- Overall accuracy is **near-chance** for a 25-class problem (random = 4%). "
                     "Almost certainly insufficient training data per class, or test set is too "
                     "sparse to evaluate. Action: increase `semlex_clips_per_sign`, or train with "
                     "data augmentation enabled (already wired into `ml/dataset.py`).")
    elif accuracy < 0.50:
        lines.append("- Overall accuracy is **above chance but well below pilot-acceptable** "
                     "(target ≥ 0.50 for signer-disjoint top-1). Focus next iteration on the "
                     "weakest classes above — they often share a confusable pair listed below.")
    elif accuracy < 0.75:
        lines.append("- Overall accuracy is **pilot-acceptable**. Watch the confusable pairs "
                     "table — the rubric's threshold policy (pass ≥ 0.90, retry 0.70–0.90) "
                     "will route most low-confidence answers to retry, which is the correct "
                     "behavior for visually-similar signs.")
    else:
        lines.append("- Overall accuracy is **strong for a from-scratch CNN at this data scale**. "
                     "Document the controlled conditions and ship.")
    lines.append("")

    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", help="Release tag (default: latest). Ignored if --metrics-file is set.")
    p.add_argument("--repo", default=os.environ.get("GITHUB_REPO", DEFAULT_REPO))
    p.add_argument("--metrics-file", help="Path to a local eval_metrics.json (skip download).")
    p.add_argument("--out", help="Write report to this path. Default: print to stdout.")
    args = p.parse_args()

    if args.metrics_file:
        metrics = json.loads(Path(args.metrics_file).read_text(encoding="utf-8"))
    else:
        metrics = _fetch_latest_metrics(args.repo, args.tag)

    report = _format_report(metrics)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report, encoding="utf-8")
        print(f"Wrote {out_path}")
    else:
        print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
