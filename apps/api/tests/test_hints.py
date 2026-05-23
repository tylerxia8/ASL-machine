import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HINTS_DIR = ROOT / "content" / "hints"


def test_hints_cover_vocabulary():
    vocab = (ROOT / "content" / "vocabulary.csv").read_text(encoding="utf-8").strip().split("\n")[1:]
    sign_ids = [line.split(",")[0] for line in vocab]
    index_path = HINTS_DIR / "_index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        missing = [s for s in sign_ids if s not in index]
    else:
        missing = [s for s in sign_ids if not (HINTS_DIR / f"{s}.json").exists()]
    assert not missing, f"Missing hints: {missing[:5]}"
