from pathlib import Path
import json

import import_captures


def test_reviewed_video_meta_reads_accepted_clip_manifest(tmp_path, monkeypatch):
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    monkeypatch.setattr(import_captures, "INCOMING", incoming)
    (incoming / "reviewed_capture_manifest_1.json").write_text(
        json.dumps(
            {
                "clips": [
                    {
                        "filename": "badname.webm",
                        "sign_id": "hello",
                        "signer_id": "signer_a",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    assert import_captures._reviewed_video_meta() == {"badname.webm": ("hello", "signer_a")}


def test_reviewed_video_meta_ignores_non_manifest_json(tmp_path, monkeypatch):
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    monkeypatch.setattr(import_captures, "INCOMING", incoming)
    (incoming / "legacy_clip.json").write_text(json.dumps({"frames": []}), encoding="utf-8")

    assert import_captures._reviewed_video_meta() is None
