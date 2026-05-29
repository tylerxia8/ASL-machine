"""Extract MediaPipe hand-landmark features for clips in manifest.json.

This deliberately uses a pretrained detector. It is for the product-quality
model path, not the original rubric-strict no-pretrained path.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml"))

from clip_io import load_clip  # noqa: E402
from landmark_dataset import HAND_FEATURES, NUM_FRAMES, feature_path_for_clip  # noqa: E402

HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
DEFAULT_MODEL_PATH = ROOT / "ml" / "data" / "mediapipe" / "hand_landmarker.task"


def _sample_indices(length: int, target: int) -> np.ndarray:
    if length <= 0:
        raise ValueError("Cannot sample from empty frame list")
    return np.linspace(0, length - 1, target).astype(int)


def _load_video_frames(path: Path, target: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise ValueError(f"No decodable frames in {path}")
    return [frames[i] for i in _sample_indices(len(frames), target)]


def _load_clip_frames(path: Path, target: int) -> list[np.ndarray]:
    frames = load_clip(path)
    idx = _sample_indices(frames.shape[0], target)
    out = np.clip(frames[idx] * 255.0, 0, 255).astype(np.uint8)
    return [frame for frame in out]


def _clip_source_path(clip_path: Path) -> Path | None:
    try:
        data = np.load(clip_path)
        source = data.get("source_path")
        if source is None:
            return None
        return ROOT / str(source)
    except Exception:
        return None


def _landmark_list(landmarks) -> list:
    if hasattr(landmarks, "landmark"):
        return list(landmarks.landmark)
    return list(landmarks)


def _hand_vector(landmarks) -> np.ndarray:
    arr = np.array([[lm.x, lm.y, lm.z] for lm in _landmark_list(landmarks)], dtype=np.float32)
    wrist = arr[0].copy()
    rel = arr - wrist
    return np.concatenate([wrist, rel.reshape(-1)], axis=0)


def _features_for_result(result) -> np.ndarray:
    out = np.zeros((HAND_FEATURES,), dtype=np.float32)
    hand_landmarks = getattr(result, "multi_hand_landmarks", None)
    handedness = getattr(result, "multi_handedness", None)
    if hand_landmarks is None:
        hand_landmarks = getattr(result, "hand_landmarks", None)
        handedness = getattr(result, "handedness", None)
    if not hand_landmarks:
        return out

    slots: dict[str, np.ndarray] = {}
    for idx, landmarks in enumerate(hand_landmarks):
        label = ""
        if handedness and idx < len(handedness):
            entry = handedness[idx]
            if isinstance(entry, (list, tuple)):
                first = entry[0]
            elif hasattr(entry, "classification"):
                first = entry.classification[0]
            else:
                first = entry
            label = getattr(first, "category_name", None) or getattr(first, "label", "")
            label = label.lower()
        vec = _hand_vector(landmarks)
        if label.startswith("left"):
            slots["left"] = vec
        elif label.startswith("right"):
            slots["right"] = vec
        else:
            # Fallback: lower wrist x is treated as left image slot.
            key = "left" if vec[0] < 0.5 else "right"
            slots[key] = vec

    if "left" in slots:
        out[:66] = slots["left"]
    if "right" in slots:
        out[66:] = slots["right"]
    return out


def _ensure_model(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size == 0:
        print(f"Downloading MediaPipe hand landmarker model -> {path}")
        urllib.request.urlretrieve(HAND_MODEL_URL, path)
    return path


def create_landmarker(model_path: Path, min_detection_confidence: float):
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=min_detection_confidence,
        min_hand_presence_confidence=0.25,
        min_tracking_confidence=0.30,
    )
    return mp, vision.HandLandmarker.create_from_options(options)


def extract_features(frames: list[np.ndarray], mp, landmarker, start_frame: int = 0) -> np.ndarray:
    features = np.zeros((NUM_FRAMES, HAND_FEATURES), dtype=np.float32)
    for i, frame in enumerate(frames):
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame))
        result = landmarker.detect_for_video(image, (start_frame + i) * 1000 // 12)
        features[i] = _features_for_result(result)
    return features


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(ROOT / "ml" / "data" / "manifest.json"))
    parser.add_argument("--out-dir", default=str(ROOT / "ml" / "data" / "hand_landmarks"))
    parser.add_argument("--prefer-source-video", action="store_true", default=True)
    parser.add_argument("--min-detection-confidence", type=float, default=0.25)
    parser.add_argument("--hand-model", default=str(DEFAULT_MODEL_PATH))
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    failures: list[tuple[str, str]] = []
    detected_frames = 0
    total_frames = 0
    model_path = _ensure_model(Path(args.hand_model))
    mp, landmarker = create_landmarker(model_path, args.min_detection_confidence)
    with landmarker:
        for clip_index, row in enumerate(tqdm(manifest["clips"], desc="Extract hand landmarks")):
            clip_path = ROOT / row["path"]
            out_path = feature_path_for_clip(row["path"], out_dir)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                source = _clip_source_path(clip_path) if args.prefer_source_video else None
                if source is not None and source.exists():
                    frames = _load_video_frames(source, NUM_FRAMES)
                    source_kind = "source_video"
                else:
                    frames = _load_clip_frames(clip_path, NUM_FRAMES)
                    source_kind = "imported_clip"
                features = extract_features(frames, mp, landmarker, start_frame=clip_index * NUM_FRAMES)
                present = np.abs(features).sum(axis=1) > 0
                detected_frames += int(present.sum())
                total_frames += int(features.shape[0])
                with open(out_path, "wb") as f:
                    np.savez_compressed(
                        f,
                        features=features,
                        sign_id=row["sign_id"],
                        signer_id=row["signer_id"],
                        split=row["split"],
                        source_kind=source_kind,
                        detected_frame_count=int(present.sum()),
                    )
            except Exception as exc:
                failures.append((row["path"], str(exc)))

    coverage = detected_frames / max(total_frames, 1)
    print(f"Hand landmark frame coverage: {detected_frames}/{total_frames} ({coverage:.1%})")
    if failures:
        print(f"Failures: {len(failures)}", file=sys.stderr)
        for path, msg in failures[:20]:
            print(f"  {path}: {msg}", file=sys.stderr)
    if coverage < 0.20:
        print("FATAL: landmark detection coverage is too low for training.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
