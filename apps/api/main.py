from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from auth import get_user_id
from config import settings
from database import get_db, init_db
from models import Attempt, PracticeSession, Profile, SignMastery
from schemas import (
    AttemptCreate,
    AttemptOut,
    HintOut,
    MasteryOut,
    ProgressSummary,
    SessionCreate,
    SessionOut,
    SignMeta,
)

ROOT = Path(__file__).resolve().parents[2]
CONTENT = ROOT / "content"

app = FastAPI(title="ASL Pilot API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_origins.split(",")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    init_db()


def ensure_profile(db: Session, user_id: str):
    if not db.get(Profile, user_id):
        db.add(Profile(id=user_id, display_name="Learner"))
        db.commit()


@app.get("/health")
def health():
    return {"status": "ok"}


def _read_sign_rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _trained_sign_ids() -> set[str]:
    return {r["sign_id"] for r in _read_sign_rows(CONTENT / "wave1_signs.csv")}


@app.get("/signs", response_model=list[SignMeta])
def list_signs(wave: int | None = None):
    trained = _trained_sign_ids()
    if wave == 1:
        rows = _read_sign_rows(CONTENT / "wave1_signs.csv")
    else:
        rows = _read_sign_rows(CONTENT / "vocabulary.csv")
        if wave is not None:
            rows = [r for r in rows if int(r.get("priority_wave", 1)) <= wave]
    return [
        SignMeta(
            sign_id=row["sign_id"],
            gloss=row["gloss"],
            category=row.get("category", ""),
            unit=row.get("unit", ""),
            trained=row["sign_id"] in trained,
        )
        for row in rows
    ]


@app.get("/signs/{sign_id}/hint", response_model=HintOut)
def get_hint(sign_id: str, reason: str = "fail"):
    path = CONTENT / "hints" / f"{sign_id}.json"
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        index_path = CONTENT / "hints" / "_index.json"
        if not index_path.exists():
            raise HTTPException(404, "Hint not found")
        index = json.loads(index_path.read_text(encoding="utf-8"))
        if sign_id not in index:
            raise HTTPException(404, "Hint not found")
        data = index[sign_id]
    if reason == "framing":
        message = data.get("framing", "Adjust camera framing per guide box.")
    elif reason == "confusion" and data.get("common_confusions"):
        message = data["common_confusions"]
    else:
        message = f"Handshape: {data['handshape']} Movement: {data['movement']}"
    return HintOut(
        sign_id=sign_id,
        gloss=data.get("gloss", sign_id),
        message=message,
        handshape=data["handshape"],
        movement=data["movement"],
        location=data["location"],
        framing=data["framing"],
    )


@app.post("/sessions", response_model=SessionOut)
def create_session(
    _: SessionCreate,
    user_id: str = Depends(get_user_id),
    db: Session = Depends(get_db),
):
    ensure_profile(db, user_id)
    session = PracticeSession(user_id=user_id)
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


@app.post("/attempts", response_model=AttemptOut)
def record_attempt(
    body: AttemptCreate,
    user_id: str = Depends(get_user_id),
    db: Session = Depends(get_db),
):
    ensure_profile(db, user_id)
    mastery = db.get(SignMastery, {"user_id": user_id, "sign_id": body.sign_id})
    if not mastery:
        mastery = SignMastery(user_id=user_id, sign_id=body.sign_id)
        db.add(mastery)
    mastery.total_attempts += 1
    mastery.last_practiced_at = datetime.utcnow()

    if body.outcome == "pass":
        mastery.total_passes += 1
        mastery.consecutive_passes += 1
        if mastery.consecutive_passes >= 2 or (body.confidence and body.confidence >= 0.95):
            mastery.mastered = True
    else:
        mastery.consecutive_passes = 0

    count = db.query(Attempt).filter(Attempt.user_id == user_id, Attempt.sign_id == body.sign_id).count()
    attempt = Attempt(
        user_id=user_id,
        session_id=body.session_id,
        sign_id=body.sign_id,
        outcome=body.outcome,
        confidence=body.confidence,
        predicted_label=body.predicted_label,
        attempt_number=count + 1,
    )
    db.add(attempt)
    if body.session_id:
        sess = db.get(PracticeSession, body.session_id)
        if sess:
            sess.signs_practiced += 1
            if body.outcome == "pass":
                sess.signs_passed += 1
    db.commit()
    db.refresh(attempt)
    return attempt


@app.get("/progress", response_model=ProgressSummary)
def get_progress(user_id: str = Depends(get_user_id), db: Session = Depends(get_db)):
    ensure_profile(db, user_id)
    attempts = (
        db.query(Attempt)
        .filter(Attempt.user_id == user_id)
        .order_by(Attempt.created_at.desc())
        .limit(20)
        .all()
    )
    masteries = db.query(SignMastery).filter(SignMastery.user_id == user_id).all()
    total_attempts = sum(m.total_attempts for m in masteries)
    total_passes = sum(m.total_passes for m in masteries)
    mastered_count = sum(1 for m in masteries if m.mastered)
    return ProgressSummary(
        total_attempts=total_attempts,
        total_passes=total_passes,
        mastered_count=mastered_count,
        recent_attempts=attempts,
    )


@app.get("/mastery", response_model=list[MasteryOut])
def list_mastery(user_id: str = Depends(get_user_id), db: Session = Depends(get_db)):
    ensure_profile(db, user_id)
    return db.query(SignMastery).filter(SignMastery.user_id == user_id).all()
