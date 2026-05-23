from datetime import datetime

from pydantic import BaseModel, Field


class AttemptCreate(BaseModel):
    sign_id: str
    outcome: str = Field(pattern="^(pass|fail|retry)$")
    confidence: float | None = None
    predicted_label: str | None = None
    session_id: str | None = None


class AttemptOut(BaseModel):
    id: str
    sign_id: str
    outcome: str
    confidence: float | None
    predicted_label: str | None
    attempt_number: int
    created_at: datetime

    class Config:
        from_attributes = True


class SessionCreate(BaseModel):
    pass


class SessionOut(BaseModel):
    id: str
    started_at: datetime
    signs_practiced: int
    signs_passed: int

    class Config:
        from_attributes = True


class MasteryOut(BaseModel):
    sign_id: str
    mastered: bool
    consecutive_passes: int
    total_attempts: int
    total_passes: int
    last_practiced_at: datetime | None

    class Config:
        from_attributes = True


class ProgressSummary(BaseModel):
    total_attempts: int
    total_passes: int
    mastered_count: int
    recent_attempts: list[AttemptOut]


class SignMeta(BaseModel):
    sign_id: str
    gloss: str
    category: str
    unit: str
    trained: bool = False


class HintOut(BaseModel):
    sign_id: str
    gloss: str
    message: str
    handshape: str
    movement: str
    location: str
    framing: str
