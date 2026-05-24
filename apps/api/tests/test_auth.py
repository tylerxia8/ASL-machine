"""Verify the user-id resolution priority order.

JWT (when configured) > X-User-Id > dev fallback.
"""
import asyncio
import sys
from pathlib import Path

import pytest
from fastapi import HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from auth import get_user_id
from config import settings


def _run(authorization=None, x_user_id=None):
    return asyncio.run(get_user_id(authorization=authorization, x_user_id=x_user_id))


def test_falls_back_to_dev_user_when_no_headers():
    assert _run() == settings.dev_user_id


def test_uses_x_user_id_when_no_jwt():
    assert _run(x_user_id="alice-123") == "alice-123"


def test_jwt_takes_priority_when_configured(monkeypatch):
    secret = "test-secret-value-32-chars-long-XX"
    monkeypatch.setattr(settings, "supabase_jwt_secret", secret)
    from jose import jwt
    token = jwt.encode({"sub": "supabase-uid", "aud": "authenticated"}, secret, algorithm="HS256")
    # Even if X-User-Id says "imposter", the JWT's sub wins.
    assert _run(authorization=f"Bearer {token}", x_user_id="imposter") == "supabase-uid"


def test_invalid_jwt_raises_401(monkeypatch):
    monkeypatch.setattr(settings, "supabase_jwt_secret", "test-secret")
    with pytest.raises(HTTPException) as exc:
        _run(authorization="Bearer notatoken.notvalid.atall", x_user_id="alice")
    assert exc.value.status_code == 401


def test_jwt_without_secret_falls_through_to_x_user_id(monkeypatch):
    # If server isn't configured to verify, X-User-Id is still honored (dev/closed-pilot mode).
    monkeypatch.setattr(settings, "supabase_jwt_secret", "")
    assert _run(authorization="Bearer anything", x_user_id="alice-123") == "alice-123"
