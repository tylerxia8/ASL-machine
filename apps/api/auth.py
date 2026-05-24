"""User-id resolution for API requests.

Trust model (in priority order):
1. **Authorization: Bearer <JWT>** — verified against `SUPABASE_JWT_SECRET`.
   When the secret is configured AND a Bearer header is present, the JWT
   MUST validate; otherwise the request is rejected with 401. This is the
   production path.
2. **X-User-Id: <id>** — accepted blindly (no signature). This is the
   dev/closed-pilot path. The pilot's `CONTROLLED_CONDITIONS` doc notes
   that for production deployments, the frontend should stop sending
   X-User-Id so the JWT path is the only one available.
3. **`settings.dev_user_id`** — last-resort fallback so dev sessions still
   work even when the frontend forgets both headers.
"""
from fastapi import Header, HTTPException

from config import settings


async def get_user_id(
    authorization: str | None = Header(default=None),
    x_user_id: str | None = Header(default=None),
) -> str:
    if authorization and authorization.startswith("Bearer "):
        if not settings.supabase_jwt_secret:
            # JWT was sent but server isn't configured to verify it.
            # In closed-pilot we fall through to X-User-Id; in production
            # the secret MUST be set so this path is unreachable.
            pass
        else:
            token = authorization.split(" ", 1)[1]
            try:
                from jose import jwt

                payload = jwt.decode(
                    token,
                    settings.supabase_jwt_secret,
                    algorithms=["HS256"],
                    audience="authenticated",
                )
            except Exception as e:
                raise HTTPException(status_code=401, detail="Invalid token") from e
            sub = payload.get("sub")
            if not sub:
                raise HTTPException(status_code=401, detail="Token missing subject")
            return sub

    if x_user_id:
        return x_user_id

    return settings.dev_user_id
