from fastapi import Header, HTTPException

from config import settings


async def get_user_id(authorization: str | None = Header(default=None), x_user_id: str | None = Header(default=None)) -> str:
    """Resolve user from Supabase JWT or dev header."""
    if x_user_id:
        return x_user_id
    if authorization and authorization.startswith("Bearer ") and settings.supabase_jwt_secret:
        token = authorization.split(" ", 1)[1]
        try:
            from jose import jwt

            payload = jwt.decode(token, settings.supabase_jwt_secret, algorithms=["HS256"], audience="authenticated")
            sub = payload.get("sub")
            if sub:
                return sub
        except Exception as e:
            raise HTTPException(status_code=401, detail="Invalid token") from e
    return settings.dev_user_id
