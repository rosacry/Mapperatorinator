"""
mapper_api.py – helper for osu! API username lookup
"""
from __future__ import annotations
import os, time, requests, threading
from typing import Optional

_OSU_TOKEN: str | None = None          # cached bearer
_TOKEN_EXPIRES: float = 0.0            # unix time in s
_LOCK = threading.Lock()

def _refresh_token() -> str:
    """
    Fetch a client-credentials token and cache it.
    Environment vars:
        OSU_CLIENT_ID     – e.g. 42371
        OSU_CLIENT_SECRET – your secret
    """
    global _OSU_TOKEN, _TOKEN_EXPIRES
    cid  = os.getenv("OSU_CLIENT_ID")
    sec  = os.getenv("OSU_CLIENT_SECRET")
    if not cid or not sec:
        raise RuntimeError("osu! API credentials not set")
    r = requests.post(
        "https://osu.ppy.sh/oauth/token",
        json={
            "client_id": int(cid),
            "client_secret": sec,
            "grant_type": "client_credentials",
            "scope": "public",
        },
        timeout=10,
    )
    r.raise_for_status()
    data = r.json()
    _OSU_TOKEN      = data["access_token"]
    _TOKEN_EXPIRES  = time.time() + data["expires_in"] - 60   # refresh 1 min early
    return _OSU_TOKEN

def _get_token() -> str:
    with _LOCK:
        if _OSU_TOKEN is None or time.time() >= _TOKEN_EXPIRES:
            return _refresh_token()
        return _OSU_TOKEN

def lookup_username(mapper_id: int | str) -> Optional[str]:
    """Return mapper’s current username or None if not found."""
    try:
        token = _get_token()
    except Exception as e:
        print("osu! token error:", e)
        return None
    url = f"https://osu.ppy.sh/api/v2/users/{mapper_id}/osu"
    r = requests.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=10)
    if r.status_code != 200:
        print("osu! lookup failed:", r.status_code, r.text)
        return None
    return r.json().get("username")
