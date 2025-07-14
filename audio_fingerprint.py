from __future__ import annotations
import asyncio, os
from typing import Optional, Tuple
import acoustid, musicbrainzngs            # new
from shazamio import Shazam                # fallback

ACOUSTID_KEY = os.getenv("ACOUSTID_API_KEY")  # free key from acoustid.org
musicbrainzngs.set_useragent("Mapperatorinator", "1.0")

async def _shazam(path: str) -> Tuple[Optional[str], Optional[str]]:
    out = await Shazam().recognize(path)
    track = out.get("track")
    if track:
        return track.get("subtitle"), track.get("title")
    return None, None

def _acoustid(path: str) -> Tuple[Optional[str], Optional[str]]:
    if not ACOUSTID_KEY:
        return None, None
    try:
        code, duration = acoustid.fingerprint_file(path)
        res = acoustid.lookup(ACOUSTID_KEY, code, duration, meta="recordings")
        for score, rid, title, artist in (
            (r["score"],
             rec["id"],
             rec["title"],
             next((a["name"] for a in rec["artists"]), None))
            for r in res["results"]
            for rec in r.get("recordings", [])
        ):
            if score > 0.6:
                return artist, title
    except acoustid.AcoustidError as e:
        print("AcoustID error:", e)
    return None, None

def identify_song(path: str) -> Tuple[Optional[str], Optional[str]]:
    # 1️⃣ offline / AcoustID first
    artist, title = _acoustid(path)
    if artist and title:
        return artist, title
    # 2️⃣ fallback to Shazam (async helper)
    try:
        return asyncio.run(_shazam(path))
    except Exception as e:
        print("Shazam lookup failed:", e)
        return None, None
