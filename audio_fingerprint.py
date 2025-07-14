from __future__ import annotations
import asyncio, os
from typing import Optional, Tuple
import acoustid, musicbrainzngs            # new
from shazamio import Shazam                # fallback

# Tell the AcoustID library to use the local fpcalc.exe we just copied
os.environ.setdefault("FPCALC", os.path.join(os.path.dirname(__file__), "fpcalc.exe"))

ACOUSTID_KEY = "UT7pFXWpWG"  # free key from acoustid.org
musicbrainzngs.set_useragent("Mapperatorinator", "1.0")

async def _shazam(path: str) -> Tuple[Optional[str], Optional[str]]:
    out = await Shazam().recognize(path)
    track = out.get("track")
    if track:
        return track.get("subtitle"), track.get("title")
    return None, None

def _acoustid(path: str) -> tuple[Optional[str], Optional[str]]:
    """
    Try offline AcoustID → MusicBrainz lookup.
    Falls back to Shazam in `identify_song()` if no match or an error occurs.
    """
    if not ACOUSTID_KEY:
        return None, None

    try:
        fp, duration = acoustid.fingerprint_file(path)
        if isinstance(fp, (bytes, bytearray)):
            fp = fp.decode("ascii", "ignore")

        # NEW order: duration first, fingerprint second
        result = acoustid.lookup(ACOUSTID_KEY, duration, fp, meta="recordings")

        # Guard against API errors (no "results" key)
        if not isinstance(result, dict) or result.get("status") != "ok":
            err = result.get("error", {}).get("message", "unknown response")
            print(f"AcoustID lookup failed: {err}")
            return None, None

        for r in result.get("results", []):
            score = r.get("score", 0)
            for rec in r.get("recordings", []):
                artist = next((a["name"] for a in rec.get("artists", [])), None)
                title  = rec.get("title")
                if score > 0.6 and artist and title:
                    return artist, title

    except acoustid.AcoustidError as e:
        print("AcoustID error:", e)
    except Exception as e:
        print("AcoustID unexpected error:", e)

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

# if __name__ == "__main__":                         # temp‑debug block
#     import json, sys
#     path = sys.argv[1] if len(sys.argv) > 1 else None
#     print("Key in code →", repr(ACOUSTID_KEY))

#     # just hit the /version endpoint – lightweight and needs only the key
#     import urllib.request, urllib.parse, urllib.error

#     qs = urllib.parse.urlencode({"client": ACOUSTID_KEY})
#     try:
#         with urllib.request.urlopen(f"https://api.acoustid.org/v2/version?{qs}", timeout=10) as r:
#             print("HTTP", r.status)
#             print(json.load(r))
#     except urllib.error.HTTPError as e:
#         print("HTTP", e.code)
#         print(e.read().decode())
#     sys.exit(0)
