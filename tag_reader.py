"""
tag_reader.py – tiny wrapper around Mutagen for artist/title tags
"""
from __future__ import annotations
from typing import Tuple, Optional
from mutagen import File as MutagenFile

def read_artist_title(audio_path: str) -> Tuple[Optional[str], Optional[str]]:
    """Returns (artist, title) or (None, None) if unreadable/absent."""
    try:
        audio = MutagenFile(audio_path, easy=True)
        artist = title = None
        if audio:
            artist = (audio.get("artist") or [None])[0]
            title  = (audio.get("title")  or [None])[0]

        # ── Fallback: guess from filename / folder ───────────────────────
        if not title:
            import os
            base = os.path.splitext(os.path.basename(audio_path))[0]
            if base.lower().startswith("audio"):
                base = os.path.basename(os.path.dirname(audio_path))
            title = base

        if not artist and " - " in title:
            artist, title = [p.strip() for p in title.split(" - ", 1)]

        return artist, title
    except Exception as e:
        print("Mutagen error:", e)
        return None, None
