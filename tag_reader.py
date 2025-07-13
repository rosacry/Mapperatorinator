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
        if not audio:
            return None, None
        artist = (audio.get("artist") or [None])[0]
        title  = (audio.get("title")  or [None])[0]
        return artist, title
    except Exception as e:
        print("Mutagen error:", e)
        return None, None
