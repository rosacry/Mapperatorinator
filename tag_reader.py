# tag_reader.py – offline tag reader + JP → romaji
from __future__ import annotations
from typing import Tuple, Optional
from mutagen import File as MutagenFile

# NEW ▶ JP‐>romaji converter (pykakasi is tiny & pure-py)
try:
    from pykakasi import kakasi
    _kks = kakasi(); _kks.setMode("J","a"); _kks.setMode("K","a"); _kks.setMode("H","a")
    _jp2romaji = _kks.getConverter().do
    _is_jp = lambda s: any("\u3040" <= ch <= "\u30ff" or "\u4e00" <= ch <= "\u9fff" for ch in s)
except ImportError:
    _jp2romaji = lambda s: s          # silently noop if pykakasi missing
    _is_jp     = lambda s: False
# ▲ NEW

def _first(tag):
    return (tag or [None])[0]

def read_artist_title(audio_path: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract artist & title **only from embedded tags**.
    • Tries ID3 (TPE1/TIT2) / MP4 / Vorbis tags via Mutagen.
    • If tag fields are JP characters, auto-converts to romaji.
    • No more guessing from filename or folder.
    """
    try:
        audio = MutagenFile(audio_path, easy=False)
        artist = title = None

        if audio:                                            # --- ID3 / MP4 / Vorbis
            artist = _first(audio.tags.get("TPE1")) \
                     or _first(audio.tags.get("©ART")) \
                     or _first(audio.tags.get("artist"))
            title  = _first(audio.tags.get("TIT2")) \
                     or _first(audio.tags.get("©nam")) \
                     or _first(audio.tags.get("title"))

        # romaji pass
        if artist and _is_jp(artist): artist = _jp2romaji(artist)
        if title  and _is_jp(title):  title  = _jp2romaji(title)

        return artist, title
    except Exception as e:
        print("Mutagen error:", e)
        return None, None
