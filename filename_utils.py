"""
filename_utils.py – safe, prettified osu! file naming
"""
import re, os, unicodedata, pathlib
from typing import Optional

def _slugify(text: str, allow_unicode: bool = True) -> str:
    """Make a filename-safe slug (keep spaces)."""
    text = text.strip()
    if allow_unicode:
        text = unicodedata.normalize("NFKC", text)
    else:
        text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    # Remove forbidden characters on Windows / POSIX
    text = re.sub(r'[<>:"/\\|?*\0-\31]', "", text)
    # Trim repeated spaces & dots
    text = re.sub(r"\s{2,}", " ", text)
    return text[:180]   # safe length

def build_osu_filename(artist: str, title: str,
                       creator: str, difficulty: str,
                       ext: str = ".osu") -> str:
    """
    Returns e.g.  "Foreground Eclipse - To The Terminus (Mapperatorinator) [browiec/1.2/0.9/0.9/NAN].osu"
    """
    name = f"{artist} - {title} ({creator}) [{difficulty}]{ext}"
    return _slugify(name, allow_unicode=True)

def rename_output(old_path: str,
                  artist: str, title: str,
                  creator: str, difficulty: str) -> str:
    """Rename file on disk, returns new absolute path."""
    directory = os.path.dirname(old_path)
    new_name  = build_osu_filename(artist, title, creator, difficulty,
                                   ext=pathlib.Path(old_path).suffix)
    new_path  = os.path.join(directory, new_name)
    # Handle collisions
    counter = 1
    base, ext = os.path.splitext(new_path)
    while os.path.exists(new_path):
        new_path = f"{base} ({counter}){ext}"
        counter += 1
    os.rename(old_path, new_path)
    return new_path
