"""Text analysis helpers: lowercase words, stopwords, a light suffix stemmer."""
from __future__ import annotations

import re

_WORD = re.compile(r"[a-z0-9]+")
_PIECES = re.compile(r"[A-Za-z0-9]+|[^A-Za-z0-9]+")

STOP = frozenset("""
a an the of and or to in on at by for with from as is are was were be been being that which this these those it its
into onto than then thus such wherein whereby said each least one more also may can configured further comprising
comprises comprise including includes include having has have based using use used via when while where upon between
among over under both either not no only so if we our propose proposed
""".split())

_SUFFIXES = ("ations", "ation", "ings", "ing", "edly", "ed", "ies", "es", "s", "ly")

# Small hand-written technical thesaurus (first word of each group is the canonical form).
# It is switchable so the eval can report results with and without it.
USE_THESAURUS = True
SYNONYM_GROUPS = """
sensor detector transducer probe
controller processor regulator
adjust regulate modulate
cool cold chill
layer sheet film
channel duct passage conduit
gas emission exhaust
predict forecast
flow throughput
transmit send report
anomaly abnormality fault deviation
vibration oscillation
accelerometer acceleration inertial
capsule microcapsule encapsulated shell
crack fracture rupture
platelet flake nanosheet nanoplatelet
adhesion bonding coupling
monitor measure sense track
camera imager imaging
melt molten
recoater spreader
heat preheat
fire flame
barrier partition
swell intumescent expand
ultraviolet uv
cure harden
harvester generator scavenging
radio transceiver
clock time timestamp
binder matrix resin
node device
""".strip().splitlines()


def stem(w: str) -> str:
    for suf in _SUFFIXES:
        if w.endswith(suf) and len(w) - len(suf) >= 4:
            return w[: -len(suf)] + ("y" if suf == "ies" else "")
    return w


def _build_synonyms() -> dict[str, str]:
    m = {}
    for line in SYNONYM_GROUPS:
        ws = [stem(w) for w in line.split()]
        for w in ws[1:]:
            m[w] = ws[0]
    return m


_SYN = _build_synonyms()


def words(text: str) -> list[str]:
    """Lowercase words; hyphenated compounds also yield their joined form (micro-channel -> microchannel)."""
    out = []
    for tok in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower()):
        parts = tok.split("-")
        out.extend(parts)
        if len(parts) > 1:
            out.append("".join(parts))
    return out


def analyze(text: str) -> list[str]:
    """Stemmed content terms (stopwords removed, optional thesaurus applied)."""
    out = [stem(w) for w in words(text) if w not in STOP and len(w) > 1]
    return [_SYN.get(w, w) for w in out] if USE_THESAURUS else out


def analyze_with_bigrams(text: str) -> list[str]:
    t = analyze(text)
    return t + [f"{a}_{b}" for a, b in zip(t, t[1:])]


def _canon(w: str) -> str:
    w = stem(w)
    return _SYN.get(w, w) if USE_THESAURUS else w


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text.strip())
    return [p.strip() for p in parts if p.strip()]


def highlight_segments(text: str, stems: set[str]) -> list[dict]:
    """Split text into [{t, hit}] segments so a UI can mark matches without raw HTML."""
    out = []
    for piece in _PIECES.findall(text):
        hit = bool(_WORD.fullmatch(piece.lower())) and piece.lower() not in STOP and _canon(piece.lower()) in stems
        if out and out[-1]["hit"] == hit:
            out[-1]["t"] += piece
        else:
            out.append({"t": piece, "hit": hit})
    return out
