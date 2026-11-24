"""Text analysis helpers: lowercase words, stopwords, a light suffix stemmer."""
from __future__ import annotations

import re

_WORD = re.compile(r"[a-z0-9]+")
_PIECES = re.compile(r"[A-Za-z0-9]+|[^A-Za-z0-9]+")

STOP = frozenset("""
a an the of and or to in on at by for with from as is are was were be been being that which this these those it its
into onto than then thus such wherein whereby said each least one more also may can configured further comprising
comprises comprise including includes include having has have based using use used via when while where upon between
among over under both either not no only so if we our propose proposed claim
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
fibre fiber
micrometre micrometer micron um
millimetre millimeter mm
aluminium aluminum
""".strip().splitlines()


def stem(w: str) -> str:
    for suf in _SUFFIXES:
        if w.endswith(suf) and len(w) - len(suf) >= 4:
            w = w[: -len(suf)] + ("y" if suf == "ies" else "")
            break
    # charge / charging / charged and micrometre / micrometres meet at the same stem
    return w[:-1] if w.endswith("e") and len(w) >= 5 else w


_NUMBER_WORDS = {w: str(n) for n, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve".split())}
_NUMBER_WORDS.update({"twenty": "20", "thirty": "30", "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70",
                      "eighty": "80", "ninety": "90", "hundred": "100", "thousand": "1000"})


def normalize_text(text: str) -> str:
    """Spelling-neutral text: lowercase, 'per cent' / '%' -> 'percent', 'one hundred' -> '100', 5 mm -> 5 mm."""
    t = text.lower().replace("µ", "u")
    t = re.sub(r"(\d)\s*%", r"\1 percent", t)
    t = t.replace("per cent", "percent").replace("one hundred", "hundred")
    return t


def _build_synonyms() -> dict[str, str]:
    m = {}
    for line in SYNONYM_GROUPS:
        ws = [stem(w) for w in line.split()]
        for w in ws[1:]:
            m[w] = ws[0]
    return m


_SYN = _build_synonyms()


def words(text: str) -> list[str]:
    """Lowercase words; hyphenated compounds also yield their joined form (micro-channel -> microchannel).
    Spelled-out numbers become number tokens (n40) so 'forty percent' meets '40%'."""
    out = []
    for tok in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", normalize_text(text)):
        parts = tok.split("-")
        out.extend(parts)
        if len(parts) > 1:
            out.append("".join(parts))
    return [("n" + _NUMBER_WORDS[w]) if w in _NUMBER_WORDS and w != "one" else ("n" + w) if w.isdigit() else w
            for w in out]


def analyze(text: str) -> list[str]:
    """Stemmed content terms (stopwords removed, optional thesaurus applied)."""
    out = [stem(w) for w in words(text) if w not in STOP and len(w) > 1]
    return [_SYN.get(w, w) for w in out] if USE_THESAURUS else out


def analyze_with_bigrams(text: str) -> list[str]:
    t = analyze(text)
    return t + [f"{a}_{b}" for a, b in zip(t, t[1:])]


def _canon(w: str) -> str:
    if w in _NUMBER_WORDS and w != "one":
        w = "n" + _NUMBER_WORDS[w]
    elif w.isdigit():
        w = "n" + w
    w = stem(w)
    return _SYN.get(w, w) if USE_THESAURUS else w


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text.strip())
    return [p.strip() for p in parts if p.strip()]


def highlight_segments(text: str, stems: set[str], soft: set[str] | None = None) -> list[dict]:
    """Split text into [{t, hit, soft}] segments so a UI can mark matches without raw HTML.
    `stems` are exact matches; `soft` are terms matched only by similarity (hit is true for both)."""
    soft = soft or set()
    marks = []
    for piece in _PIECES.findall(text):
        low = piece.lower()
        c = _canon(low) if _WORD.fullmatch(low) and low not in STOP else None
        exact = c is not None and c in stems
        sft = c is not None and not exact and c in soft
        marks.append([piece, exact or sft, sft])
    # a space between two marked words of the same kind belongs to the mark, so phrases read as one highlight
    for k in range(1, len(marks) - 1):
        a, m, b = marks[k - 1], marks[k], marks[k + 1]
        if not m[1] and not m[0].strip() and a[1] and b[1] and a[2] == b[2]:
            m[1], m[2] = True, a[2]
    out: list[dict] = []
    for piece, hit, sft in marks:
        if out and out[-1]["hit"] == hit and out[-1]["soft"] == sft:
            out[-1]["t"] += piece
        else:
            out.append({"t": piece, "hit": hit, "soft": sft})
    return out


# Boilerplate that opens patent sentences and claims. Stripping it keeps the words every claim shares ("battery
# pack ... further comprising") from inflating feature matches.
_BOILER = [re.compile(p, re.I) for p in (
    r"^(?:an?|the)\s+[^:;,]{0,90}?\s+comprising\s*:\s*",
    r"^the\s+[^,]{0,90}?\s+of\s+claim\s+\d+,?\s*(?:further\s+)?(?:comprising|including)\s+",
    r"^(?:in\s+(?:some|certain)\s+embodiments,?\s+)?the\s+[\w\s-]{0,60}?\s+(?:further\s+)?(?:includes|comprises|has)\s+",
    r"^an?\s+[\w\s-]{0,60}?\s+is\s+disclosed\s+that\s+(?:includes|has)\s+",
    r"^embodiments\s+of\s+an?\s+[\w\s-]{0,60}?\s+are\s+described,\s*characteri[sz]ed\s+by\s+",
    r"^(?:variants\s+include|additionally,|this\s+[\w\s-]{0,30}?\s+(?:reduces|improves|extends|lowers))\s*",
)]


def core_text(text: str) -> str:
    """Passage text without the opening boilerplate (claim preambles, 'In some embodiments the X includes')."""
    t = text.strip()
    for _ in range(2):
        for rx in _BOILER:
            n = rx.sub("", t, count=1)
            if n != t and len(n.split()) >= 3:
                t = n
                break
    return re.sub(r"\s+is provided\.?$", "", t).strip()
