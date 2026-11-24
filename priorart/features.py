"""Feature extraction: break a disclosure into key technical features.

The default extractor is deterministic (rules over clauses). An optional
LLM-backed extractor sits behind the same interface and is selected with
PRIORART_FEATURE_PROVIDER=llm (OpenAI-compatible chat endpoint configured via
env vars). Any LLM failure falls back to the deterministic extractor.
"""
from __future__ import annotations

import json
import logging
import os
import re
import urllib.request
from dataclasses import dataclass, asdict
from typing import Protocol

from .text import analyze, split_sentences

log = logging.getLogger("priorart.features")

PREAMBLE = re.compile(
    r"^(we|our|this (invention|disclosure|application)|the (present )?(invention|disclosure))\b.{0,40}"
    r"\b(propos|disclos|relat|describ|introduc|present|concern)", re.I)
CLAUSE_SPLIT = re.compile(r";\s+|,\s+(?:and|while|whereas|rather than|in which case|so that)\s+|,\s+which\s+(?=\w)", re.I)
LEAD = re.compile(r"^(and|also|further|additionally|moreover|then|otherwise|so)\b[,\s]*", re.I)
SUBJ_VERB = re.compile(
    r"^(?:the|this|our|each)\s+[\w-]+(?:\s+(?:also|further|additionally))?\s+"
    r"(?:comprises?|includes?|has|have|employs?|provides?|uses?|contains?)\s+(?=(?:an?|the|one|at|\d)\s)", re.I)
MAX_FEATURES = 8
MIN_TERMS = 4


@dataclass
class Feature:
    id: str
    text: str
    stems: list[str]

    def to_dict(self):
        return asdict(self)


class FeatureExtractor(Protocol):
    name: str

    def extract(self, text: str) -> list[Feature]: ...


def _jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / max(1, len(a | b))


def build_features(clauses: list[str]) -> list[Feature]:
    feats: list[Feature] = []
    for c in clauses:
        c = LEAD.sub("", c.strip()).rstrip(" .;,")
        c = SUBJ_VERB.sub("", c)
        st = analyze(c)
        if len(set(st)) < MIN_TERMS:
            continue
        if any(_jaccard(st, f.stems) > 0.7 for f in feats):
            continue
        c = c[0].upper() + c[1:]
        feats.append(Feature(id=f"F{len(feats) + 1}", text=c, stems=sorted(set(st))))
        if len(feats) >= MAX_FEATURES:
            break
    return feats


class HeuristicExtractor:
    name = "heuristic"

    def extract(self, text: str) -> list[Feature]:
        clauses: list[str] = []
        for sent in split_sentences(text):
            if PREAMBLE.search(sent):
                continue
            clauses.extend(p for p in CLAUSE_SPLIT.split(sent) if p)
        return build_features(clauses)


class LLMExtractor:
    """OpenAI-compatible chat-completions client. Config comes from env only."""
    name = "llm"

    PROMPT = ("Split the invention disclosure into at most 8 distinct technical features. "
              "Return only a JSON array of short strings, each a self-contained feature using the disclosure's own wording.")

    def __init__(self):
        self.base = os.environ.get("PRIORART_LLM_BASE_URL", "").rstrip("/")
        self.credential = os.environ.get("PRIORART_LLM_KEY", "")
        self.model = os.environ.get("PRIORART_LLM_MODEL", "")
        self.fallback = HeuristicExtractor()

    def _post(self, payload: dict) -> dict:
        req = urllib.request.Request(
            self.base + "/chat/completions", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.credential}"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())

    def extract(self, text: str) -> list[Feature]:
        try:
            if not (self.base and self.model):
                raise RuntimeError("PRIORART_LLM_BASE_URL / PRIORART_LLM_MODEL not set")
            resp = self._post({"model": self.model, "temperature": 0, "messages": [
                {"role": "system", "content": self.PROMPT}, {"role": "user", "content": text}]})
            content = resp["choices"][0]["message"]["content"]
            items = json.loads(content[content.index("["): content.rindex("]") + 1])
            feats = build_features([str(i) for i in items])
            if feats:
                return feats
        except Exception as e:  # noqa: BLE001 - deliberate: never fail a search because of the LLM
            log.warning("LLM feature extraction failed (%s); using heuristic extractor", e)
        return self.fallback.extract(text)


def get_extractor() -> FeatureExtractor:
    return LLMExtractor() if os.environ.get("PRIORART_FEATURE_PROVIDER", "heuristic").lower() == "llm" else HeuristicExtractor()
