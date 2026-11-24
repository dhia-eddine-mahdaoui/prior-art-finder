"""Optional neural embeddings (a small sentence-embedding model run locally through ONNX, via `fastembed`).

Off by default: the project must run offline with no downloads. Enable with PRIORART_DENSE=1 (or `--dense` on the
CLI) after `pip install fastembed`; the model (about 65 MB) is downloaded once on first use into .cache/.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

MODEL = "BAAI/bge-small-en-v1.5"
_ENC = None


def enabled_by_env() -> bool:
    return os.environ.get("PRIORART_DENSE", "").lower() in ("1", "true", "yes")


def available() -> bool:
    try:
        import fastembed  # noqa: F401
        return True
    except Exception:
        return False


class Encoder:
    def __init__(self):
        from fastembed import TextEmbedding
        root = Path(__file__).resolve().parent.parent / ".cache" / "fastembed"
        os.environ.setdefault("FASTEMBED_CACHE_PATH", str(root))
        self.model = TextEmbedding(MODEL, threads=2)

    def encode(self, texts: list[str]) -> np.ndarray:
        v = np.array(list(self.model.embed(list(texts), batch_size=32)), dtype=np.float32)
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)


def get_encoder() -> Encoder:
    global _ENC
    if _ENC is None:
        _ENC = Encoder()
    return _ENC
