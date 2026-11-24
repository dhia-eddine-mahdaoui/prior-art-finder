import pytest

from priorart.pipeline import load_samples
from priorart.retrieval import get_index


@pytest.fixture(scope="session")
def index():
    return get_index(use_cache=False)


@pytest.fixture(scope="session")
def samples():
    return {s["id"]: s for s in load_samples()}
