import os
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def artifacts_dir() -> Path:
    """Directory for storing test artifacts."""
    out = Path(os.getcwd()) / "test_artifacts"
    out.mkdir(exist_ok=True)
    return out
