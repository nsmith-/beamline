import os
from pathlib import Path

import pytest


@pytest.fixture
def artifacts_dir(request: pytest.FixtureRequest) -> Path:
    """Directory for storing test artifacts.

    Creates a "test_artifacts" directory in the current working directory and returns
    a Path pointing to a subdirectory for the current test module.
    """
    testpath = str(request.path.relative_to(Path(__file__).parent))
    testpath = testpath.removesuffix(".py")
    out = Path(os.getcwd()) / "test_artifacts" / testpath
    out.mkdir(parents=True, exist_ok=True)
    return out
