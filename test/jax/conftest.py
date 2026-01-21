import importlib.util


def pytest_ignore_collect(collection_path, config):
    """Skip collecting tests in this directory if jax is not available."""
    return importlib.util.find_spec("jax") is None
