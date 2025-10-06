def pytest_ignore_collect(collection_path, config):
    """Skip collecting tests in this directory if jax is not available."""
    try:
        import jax
    except ImportError:
        return True
    return False
