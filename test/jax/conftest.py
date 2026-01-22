def pytest_ignore_collect(collection_path, config):
    """Skip collecting tests in this directory if jax is not available."""
    try:
        import jax
    except ImportError:
        return True
    else:
        # Enable JAX NaN debugging for all tests
        jax.config.update("jax_debug_nans", True)
    return False
