import pytest

try:
    import jax.numpy as jnp

    HAS_JAX = True
except ModuleNotFoundError:
    HAS_JAX = False


@pytest.fixture(autouse=True)
def jnp_clip_compatibility_shim():
    """Shim for jnp.clip to support a_max keyword in older JAX versions."""
    if not HAS_JAX:
        yield
        return

    original_clip = jnp.clip

    def safer_clip(a, *args, **kwargs):
        if "a_max" in kwargs:
            kwargs["max"] = kwargs.pop("a_max")
        if "a_min" in kwargs:
            kwargs["min"] = kwargs.pop("a_min")
        return original_clip(a, *args, **kwargs)

    jnp.clip = safer_clip
    yield
    jnp.clip = original_clip


def pytest_ignore_collect(collection_path, config):
    """Skip optional Optax/JAX optimizer tests when optax is not installed.

    The source package keeps Optax as an optional optimization dependency in this
    archive. Core TFNE and jtfne smoke tests must remain runnable on CPU-only or
    minimal Colab runtimes without failing during collection.
    """
    try:
        import optax  # noqa: F401

        has_optax = True
    except ModuleNotFoundError:
        has_optax = False
    if has_optax:
        return False
    path_str = str(collection_path)
    optax_paths = (
        "tests/optim/",
        "tests/test_optim_network_pipeline.py",
    )
    return any(marker in path_str for marker in optax_paths)
