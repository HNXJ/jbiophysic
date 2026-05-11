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
