def test_public_import_smoke():
    """Smoke test: core imports and version accessible without all extras."""
    import jbiophysic
    from jbiophysic import optim, tfne

    assert jbiophysic.__name__ == "jbiophysic"
    assert hasattr(jbiophysic, "__version__")
    assert isinstance(jbiophysic.__version__, str)
    assert len(jbiophysic.__version__) > 0

    # TFNE module (JAX-backed but guarded).
    assert hasattr(tfne, "make_regular_grid")
    assert hasattr(tfne, "gaussian_mollifier")

    # Optim module: always has bounds helpers, optax-backed symbols conditional.
    assert hasattr(optim, "Bound")
    assert hasattr(optim, "OptimizerManifest")
    # gsdr_direction only present if optax installed; test guards this.
    try:
        import optax  # noqa: F401

        assert hasattr(optim, "gsdr_direction"), "optax installed but gsdr_direction missing"
    except ImportError:
        # optax not installed; optax-backed symbols are expected to be absent.
        pass
