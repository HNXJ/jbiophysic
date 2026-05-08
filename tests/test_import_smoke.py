def test_public_import_smoke():
    import jbiophysic
    from jbiophysic import optim, tfne

    assert jbiophysic.__name__ == "jbiophysic"
    assert hasattr(tfne, "make_regular_grid")
    assert hasattr(tfne, "gaussian_mollifier")
    assert hasattr(optim, "gsdr_direction")
