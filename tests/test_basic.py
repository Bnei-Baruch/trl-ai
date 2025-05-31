def test_import():
    """Test that the package can be imported."""
    import src
    assert src.__version__ == "0.1.0" 