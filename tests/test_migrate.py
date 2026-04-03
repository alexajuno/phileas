"""Tests for phileas.migrate — migration support for pre-config installations."""

from phileas.migrate import create_default_config, detect_existing_data


def test_detect_existing_data(tmp_path):
    (tmp_path / "memory.db").touch()
    (tmp_path / "chroma").mkdir()
    (tmp_path / "graph").mkdir()
    result = detect_existing_data(tmp_path)
    assert result["has_data"] is True
    assert result["has_sqlite"] is True
    assert result["has_chroma"] is True
    assert result["has_graph"] is True
    assert result["has_config"] is False


def test_detect_no_data(tmp_path):
    result = detect_existing_data(tmp_path)
    assert result["has_data"] is False
    assert result["has_sqlite"] is False
    assert result["has_chroma"] is False
    assert result["has_graph"] is False
    assert result["has_config"] is False


def test_detect_partial_data(tmp_path):
    """Only sqlite present — has_data True, chroma/graph False."""
    (tmp_path / "memory.db").touch()
    result = detect_existing_data(tmp_path)
    assert result["has_data"] is True
    assert result["has_sqlite"] is True
    assert result["has_chroma"] is False
    assert result["has_graph"] is False


def test_detect_config_present(tmp_path):
    (tmp_path / "config.toml").touch()
    result = detect_existing_data(tmp_path)
    assert result["has_config"] is True
    # config.toml alone does not count as data
    assert result["has_data"] is False


def test_create_default_config(tmp_path):
    path = create_default_config(tmp_path)
    assert (tmp_path / "config.toml").exists()
    assert path == tmp_path / "config.toml"
    content = (tmp_path / "config.toml").read_text()
    assert "similarity_floor = 0.5" in content
    assert "relevance_weight = 0.55" in content


def test_create_default_config_creates_parent(tmp_path):
    """create_default_config should create missing parent directories."""
    nested = tmp_path / "a" / "b" / "phileas"
    create_default_config(nested)
    assert (nested / "config.toml").exists()


def test_create_default_config_embeds_home(tmp_path):
    """The home path should appear in the generated config."""
    create_default_config(tmp_path)
    content = (tmp_path / "config.toml").read_text()
    assert str(tmp_path) in content
