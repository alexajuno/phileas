"""Tests for the Phileas CLI."""

import json

from click.testing import CliRunner

from phileas.cli import app


def _runner_with_home(tmp_dir, monkeypatch):
    """Set PHILEAS_HOME to a temp dir and return a CliRunner."""
    monkeypatch.setenv("PHILEAS_HOME", str(tmp_dir))
    return CliRunner()


# ------------------------------------------------------------------
# status
# ------------------------------------------------------------------


def test_cli_status(tmp_dir, monkeypatch):
    runner = _runner_with_home(tmp_dir, monkeypatch)
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0
    assert "Phileas Memory System" in result.output


# ------------------------------------------------------------------
# remember
# ------------------------------------------------------------------


def test_cli_remember(tmp_dir, monkeypatch):
    runner = _runner_with_home(tmp_dir, monkeypatch)
    result = runner.invoke(app, ["remember", "I like Python"])
    assert result.exit_code == 0
    assert "Stored" in result.output


def test_cli_remember_with_type(tmp_dir, monkeypatch):
    runner = _runner_with_home(tmp_dir, monkeypatch)
    result = runner.invoke(app, ["remember", "Name is Giao", "--type", "profile", "--importance", "9"])
    assert result.exit_code == 0
    assert "Stored" in result.output


# ------------------------------------------------------------------
# recall
# ------------------------------------------------------------------


def test_cli_recall(tmp_dir, monkeypatch):
    runner = _runner_with_home(tmp_dir, monkeypatch)
    # Store first, then recall
    runner.invoke(app, ["remember", "I like Python programming"])
    result = runner.invoke(app, ["recall", "Python"])
    assert result.exit_code == 0


def test_cli_recall_empty(tmp_dir, monkeypatch):
    runner = _runner_with_home(tmp_dir, monkeypatch)
    result = runner.invoke(app, ["recall", "nonexistent"])
    assert result.exit_code == 0
    assert "No memories found" in result.output


def test_cli_recall_with_type(tmp_dir, monkeypatch):
    runner = _runner_with_home(tmp_dir, monkeypatch)
    runner.invoke(app, ["remember", "I like React", "--type", "knowledge"])
    result = runner.invoke(app, ["recall", "React", "--type", "knowledge"])
    assert result.exit_code == 0


# ------------------------------------------------------------------
# forget
# ------------------------------------------------------------------


def test_cli_forget(tmp_dir, monkeypatch):
    runner = _runner_with_home(tmp_dir, monkeypatch)
    # Store a memory, then get its ID from list
    runner.invoke(app, ["remember", "wrong fact"])
    list_result = runner.invoke(app, ["list"])
    assert list_result.exit_code == 0

    # Get the memory ID from the DB directly
    from phileas.config import load_config
    from phileas.db import Database

    cfg = load_config()
    db = Database(path=cfg.db_path)
    items = db.get_active_items()
    assert len(items) > 0

    result = runner.invoke(app, ["forget", items[0].id])
    assert result.exit_code == 0
    assert "archived" in result.output.lower()


# ------------------------------------------------------------------
# update
# ------------------------------------------------------------------


def test_cli_update(tmp_dir, monkeypatch):
    runner = _runner_with_home(tmp_dir, monkeypatch)
    runner.invoke(app, ["remember", "old summary"])

    from phileas.config import load_config
    from phileas.db import Database

    cfg = load_config()
    db = Database(path=cfg.db_path)
    items = db.get_active_items()
    assert len(items) > 0

    result = runner.invoke(app, ["update", items[0].id, "new summary"])
    assert result.exit_code == 0
    assert "Updated" in result.output


def test_cli_update_not_found(tmp_dir, monkeypatch):
    runner = _runner_with_home(tmp_dir, monkeypatch)
    result = runner.invoke(app, ["update", "nonexistent-id", "new summary"])
    assert result.exit_code == 1


# ------------------------------------------------------------------
# list
# ------------------------------------------------------------------


def test_cli_list(tmp_dir, monkeypatch):
    runner = _runner_with_home(tmp_dir, monkeypatch)
    runner.invoke(app, ["remember", "fact one"])
    runner.invoke(app, ["remember", "fact two"])
    result = runner.invoke(app, ["list"])
    assert result.exit_code == 0


def test_cli_list_with_type(tmp_dir, monkeypatch):
    runner = _runner_with_home(tmp_dir, monkeypatch)
    runner.invoke(app, ["remember", "profile fact", "--type", "profile"])
    result = runner.invoke(app, ["list", "--type", "profile"])
    assert result.exit_code == 0


def test_cli_list_with_limit(tmp_dir, monkeypatch):
    runner = _runner_with_home(tmp_dir, monkeypatch)
    for i in range(5):
        runner.invoke(app, ["remember", f"fact {i}"])
    result = runner.invoke(app, ["list", "--limit", "2"])
    assert result.exit_code == 0


# ------------------------------------------------------------------
# show
# ------------------------------------------------------------------


def test_cli_show(tmp_dir, monkeypatch):
    runner = _runner_with_home(tmp_dir, monkeypatch)
    runner.invoke(app, ["remember", "detailed memory"])

    from phileas.config import load_config
    from phileas.db import Database

    cfg = load_config()
    db = Database(path=cfg.db_path)
    items = db.get_active_items()
    assert len(items) > 0

    result = runner.invoke(app, ["show", items[0].id])
    assert result.exit_code == 0
    assert "Memory Detail" in result.output


def test_cli_show_not_found(tmp_dir, monkeypatch):
    runner = _runner_with_home(tmp_dir, monkeypatch)
    result = runner.invoke(app, ["show", "nonexistent-id"])
    assert result.exit_code == 1


# ------------------------------------------------------------------
# export
# ------------------------------------------------------------------


def test_cli_export_stdout(tmp_dir, monkeypatch):
    runner = _runner_with_home(tmp_dir, monkeypatch)
    runner.invoke(app, ["remember", "exportable fact"])
    result = runner.invoke(app, ["export"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert isinstance(data, list)
    assert len(data) >= 1


def test_cli_export_to_file(tmp_dir, monkeypatch):
    runner = _runner_with_home(tmp_dir, monkeypatch)
    runner.invoke(app, ["remember", "exportable fact"])
    out_path = tmp_dir / "export.json"
    result = runner.invoke(app, ["export", "--output", str(out_path)])
    assert result.exit_code == 0
    assert out_path.exists()
    data = json.loads(out_path.read_text())
    assert isinstance(data, list)


# ------------------------------------------------------------------
# init
# ------------------------------------------------------------------


def test_cli_init(tmp_dir, monkeypatch):
    runner = _runner_with_home(tmp_dir, monkeypatch)
    # Provide input: default home path, skip LLM
    result = runner.invoke(app, ["init"], input=f"{tmp_dir}\nskip\n")
    assert result.exit_code == 0


# ------------------------------------------------------------------
# ingest (without LLM — should fail gracefully)
# ------------------------------------------------------------------


def test_cli_ingest_no_llm(tmp_dir, monkeypatch):
    runner = _runner_with_home(tmp_dir, monkeypatch)
    result = runner.invoke(app, ["ingest", "some text to ingest"])
    assert result.exit_code == 1
    assert "LLM not configured" in result.stderr


# ------------------------------------------------------------------
# consolidate (without LLM — should fail gracefully)
# ------------------------------------------------------------------


def test_cli_consolidate_no_llm(tmp_dir, monkeypatch):
    runner = _runner_with_home(tmp_dir, monkeypatch)
    result = runner.invoke(app, ["consolidate"])
    assert result.exit_code == 1
    assert "LLM not configured" in result.stderr


# ------------------------------------------------------------------
# contradictions (without LLM — should fail gracefully)
# ------------------------------------------------------------------


def test_cli_contradictions_no_llm(tmp_dir, monkeypatch):
    runner = _runner_with_home(tmp_dir, monkeypatch)
    result = runner.invoke(app, ["contradictions"])
    assert result.exit_code == 1
    assert "LLM not configured" in result.stderr


# ------------------------------------------------------------------
# version
# ------------------------------------------------------------------


def test_cli_version(tmp_dir, monkeypatch):
    runner = _runner_with_home(tmp_dir, monkeypatch)
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "version" in result.output.lower()


# ------------------------------------------------------------------
# help
# ------------------------------------------------------------------


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Phileas" in result.output
    assert "status" in result.output
    assert "remember" in result.output
    assert "recall" in result.output


# ------------------------------------------------------------------
# full end-to-end integration smoke test
# ------------------------------------------------------------------


def test_full_flow(tmp_dir, monkeypatch):
    """End-to-end: init (skip LLM) -> remember -> recall -> list -> status -> export."""
    monkeypatch.setenv("PHILEAS_HOME", str(tmp_dir))
    runner = CliRunner()

    # Init with skip LLM
    result = runner.invoke(app, ["init"], input=f"{tmp_dir}\nskip\n")
    assert result.exit_code == 0

    # Remember
    result = runner.invoke(app, ["remember", "I love building memory systems"])
    assert result.exit_code == 0
    assert "Stored" in result.output or "stored" in result.output.lower()

    # Remember another
    result = runner.invoke(app, ["remember", "Python is my favorite language", "--type", "behavior"])
    assert result.exit_code == 0

    # Recall
    result = runner.invoke(app, ["recall", "memory systems"])
    assert result.exit_code == 0
    assert "memory" in result.output.lower()

    # List
    result = runner.invoke(app, ["list"])
    assert result.exit_code == 0

    # Status
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0

    # Export
    result = runner.invoke(app, ["export"])
    assert result.exit_code == 0
    assert "memory systems" in result.output.lower()
