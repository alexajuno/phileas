"""The secret-config RPC the settings UI calls: ``config_set_secret`` / ``config_unset_secret``.

These are the web dashboard's write path for an API key. They store the key in the
profile's 0600 secrets file (never config.toml), echo presence rather than the value,
and leave the environment untouched. Exercised through ``daemon._dispatch`` with a
minimal engine stub (only ``.config`` is read), so no models or network are needed.
"""

from __future__ import annotations

import stat
from types import SimpleNamespace

import pytest

from phileas import daemon, secrets
from phileas.config import load_config

_ENV = "PHILEAS_ANTHROPIC_API_KEY"


@pytest.fixture
def engine(tmp_path, monkeypatch):
    monkeypatch.delenv(_ENV, raising=False)
    return SimpleNamespace(config=load_config(home=tmp_path))


def test_set_secret_writes_0600_file_and_reports_stored(engine, tmp_path):
    canary = "sk-LEAKCANARY-42"
    res = daemon._dispatch(engine, "config_set_secret", {"value": canary})

    # Stored in the 0600 secrets file, and reachable via the file.
    assert secrets.read_stored_key(tmp_path, _ENV) == canary
    assert stat.S_IMODE(secrets.secrets_path(tmp_path).stat().st_mode) == 0o600
    # Never in config.toml.
    cfg_path = tmp_path / "config.toml"
    assert not cfg_path.exists() or canary not in cfg_path.read_text()
    # The response echoes presence and source, never the value.
    assert res["restart_required"] is True
    assert res["config"]["secrets"]["llm_api_key_set"] is True
    assert res["config"]["secrets"]["llm_api_key_source"] == "stored"
    assert canary not in str(res)


def test_set_secret_rejects_empty_value(engine):
    with pytest.raises(ValueError):
        daemon._dispatch(engine, "config_set_secret", {"value": "   "})
    with pytest.raises(ValueError):
        daemon._dispatch(engine, "config_set_secret", {"value": None})


def test_set_secret_honors_explicit_name(engine, tmp_path):
    daemon._dispatch(engine, "config_set_secret", {"name": "PHILEAS_OPENAI_API_KEY", "value": "sk-oai"})
    assert secrets.read_stored_key(tmp_path, "PHILEAS_OPENAI_API_KEY") == "sk-oai"
    assert secrets.read_stored_key(tmp_path, _ENV) is None


def test_unset_secret_removes_stored(engine, tmp_path):
    daemon._dispatch(engine, "config_set_secret", {"value": "sk-abc"})
    assert secrets.read_stored_key(tmp_path, _ENV) == "sk-abc"

    res = daemon._dispatch(engine, "config_unset_secret", {})
    assert secrets.read_stored_key(tmp_path, _ENV) is None
    assert res["config"]["secrets"]["llm_api_key_set"] is False
