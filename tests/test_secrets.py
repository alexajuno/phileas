"""The 0600 secrets store: environment-first resolution, owner-only file, round-trips.

Offline and filesystem-only: each case writes into a fresh ``tmp_path`` home, so no
real profile secrets file is read or touched.
"""

from __future__ import annotations

import stat

import pytest

from phileas import secrets

_ENV = "PHILEAS_ANTHROPIC_API_KEY"


def test_resolve_prefers_env_over_stored(tmp_path, monkeypatch):
    secrets.store_key(tmp_path, _ENV, "from-file")
    monkeypatch.setenv(_ENV, "from-env")
    assert secrets.resolve_key(tmp_path, _ENV) == "from-env"


def test_resolve_falls_back_to_stored(tmp_path, monkeypatch):
    monkeypatch.delenv(_ENV, raising=False)
    secrets.store_key(tmp_path, _ENV, "from-file")
    assert secrets.resolve_key(tmp_path, _ENV) == "from-file"


def test_resolve_none_when_absent(tmp_path, monkeypatch):
    monkeypatch.delenv("PHILEAS_OPENAI_API_KEY", raising=False)
    assert secrets.resolve_key(tmp_path, "PHILEAS_OPENAI_API_KEY") is None


def test_resolve_without_home_is_env_only(monkeypatch):
    monkeypatch.setenv(_ENV, "e")
    assert secrets.resolve_key(None, _ENV) == "e"
    monkeypatch.delenv(_ENV)
    assert secrets.resolve_key(None, _ENV) is None


def test_store_writes_owner_only_mode(tmp_path):
    path = secrets.store_key(tmp_path, _ENV, "sk-x")
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_store_strips_surrounding_whitespace(tmp_path):
    secrets.store_key(tmp_path, "K", "  sk-trimmed\n")
    assert secrets.read_stored_key(tmp_path, "K") == "sk-trimmed"


def test_store_rejects_empty(tmp_path):
    with pytest.raises(ValueError):
        secrets.store_key(tmp_path, "K", "   ")


def test_store_tightens_loose_permissions_and_keeps_siblings(tmp_path):
    path = secrets.secrets_path(tmp_path)
    path.write_text('K = "v"\n')
    path.chmod(0o644)
    secrets.store_key(tmp_path, "K2", "v2")
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert secrets.read_stored_key(tmp_path, "K") == "v"  # existing key preserved


def test_remove_deletes_file_when_last_key_goes(tmp_path):
    secrets.store_key(tmp_path, "ONLY", "v")
    assert secrets.remove_key(tmp_path, "ONLY") is True
    assert not secrets.secrets_path(tmp_path).exists()


def test_remove_keeps_other_keys_and_mode(tmp_path):
    secrets.store_key(tmp_path, "A", "1")
    secrets.store_key(tmp_path, "B", "2")
    assert secrets.remove_key(tmp_path, "A") is True
    assert secrets.stored_key_names(tmp_path) == ["B"]
    assert stat.S_IMODE(secrets.secrets_path(tmp_path).stat().st_mode) == 0o600


def test_remove_missing_returns_false(tmp_path):
    assert secrets.remove_key(tmp_path, "NOPE") is False


def test_load_ignores_non_string_values(tmp_path):
    # A hand-edited non-string value is ignored rather than surfaced as a key.
    secrets.secrets_path(tmp_path).write_text('GOOD = "v"\nBAD = 5\n')
    assert secrets.stored_key_names(tmp_path) == ["GOOD"]
    assert secrets.read_stored_key(tmp_path, "BAD") is None
