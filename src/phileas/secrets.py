"""Persisted extraction API keys, kept out of ``config.toml``.

A provider API key is a secret, so it never belongs in ``config.toml``: that file
is plain, meant to be read, copied, and hand-edited, and a settings UI echoes it
back. The environment is the canonical place a key lives, and it stays the first
place looked at: an operator who exports ``PHILEAS_ANTHROPIC_API_KEY`` (or sets it
in a systemd drop-in) needs nothing here.

Requiring every user to wire an env var by hand is friction, though, so this gives
the CLI and the settings UI a place to stash a key without that step: a per-profile
``secrets.toml`` beside ``config.toml`` in the profile home, written ``0600`` (owner
read/write only), and read at call time as a fallback behind the environment.

Resolution is always environment-first. ``os.environ[name]`` wins over the stored
file, so a deployment that injects the key through the environment is never shadowed
by a stale stored value, and a test that sets the env var touches no disk. The file
is keyed by env-var NAME (``PHILEAS_ANTHROPIC_API_KEY``), storing exactly the value
that env var would carry, so switching provider (and with it ``api_key_env``) picks
up the matching stored key on its own.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised only on 3.10 and below
    import tomli as tomllib  # type: ignore[no-redef]

# Owner read/write, nothing for group or others: the standard mode for a file
# that holds a credential (the same mode ssh demands of a private key).
_SECRET_FILE_MODE = 0o600


def secrets_path(home: Path) -> Path:
    """Path to the profile's secret store, beside its ``config.toml``."""
    return home / "secrets.toml"


def load_secrets(home: Path) -> dict[str, str]:
    """The stored ``{env-var-name: key}`` map, or ``{}`` when nothing is stored."""
    path = secrets_path(home)
    if not path.is_file():
        return {}
    with open(path, "rb") as f:
        data: dict[str, Any] = tomllib.load(f)
    # Only string values are keys; ignore anything a hand-edit may have introduced.
    return {k: v for k, v in data.items() if isinstance(v, str)}


def stored_key_names(home: Path) -> list[str]:
    """Env-var names that currently have a stored key, sorted for stable display."""
    return sorted(load_secrets(home))


def read_stored_key(home: Path, name: str) -> str | None:
    """The stored key for env var ``name``, ignoring the environment. ``None`` if absent."""
    return load_secrets(home).get(name)


def resolve_key(home: Path | None, name: str) -> str | None:
    """Resolve a key env-first: the environment wins, the stored file is the fallback.

    ``home`` is optional so a caller without a resolved profile home (an isolated
    unit test, say) still gets the environment lookup and simply skips the file.
    """
    from_env = os.environ.get(name)
    if from_env:
        return from_env
    if home is None:
        return None
    return read_stored_key(home, name)


def store_key(home: Path, name: str, value: str) -> Path:
    """Store ``value`` for env var ``name`` in the ``0600`` secret file.

    Creates the home and the file as needed, strips surrounding whitespace from
    the pasted value, and enforces ``0600`` on every write (not just on create),
    so a file that somehow gained looser permissions is tightened here. Raises
    ``ValueError`` on an empty value.
    """
    key = value.strip()
    if not key:
        raise ValueError("refusing to store an empty key")
    home.mkdir(parents=True, exist_ok=True)
    path = secrets_path(home)
    data = load_secrets(home)
    data[name] = key
    _write(path, data)
    return path


def remove_key(home: Path, name: str) -> bool:
    """Drop the stored key for ``name``. Returns whether one was there to remove.

    Removing the last stored key deletes the file rather than leaving an empty
    ``secrets.toml`` behind.
    """
    data = load_secrets(home)
    if name not in data:
        return False
    del data[name]
    path = secrets_path(home)
    if data:
        _write(path, data)
    else:
        path.unlink(missing_ok=True)
    return True


def _write(path: Path, data: dict[str, str]) -> None:
    """Write ``data`` to ``path`` as TOML, owner-only (``0600``)."""
    import tomli_w

    # Open with 0600 so the key is never briefly world-readable between create and
    # chmod; chmod after as well so a pre-existing looser file is tightened.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, _SECRET_FILE_MODE)
    with os.fdopen(fd, "wb") as f:
        tomli_w.dump(data, f)
    os.chmod(path, _SECRET_FILE_MODE)
