"""Migration support for Phileas.

Detects existing ~/.phileas/ data from pre-config installations and creates
a default config.toml with values matching the pre-config hardcoded defaults.
"""

from __future__ import annotations

from pathlib import Path

_CONFIG_TEMPLATE = """\
# Phileas configuration
# Generated during migration from pre-config installation

[storage]
home = "{home}"

[recall]
similarity_floor = 0.5
relevance_floor = 0.15
graph_boost = 0.5
mmr_lambda = 0.7
default_top_k = 10

[scoring]
relevance_weight = 0.55
importance_weight = 0.2
recency_weight = 0.15
access_weight = 0.1

[logging]
level = "INFO"
file_max_bytes = 5242880
file_backup_count = 3
"""


def detect_existing_data(home: Path) -> dict:
    """Check for existing Phileas data in a directory.

    Returns a dict with keys:
        has_data    -- True if any recognized data artifact exists
        has_sqlite  -- True if memory.db is present
        has_chroma  -- True if chroma/ directory is present
        has_graph   -- True if graph/ directory is present
        has_config  -- True if config.toml is present
    """
    has_sqlite = (home / "memory.db").exists()
    has_chroma = (home / "chroma").is_dir()
    has_graph = (home / "graph").is_dir()
    has_config = (home / "config.toml").exists()
    has_data = has_sqlite or has_chroma or has_graph

    return {
        "has_data": has_data,
        "has_sqlite": has_sqlite,
        "has_chroma": has_chroma,
        "has_graph": has_graph,
        "has_config": has_config,
    }


def create_default_config(home: Path) -> Path:
    """Create config.toml with defaults matching pre-config hardcoded values.

    The file is written to ``home/config.toml``.  The parent directory is
    created if it does not already exist.

    Returns the path to the created config file.
    """
    home.mkdir(parents=True, exist_ok=True)
    config_path = home / "config.toml"
    config_path.write_text(_CONFIG_TEMPLATE.format(home=home))
    return config_path
