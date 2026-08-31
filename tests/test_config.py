"""Tests for config loading and the local override.

`config.local.yaml` keeps personal state — squad, bank, armbands — out of the
repository while leaving the zero-argument workflow intact. What matters is that
it merges rather than replaces, so a local file naming only a squad still gets
everything else from `config.yaml`.
"""

from __future__ import annotations

import textwrap

import pytest
import yaml

from src.config import _deep_merge, load_config, local_config_path


def write(path, data: str) -> None:
    path.write_text(textwrap.dedent(data), encoding="utf-8")


@pytest.fixture
def base(tmp_path):
    cfg = tmp_path / "config.yaml"
    write(cfg, """
        season:
          current: "2026-27"
          history: ["2025-26"]
        paths:
          data_dir: "data"
          cache_dir: "data/cache"
          processed_dir: "data/processed"
        squad:
          players: []
          bank: null
          horizon: 5
        api: {base_url: "x", ttl: {default: 1}}
        historical: {vaastav_base: "y"}
        features: {windows: [3]}
        evaluate: {min_train_gws: 4}
    """)
    return cfg


def test_local_path_sits_beside_the_base(base) -> None:
    assert local_config_path(base).name == "config.local.yaml"
    assert local_config_path(base).parent == base.parent


def test_without_a_local_file_the_base_is_used_unchanged(base) -> None:
    config = load_config(base)
    assert config.squad_players == []
    assert config.squad_bank is None
    assert config.squad_horizon == 5


def test_local_overrides_merge_rather_than_replace(base) -> None:
    """A local file naming only a squad must not wipe the rest of the block."""
    write(local_config_path(base), """
        squad:
          players: [haaland, cherki]
          bank: 1.5
    """)
    config = load_config(base)

    assert config.squad_players == ["haaland", "cherki"]
    assert config.squad_bank == 15                 # millions -> tenths
    assert config.squad_horizon == 5, "horizon should fall through from the base"
    assert config.season_current == "2026-27", "untouched sections survive"


def test_use_local_false_ignores_the_override(base) -> None:
    """How the committed page is built: nobody else's squad in it."""
    write(local_config_path(base), "squad:\n  players: [haaland]\n")

    assert load_config(base).squad_players == ["haaland"]
    assert load_config(base, use_local=False).squad_players == []


def test_deep_merge_recurses_and_does_not_mutate() -> None:
    base_cfg = {"squad": {"players": [1], "horizon": 5}, "api": {"ttl": 60}}
    override = {"squad": {"players": [2]}}

    merged = _deep_merge(base_cfg, override)

    assert merged == {"squad": {"players": [2], "horizon": 5}, "api": {"ttl": 60}}
    assert base_cfg["squad"]["players"] == [1], "inputs must be left alone"


def test_the_committed_config_ships_no_squad() -> None:
    """A fresh clone should open an empty pitch, not whoever last pushed."""
    from src.config import DEFAULT_CONFIG_PATH

    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    squad = raw.get("squad") or {}
    assert not squad.get("players")
    assert squad.get("captain") is None
    assert squad.get("bank") is None
