"""Config loading. One `config.yaml` is the single source of truth."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any] = field(repr=False)
    path: Path = DEFAULT_CONFIG_PATH

    # -- section accessors -------------------------------------------------
    @property
    def season_current(self) -> str:
        return self.raw["season"]["current"]

    @property
    def season_history(self) -> list[str]:
        return list(self.raw["season"]["history"])

    @property
    def api(self) -> dict[str, Any]:
        return self.raw["api"]

    @property
    def historical(self) -> dict[str, Any]:
        return self.raw["historical"]

    @property
    def features(self) -> dict[str, Any]:
        return self.raw["features"]

    @property
    def evaluate(self) -> dict[str, Any]:
        return self.raw["evaluate"]

    @property
    def squad(self) -> dict[str, Any]:
        """Your saved team, if `config.yaml` defines one."""
        return self.raw.get("squad") or {}

    @property
    def squad_players(self) -> list[str]:
        """The 15 names, from either a YAML list or a comma-separated string.

        Both spellings are accepted because the page's "Copy squad" button puts
        a comma-separated line on the clipboard, and it should paste straight in.
        """
        players = self.squad.get("players")
        if not players:
            return []
        if isinstance(players, str):
            return [name.strip() for name in players.split(",") if name.strip()]
        return [str(name).strip() for name in players if str(name).strip()]

    @property
    def squad_bank(self) -> int | None:
        """Bank in tenths of a million, matching the API's price units."""
        bank = self.squad.get("bank")
        return None if bank is None else round(float(bank) * 10)

    @property
    def squad_captain(self) -> str | None:
        name = self.squad.get("captain")
        return str(name).strip() if name else None

    @property
    def squad_vice(self) -> str | None:
        name = self.squad.get("vice")
        return str(name).strip() if name else None

    @property
    def squad_free_transfers(self) -> int | None:
        """Free transfers in hand, for seeding a browser with no saved squad."""
        value = self.squad.get("free_transfers")
        return None if value is None else int(value)

    @property
    def squad_horizon(self) -> int | None:
        horizon = self.squad.get("horizon")
        return None if horizon is None else int(horizon)

    # -- paths -------------------------------------------------------------
    def _path(self, key: str) -> Path:
        p = Path(self.raw["paths"][key])
        return p if p.is_absolute() else REPO_ROOT / p

    @property
    def data_dir(self) -> Path:
        return self._path("data_dir")

    @property
    def cache_dir(self) -> Path:
        return self._path("cache_dir")

    @property
    def processed_dir(self) -> Path:
        return self._path("processed_dir")

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.cache_dir, self.processed_dir):
            d.mkdir(parents=True, exist_ok=True)


def load_config(path: str | os.PathLike[str] | None = None) -> Config:
    p = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with open(p, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return Config(raw=raw, path=p)
