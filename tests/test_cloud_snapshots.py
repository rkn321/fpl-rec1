"""Bringing GitHub Actions' captures home.

What matters: for each gameweek the newest capture wins, wherever it was
taken; a download is skipped when the local copy is already newer; and every
way the GitHub CLI can fail comes back as a one-line reason, never an
exception, because the commands that fetch have a bigger job to finish.
"""

from __future__ import annotations

import io
import json
import subprocess
import textwrap
import zipfile

import pandas as pd
import pytest

from src.config import load_config
from src.data.cloud_snapshots import fetch, latest_artifacts
from src.data.snapshots import keep_if_newer, local_captured_at, snapshot_path


@pytest.fixture
def config(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(textwrap.dedent(f"""
        season:
          current: "2099-00"
          history: []
        paths:
          data_dir: "{(tmp_path / 'data').as_posix()}"
          cache_dir: "{(tmp_path / 'data' / 'cache').as_posix()}"
          processed_dir: "{(tmp_path / 'data' / 'processed').as_posix()}"
        api: {{base_url: "x", ttl: {{default: 1}}}}
        historical: {{vaastav_base: "y"}}
        features: {{windows: [3]}}
    """), encoding="utf-8")
    return load_config(cfg, use_local=False)


def snapshot_bytes(gw: int, captured: str, ep: str = "5.0") -> bytes:
    return json.dumps({
        "gameweek": gw, "captured_at": captured,
        "players": [{"id": 7, "ep_next": ep, "now_cost": 50}],
    }).encode("utf-8")


def zipped(gw: int, captured: str, ep: str = "5.0") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(f"bootstrap_gw{gw:02d}.json", snapshot_bytes(gw, captured, ep))
    return buf.getvalue()


class FakeGh:
    """Answers the two `gh api` calls the fetch makes, and records them."""

    def __init__(self, listing: str, archives: dict[int, bytes], fail: tuple[int, bytes] | None = None):
        self.listing, self.archives, self.fail = listing, archives, fail
        self.downloads: list[int] = []

    def __call__(self, args, capture_output=True, cwd=None, timeout=None):
        if self.fail:
            code, err = self.fail
            return subprocess.CompletedProcess(args, code, b"", err)
        if "--jq" in args:
            return subprocess.CompletedProcess(args, 0, self.listing.encode(), b"")
        artifact_id = int(args[-1].split("/")[-2])
        self.downloads.append(artifact_id)
        return subprocess.CompletedProcess(args, 0, self.archives[artifact_id], b"")


LISTING = "\n".join([
    "11\tsnapshot-gw06\t2099-10-01T00:20:00Z",
    "12\tsnapshot-gw06\t2099-10-01T06:20:00Z",   # newer capture of the same gameweek
    "13\tsnapshot-gw07\t2099-10-12T00:20:00Z",
    "14\tsomething-else\t2099-10-12T00:20:00Z",
    "not a row",
])


# -- the merge rule ----------------------------------------------------------------

def test_the_newer_capture_wins_and_the_older_is_ignored(config) -> None:
    assert keep_if_newer(config, snapshot_bytes(6, "2099-10-01T06:00:00+00:00", ep="6.0")) == 6
    assert keep_if_newer(config, snapshot_bytes(6, "2099-10-01T00:00:00+00:00", ep="1.0")) is None
    stored = json.loads(snapshot_path(config, 6).read_text(encoding="utf-8"))
    assert stored["players"][0]["ep_next"] == "6.0"

    assert keep_if_newer(config, snapshot_bytes(6, "2099-10-02T00:00:00+00:00", ep="7.0")) == 6
    assert local_captured_at(config, 6) == pd.Timestamp("2099-10-02T00:00:00+00:00")


def test_a_capture_with_no_players_is_refused(config) -> None:
    empty = json.dumps({"gameweek": 6, "captured_at": "2099-10-01T00:00:00+00:00", "players": []})
    with pytest.raises(ValueError, match="no players"):
        keep_if_newer(config, empty.encode())
    assert not snapshot_path(config, 6).exists()


# -- the fetch ----------------------------------------------------------------------

def test_the_newest_artifact_per_gameweek_is_chosen() -> None:
    newest = latest_artifacts(FakeGh(LISTING, {}))
    assert set(newest) == {6, 7}
    assert newest[6][0] == 12, "two captures of gameweek 6: the later upload"


def test_fetch_downloads_only_what_is_newer(config) -> None:
    # Gameweek 7 is already here, captured after the cloud's upload of it.
    keep_if_newer(config, snapshot_bytes(7, "2099-10-12T03:00:00+00:00"))
    gh = FakeGh(LISTING, {12: zipped(6, "2099-10-01T06:17:00+00:00"), 13: zipped(7, "2099-10-12T00:17:00+00:00")})

    result = fetch(config, run=gh)
    assert result.error is None
    assert result.stored == [6] and result.current == [7]
    assert gh.downloads == [12], "gameweek 7 is never downloaded"
    assert local_captured_at(config, 6) == pd.Timestamp("2099-10-01T06:17:00+00:00")
    assert "GW6" in result.summary()


def test_no_artifacts_yet_is_not_an_error(config) -> None:
    result = fetch(config, run=FakeGh("", {}))
    assert result.error is None and result.summary() == "cloud captures: none yet"


@pytest.mark.parametrize(
    "failure, reason",
    [
        ((1, b"To get started with GitHub CLI, please run:  gh auth login"), "gh auth login"),
        ((1, b"HTTP 404: Not Found (https://api.github.com/repos/x/y/actions/artifacts)"), "HTTP 404"),
    ],
)
def test_gh_failures_come_back_as_a_reason(config, failure, reason) -> None:
    result = fetch(config, run=FakeGh(LISTING, {}, fail=failure))
    assert result.error and reason in result.error
    assert result.summary().startswith("cloud captures not fetched")


def test_gh_not_installed_comes_back_as_a_reason(config) -> None:
    def missing(*args, **kwargs):
        raise FileNotFoundError("gh")

    result = fetch(config, run=missing)
    assert "not installed" in result.error


def test_a_corrupt_artifact_is_reported_not_raised(config) -> None:
    result = fetch(config, run=FakeGh(LISTING, {12: b"not a zip", 13: zipped(7, "2099-10-12T00:17:00+00:00")}))
    assert result.error and "could not be read" in result.error
