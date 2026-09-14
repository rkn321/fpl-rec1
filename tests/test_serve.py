"""Tests for the local helper that writes `config.local.yaml` from the page.

The page decides "first visit or not" purely from whether the local file
exists, and the helper is the only thing that writes it — so what matters is
that it refuses anything that is not a plausible squad, writes exactly the
shape `load_config` reads back, and leaves untouched whatever else a person
has put in the file by hand.
"""

from __future__ import annotations

import json
import textwrap
import threading
import urllib.request
from functools import partial
from http.server import ThreadingHTTPServer

import pytest
import yaml

from src import serve
from src.config import load_config, local_config_path


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
    """)
    return cfg


FIFTEEN = [f"player {i}" for i in range(15)]


def good_payload(**overrides):
    payload = {
        "players": FIFTEEN,
        "captain": "player 0",
        "vice": "player 1",
        "bank": 0.3,
        "free_transfers": 1,
    }
    payload.update(overrides)
    return payload


# -- first-visit detection ---------------------------------------------------

def test_read_local_reports_missing_file(base):
    config = load_config(base)
    assert serve.read_local(config) == {"exists": False, "squad": None}


def test_read_local_returns_saved_squad_in_yaml_units(base):
    write(local_config_path(base), """
        squad:
          players: [a, b]
          captain: a
          bank: 1.5
          free_transfers: 2
    """)
    info = serve.read_local(load_config(base))
    assert info["exists"] is True
    assert info["squad"]["players"] == ["a", "b"]
    assert info["squad"]["captain"] == "a"
    assert info["squad"]["vice"] is None
    assert info["squad"]["bank"] == 1.5           # millions, as the file holds it
    assert info["squad"]["free_transfers"] == 2


def test_read_local_accepts_the_comma_separated_spelling(base):
    write(local_config_path(base), 'squad: {players: "a, b, c"}\n')
    assert serve.read_local(load_config(base))["squad"]["players"] == ["a", "b", "c"]


# -- validation ----------------------------------------------------------------

@pytest.mark.parametrize("bad", [
    None,
    [],
    {"players": "not a list"},
    {"players": FIFTEEN[:14]},                       # a squad is 15
    {"players": FIFTEEN + ["one more"]},
    {"players": FIFTEEN[:14] + [""]},                # blank name
    {"players": FIFTEEN, "bank": "lots"},
    {"players": FIFTEEN, "bank": -0.1},
    {"players": FIFTEEN, "free_transfers": 6},
    {"players": FIFTEEN, "free_transfers": 1.5},
    {"players": FIFTEEN, "captain": 42},
])
def test_validate_rejects_implausible_squads(bad):
    with pytest.raises(ValueError):
        serve._validate(bad)


def test_validate_normalises_what_it_keeps():
    out = serve._validate(good_payload(players=[" a "] + FIFTEEN[1:], bank=0.30000001, captain=" a "))
    assert out["players"][0] == "a"
    assert out["bank"] == 0.3
    assert out["captain"] == "a"


def test_validate_leaves_optional_fields_out_when_absent():
    out = serve._validate({"players": FIFTEEN})
    assert set(out) == {"players"}


# -- writing --------------------------------------------------------------------

def test_write_creates_the_file_and_load_config_reads_it_back(base):
    config = load_config(base)
    path = serve.write_local(config, good_payload())
    assert path == local_config_path(base)

    reloaded = load_config(base)
    assert reloaded.squad_players == FIFTEEN
    assert reloaded.squad_captain == "player 0"
    assert reloaded.squad_vice == "player 1"
    assert reloaded.squad_bank == 3                   # tenths, as the page uses
    assert reloaded.squad_free_transfers == 1
    # And the file now counts as "already set up".
    assert serve.read_local(config)["exists"] is True


def test_write_preserves_hand_edited_keys_outside_the_squad(base):
    write(local_config_path(base), """
        squad:
          players: [old]
          horizon: 3
        api: {ttl: {default: 99}}
    """)
    serve.write_local(load_config(base), good_payload())

    raw = yaml.safe_load(local_config_path(base).read_text(encoding="utf-8"))
    assert raw["squad"]["players"] == FIFTEEN       # replaced
    assert raw["squad"]["horizon"] == 3             # kept: not one of the page's keys
    assert raw["api"] == {"ttl": {"default": 99}}   # kept: not the squad at all


def test_write_refuses_bad_payloads_without_touching_disk(base):
    with pytest.raises(ValueError):
        serve.write_local(load_config(base), {"players": ["only one"]})
    assert not local_config_path(base).exists()


# -- the endpoints, end to end ----------------------------------------------------

@pytest.fixture
def server(base, tmp_path):
    config = load_config(base)
    site = tmp_path / "site"
    site.mkdir()
    (site / "index.html").write_text("<p>hi</p>", encoding="utf-8")

    serve._Handler.config = config
    httpd = ThreadingHTTPServer((serve.HOST, 0), partial(serve._Handler, directory=str(site)))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://{serve.HOST}:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()
        httpd.server_close()


def _get(url):
    with urllib.request.urlopen(url, timeout=5) as res:
        return res.status, json.loads(res.read())


def _post(url, payload):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as res:
            return res.status, json.loads(res.read())
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read())


def test_endpoints_round_trip(server, base):
    status, info = _get(f"{server}/api/config")
    assert status == 200 and info == {"exists": False, "squad": None}

    status, out = _post(f"{server}/api/config", good_payload())
    assert status == 200 and out["ok"] is True

    status, info = _get(f"{server}/api/config")
    assert status == 200 and info["exists"] is True
    assert info["squad"]["players"] == FIFTEEN
    assert load_config(base).squad_bank == 3


def test_endpoint_reports_validation_errors_as_400(server, base):
    status, out = _post(f"{server}/api/config", {"players": ["one"]})
    assert status == 400
    assert out["ok"] is False and "15" in out["error"]
    assert not local_config_path(base).exists()


def test_static_files_are_still_served(server):
    with urllib.request.urlopen(f"{server}/index.html", timeout=5) as res:
        assert res.status == 200
        assert b"hi" in res.read()
        assert res.headers["Cache-Control"] == "no-store"


# -- live points ------------------------------------------------------------------

class FakeClient:
    """Stands in for FPLClient: answers the two live endpoints from canned data."""

    def __init__(self):
        self.calls = []

    def get(self, path, *, ttl=None, force=False):
        self.calls.append((path, ttl))
        if path.startswith("event/"):
            return {"elements": [
                {"id": 1, "stats": {"total_points": 9, "minutes": 90}},
                {"id": 2, "stats": {"total_points": 0, "minutes": 0}},
                {"id": 3, "stats": {}},                       # no stats yet
            ]}
        if path.startswith("fixtures/"):
            return [
                {"team_h": 10, "team_a": 11, "started": True, "finished": True, "minutes": 90, "kickoff_time": "k1"},
                {"team_h": 12, "team_a": 13, "started": True, "finished": False, "finished_provisional": False, "minutes": 55, "kickoff_time": "k2"},
                {"team_h": 14, "team_a": 15, "started": False, "finished": False, "minutes": 0, "kickoff_time": "k3"},
                {"team_h": None, "team_a": 16},                # not yet scheduled: skipped
            ]
        raise AssertionError(path)


def test_live_points_shapes_the_feed_and_caches_briefly():
    client = FakeClient()
    out = serve.live_points(client, 5)
    assert out["ok"] is True and out["gw"] == 5
    assert out["points"] == {"1": [9, 90], "2": [0, 0], "3": [0, 0]}
    assert [m["finished"] for m in out["fixtures"]] == [True, False, False]
    assert [m["started"] for m in out["fixtures"]] == [True, True, False]
    assert len(out["fixtures"]) == 3                       # the unscheduled one is dropped
    # Both calls go through the client's cache with the short live TTL.
    assert all(ttl == serve.LIVE_TTL for _, ttl in client.calls)


def test_live_endpoint_serves_points_and_validates_gw(server):
    serve._Handler.client = FakeClient()
    try:
        status, out = _get(f"{server}/api/live?gw=5")
        assert status == 200 and out["ok"] is True and out["points"]["1"] == [9, 90]

        for bad in ("gw=0", "gw=39", "gw=abc", ""):
            req = urllib.request.Request(f"{server}/api/live?{bad}")
            try:
                urllib.request.urlopen(req, timeout=5)
                assert False, f"expected 400 for {bad!r}"
            except urllib.error.HTTPError as err:
                assert err.code == 400
    finally:
        serve._Handler.client = None


def test_live_endpoint_reports_an_unreachable_api_as_502(server):
    class Down:
        def get(self, path, *, ttl=None, force=False):
            raise RuntimeError("no network")
    serve._Handler.client = Down()
    try:
        req = urllib.request.Request(f"{server}/api/live?gw=5")
        try:
            urllib.request.urlopen(req, timeout=5)
            assert False, "expected 502"
        except urllib.error.HTTPError as err:
            assert err.code == 502
            assert json.loads(err.read())["ok"] is False
    finally:
        serve._Handler.client = None
