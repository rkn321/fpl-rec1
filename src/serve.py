"""A local helper that lets the page save your squad to `config.local.yaml`.

The frontend is a static file, and a browser page cannot write to disk — so
opened as a `file://` the only way to persist a squad was "Copy squad" and
paste by hand. Served from here instead, the page gets two small endpoints:

    GET  /api/config   is there a config.local.yaml, and what is in it?
    POST /api/config   write the squad, armbands, bank and free transfers to it

Whether the local file *exists* is what the page uses to decide it is someone's
first visit: absent, it asks them to pick a team and set a bank; present, they
have already done that. The file stays gitignored, so a friend's team never
touches the repository.

Bound to the loopback address only. Nothing here is reachable from the network.
"""

from __future__ import annotations

import json
import logging
import threading
import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import yaml

from .config import Config, local_config_path

log = logging.getLogger(__name__)

DEFAULT_PORT = 8765
HOST = "127.0.0.1"

# The keys the page is allowed to write. Anything else already in the local
# file — `horizon`, say — is left exactly as the user put it.
SQUAD_KEYS = ("players", "captain", "vice", "bank", "free_transfers")


def read_local(config: Config) -> dict[str, Any]:
    """What the page should know about the saved team, in the yaml's own units."""
    path = local_config_path(config.path)
    if not path.exists():
        return {"exists": False, "squad": None}
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    squad = raw.get("squad") or {}
    players = squad.get("players")
    if isinstance(players, str):
        players = [n.strip() for n in players.split(",") if n.strip()]
    return {
        "exists": True,
        "squad": {
            "players": [str(n) for n in (players or [])],
            "captain": squad.get("captain"),
            "vice": squad.get("vice"),
            "bank": squad.get("bank"),
            "free_transfers": squad.get("free_transfers"),
        },
    }


def _validate(payload: Any) -> dict[str, Any]:
    """Reject anything that is not a plausible squad before it touches disk."""
    if not isinstance(payload, dict):
        raise ValueError("expected a JSON object")

    players = payload.get("players")
    if not isinstance(players, list) or not all(isinstance(n, str) and n.strip() for n in players):
        raise ValueError("players must be a list of names")
    if len(players) != 15:
        raise ValueError(f"a squad is 15 players, got {len(players)}")

    out: dict[str, Any] = {"players": [n.strip() for n in players]}

    for key in ("captain", "vice"):
        name = payload.get(key)
        if name is not None:
            if not isinstance(name, str):
                raise ValueError(f"{key} must be a name")
            out[key] = name.strip() or None

    bank = payload.get("bank")
    if bank is not None:
        try:
            bank = float(bank)
        except (TypeError, ValueError):
            raise ValueError("bank must be a number of millions, e.g. 0.3") from None
        if bank < 0:
            raise ValueError("bank cannot be negative")
        # Prices move in 0.1m steps; keep the file from accumulating float noise.
        out["bank"] = round(bank, 1)

    ft = payload.get("free_transfers")
    if ft is not None:
        # `int()` would quietly turn 1.5 into 1; a count has to be whole.
        if isinstance(ft, bool) or not isinstance(ft, (int, float)) or ft != int(ft):
            raise ValueError("free_transfers must be a whole number")
        ft = int(ft)
        if not 0 <= ft <= 5:
            raise ValueError("free_transfers must be between 0 and 5")
        out["free_transfers"] = ft

    return out


def write_local(config: Config, payload: Any) -> Path:
    """Merge the page's squad into `config.local.yaml`, creating it if need be."""
    squad = _validate(payload)
    path = local_config_path(config.path)

    existing: dict[str, Any] = {}
    if path.exists():
        with open(path, "r", encoding="utf-8") as fh:
            existing = yaml.safe_load(fh) or {}

    block = dict(existing.get("squad") or {})
    for key in SQUAD_KEYS:
        block.pop(key, None)
    block.update(squad)
    existing["squad"] = block

    header = (
        "# Your team. Gitignored — this file holds what is yours rather than the\n"
        "# project's, and is deep-merged over config.yaml at load time.\n"
        "#\n"
        "# Written by the page's \"Save team\" button (run `fpl serve`). Editing it\n"
        "# by hand is fine too; use the page's \"Copy squad\" to refresh `players`.\n"
    )
    body = yaml.safe_dump(existing, sort_keys=False, allow_unicode=True)
    path.write_text(header + body, encoding="utf-8")
    log.info("wrote %s (%d players)", path, len(squad["players"]))
    return path


class _Handler(SimpleHTTPRequestHandler):
    """Static files from the frontend directory, plus the two config endpoints."""

    config: Config  # bound onto the class by serve()

    # The page is UTF-8 (pound signs, dashes); say so, or a browser fed a bare
    # `text/html` over HTTP guesses Latin-1 and renders "£" as "Â£".
    extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        ".html": "text/html; charset=utf-8",
        ".js": "text/javascript; charset=utf-8",
        ".css": "text/css; charset=utf-8",
    }

    def _json(self, status: int, body: dict[str, Any]) -> None:
        data = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802 - http.server's naming
        if self.path.split("?", 1)[0] == "/api/config":
            self._json(200, read_local(self.config))
            return
        super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        if self.path.split("?", 1)[0] != "/api/config":
            self._json(404, {"ok": False, "error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(length) or b"null")
            path = write_local(self.config, payload)
        except ValueError as exc:
            self._json(400, {"ok": False, "error": str(exc)})
            return
        self._json(200, {"ok": True, "path": str(path)})

    def end_headers(self) -> None:
        # The page is rebuilt often; never let the browser show a stale copy.
        if not self.path.startswith("/api/"):
            self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt: str, *args: Any) -> None:
        log.debug("%s - " + fmt, self.address_string(), *args)


def serve(
    config: Config,
    directory: Path,
    page: str,
    port: int = DEFAULT_PORT,
    open_browser: bool = True,
) -> None:
    """Serve `directory` on the loopback address until interrupted."""
    # http.server instantiates the handler per request, so the config rides on
    # the class rather than on an instance we never get to construct.
    _Handler.config = config
    handler = partial(_Handler, directory=str(directory))

    server = ThreadingHTTPServer((HOST, port), handler)
    url = f"http://{HOST}:{server.server_address[1]}/{page}"
    print(f"serving   : {url}")
    print("Pick your team on the page and press \"Save team\" to write config.local.yaml.")
    print("Press Ctrl+C to stop.")

    if open_browser:
        # Give the socket a moment to be listening before the browser asks.
        threading.Timer(0.3, webbrowser.open, args=(url,)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        server.server_close()
