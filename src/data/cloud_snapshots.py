"""Snapshots captured by GitHub Actions, brought home.

`.github/workflows/snapshot.yml` runs `fpl snapshot` every six hours whether
or not a laptop is awake, and stores each capture as a workflow artifact named
for its gameweek (`snapshot-gw06`). Artifacts expire after 90 days, so they
are a delivery route, not storage: `fetch` downloads, for each gameweek, the
newest one if it is newer than the capture on disk, and the snapshot folder
stays the one place training reads from.

Done through the GitHub CLI, which handles authentication: artifact downloads
need a signed-in user even on a public repository. Anything that goes wrong —
`gh` missing, not signed in, no network — comes back as a reason rather than
an exception, because the callers are commands with a more important job to
finish.
"""

from __future__ import annotations

import io
import logging
import subprocess
import zipfile
from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from ..config import REPO_ROOT, Config
from .snapshots import keep_if_newer, local_captured_at

log = logging.getLogger(__name__)

ARTIFACT_PREFIX = "snapshot-gw"
# `gh` fills in {owner}/{repo} from the checkout's git remote.
ARTIFACTS = "repos/{owner}/{repo}/actions/artifacts"
TIMEOUT = 60

Runner = Callable[..., subprocess.CompletedProcess]


class CloudUnavailable(RuntimeError):
    """The artifacts could not be reached; the message says why, in a line."""


@dataclass
class FetchResult:
    stored: list[int] = field(default_factory=list)   # gameweeks written
    current: list[int] = field(default_factory=list)  # gameweeks already as new locally
    error: str | None = None

    def summary(self) -> str:
        if self.error and self.stored:
            got = ", ".join(f"GW{g}" for g in self.stored)
            return f"fetched {got} from the cloud, then stopped: {self.error}"
        if self.error:
            return f"cloud captures not fetched: {self.error}"
        if self.stored:
            return "fetched from the cloud: " + ", ".join(f"GW{g}" for g in self.stored)
        if self.current:
            return "cloud captures: nothing newer than what is here"
        return "cloud captures: none yet"


def _gh(args: list[str], run: Runner) -> bytes:
    try:
        done = run(["gh", *args], capture_output=True, cwd=REPO_ROOT, timeout=TIMEOUT)
    except FileNotFoundError:
        raise CloudUnavailable("the GitHub CLI (gh) is not installed") from None
    except subprocess.TimeoutExpired:
        raise CloudUnavailable("GitHub did not answer in time") from None
    if done.returncode != 0:
        err = (done.stderr or b"").decode("utf-8", "replace").strip()
        if "auth login" in err or "not logged" in err:
            raise CloudUnavailable("gh is not signed in — run `gh auth login` once")
        raise CloudUnavailable(err.splitlines()[-1] if err else f"gh exited with {done.returncode}")
    return done.stdout


def latest_artifacts(run: Runner = subprocess.run) -> dict[int, tuple[int, pd.Timestamp]]:
    """gameweek -> (artifact id, created at) of its newest unexpired capture."""
    out = _gh(
        ["api", "-X", "GET", ARTIFACTS, "--paginate", "-F", "per_page=100", "--jq",
         f'.artifacts[] | select(.expired | not) | select(.name | startswith("{ARTIFACT_PREFIX}"))'
         ' | [.id, .name, .created_at] | @tsv'],
        run,
    )
    newest: dict[int, tuple[int, pd.Timestamp]] = {}
    for line in out.decode("utf-8").splitlines():
        parts = line.strip().split("\t")
        if len(parts) != 3:
            continue
        artifact_id, name, created = parts
        try:
            gw = int(name[len(ARTIFACT_PREFIX):])
        except ValueError:
            continue
        when = pd.Timestamp(created)
        if gw not in newest or when > newest[gw][1]:
            newest[gw] = (int(artifact_id), when)
    return newest


def _download(artifact_id: int, run: Runner) -> bytes:
    """The snapshot JSON inside an artifact's zip."""
    archive = zipfile.ZipFile(io.BytesIO(_gh(["api", f"{ARTIFACTS}/{artifact_id}/zip"], run)))
    members = [n for n in archive.namelist() if n.endswith(".json")]
    if len(members) != 1:
        raise ValueError(f"artifact {artifact_id} holds {len(members)} snapshot files, expected one")
    return archive.read(members[0])


def fetch(config: Config, run: Runner = subprocess.run) -> FetchResult:
    """Bring home every cloud capture newer than the local one for its gameweek."""
    result = FetchResult()
    try:
        artifacts = latest_artifacts(run)
        for gw, (artifact_id, created) in sorted(artifacts.items()):
            local = local_captured_at(config, gw)
            # An artifact is uploaded after its capture, so a local copy taken
            # later than the upload is certainly the newer one: skip the download.
            if local is not None and local >= created:
                result.current.append(gw)
                continue
            stored = keep_if_newer(config, _download(artifact_id, run))
            (result.stored if stored is not None else result.current).append(gw)
    except CloudUnavailable as exc:
        result.error = str(exc)
    except (ValueError, KeyError, zipfile.BadZipFile) as exc:
        result.error = f"a cloud capture could not be read ({exc})"
    if result.error:
        log.info("cloud snapshots: %s", result.error)
    return result
