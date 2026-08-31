"""Tests for the squad-picker export.

The interesting logic here is name resolution. FPL surnames collide — there are
two Jameses and two Ouattaras in the pool, in different positions — so matching
name-by-name gets a real squad wrong. The resolver uses the squad's required
shape to break those ties, and that is what these tests pin down.

They run against a small fake client, so no network is needed.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.frontend import DATA_PLACEHOLDER, TEMPLATE_PATH, _normalise, resolve_squad


class FakeClient:
    """Just enough of `FPLClient` for `resolve_squad`."""

    def __init__(self, rows: list[tuple]):
        self._df = pd.DataFrame(
            rows, columns=["id", "web_name", "first_name", "second_name", "element_type"]
        )

    def players(self) -> pd.DataFrame:
        return self._df.copy()


# id, web_name, first, second, element_type (1=GK 2=DEF 3=MID 4=FWD)
POOL = [
    (1, "Raya", "David", "Raya Martín", 1),
    (2, "Steele", "Jason", "Steele", 1),
    (3, "James", "Reece", "James", 2),          # DEF
    (4, "James", "Daniel", "James", 3),         # MID — same surname, other position
    (5, "Calafiori", "Riccardo", "Calafiori", 2),
    (6, "Pedro Porro", "Pedro", "Porro Sauceda", 2),
    (7, "Ouattara", "Abdoul", "Ouattara", 2),   # DEF
    (8, "O.Dango", "Dango", "Ouattara", 3),     # MID — same surname again
    (9, "Herrington", "Lucas", "Herrington", 2),
    (10, "Gibbs-White", "Morgan", "Gibbs-White", 3),
    (11, "Ndiaye", "Iliman", "Ndiaye", 3),
    (12, "Cherki", "Rayan", "Cherki", 3),
    (13, "Szoboszlai", "Dominik", "Szoboszlai", 3),
    (14, "Mbeumo", "Bryan", "Mbeumo", 3),
    (15, "Haaland", "Erling", "Haaland", 4),
    (16, "João Pedro", "João Pedro", "Junqueira de Jesus", 4),
    (17, "Walle Egeli", "Sindre", "Walle Egeli", 4),
]

FULL_SQUAD = [
    "haaland", "joao pedro", "gibbs-white", "ndiaye", "cherki", "szoboszlai",
    "mbeumo", "james", "calafiori", "pedro porro", "raya", "steele",
    "ouattara", "herrington", "walle egell",
]


@pytest.fixture
def client() -> FakeClient:
    return FakeClient(POOL)


def test_normalise_strips_accents_and_case() -> None:
    assert _normalise("João Pedro") == "joao pedro"
    assert _normalise("  Raya Martín ") == "raya martin"


def test_normalise_folds_letters_that_nfkd_leaves_alone() -> None:
    """Ø, ł and ß carry their mark in the glyph, so NFKD alone does not fold them."""
    assert _normalise("Ødegaard") == "odegaard"
    assert _normalise("Fabiański") == "fabianski"
    assert _normalise("Højlund") == "hojlund"
    assert _normalise("Weiß") == "weiss"


def test_resolves_a_full_squad_to_a_legal_shape(client: FakeClient) -> None:
    ids = resolve_squad(FULL_SQUAD, client)
    assert len(ids) == 15
    assert len(set(ids)) == 15, "a player cannot be picked twice"

    positions = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
    lookup = {r[0]: positions[r[4]] for r in POOL}
    shape = {pos: sum(1 for i in ids if lookup[i] == pos) for pos in ("GK", "DEF", "MID", "FWD")}
    assert shape == {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}


def test_ambiguous_surnames_resolve_by_squad_shape(client: FakeClient) -> None:
    """`james` and `ouattara` each match two players; only one pair is legal."""
    ids = resolve_squad(FULL_SQUAD, client)
    by_name = dict(zip(FULL_SQUAD, ids))

    assert by_name["james"] == 3, "should be Reece James (DEF), not Daniel James (MID)"
    assert by_name["ouattara"] == 7, "should be Abdoul Ouattara (DEF), not Dango (MID)"


def test_misspellings_still_match(client: FakeClient) -> None:
    """`walle egell` is a typo for `Walle Egeli`, and has to survive it."""
    ids = resolve_squad(["walle egell"], client)
    assert ids == [17]


def test_accented_names_match_unaccented_input(client: FakeClient) -> None:
    assert resolve_squad(["joao pedro"], client) == [16]
    assert resolve_squad(["raya martin"], client) == [1]


def test_template_carries_the_data_placeholder() -> None:
    """The exporter injects the player pool here; without it the page is inert."""
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    assert template.count(DATA_PLACEHOLDER) == 1

    # The page is embedded in a document shell at publish time, so it must not
    # bring its own.
    lowered = template.lower()
    for tag in ("<!doctype", "<html", "<head>", "<body"):
        assert tag not in lowered, f"template should not contain {tag}"


def test_team_defence_joins_seasons_on_short_name() -> None:
    """Clubs are matched across seasons by short name, since ids are reassigned.

    Uses the local cache and skips when cold, like the other real-data tests.
    """
    from src.config import load_config
    from src.data.fpl_api import FPLClient
    from src.frontend import ELITE_DEFENCE_RATIO, team_defence

    config = load_config()
    season = config.season_history[-1]
    cached = config.cache_dir / "historical" / season / "gws" / "merged_gw.csv"
    if not cached.exists():
        pytest.skip("historical season not cached")

    result = team_defence(config, FPLClient())
    teams = result["teams"]

    # Promoted clubs have no Premier League record and must simply be absent
    # rather than defaulting to something flattering.
    assert 12 <= len(teams) <= 20

    for entry in teams.values():
        assert 0.0 <= entry["csRate"] <= 1.0
        assert entry["cs"] <= entry["games"]
        assert entry["elite"] == (entry["csRate"] >= result["eliteThreshold"])

    # Both are rounded for the page, so compare within that rounding.
    assert result["eliteThreshold"] == pytest.approx(
        result["leagueMeanCsRate"] * ELITE_DEFENCE_RATIO, abs=1e-3
    )

    # The bar has to actually discriminate: a rule that flags everyone, or
    # no one, would silently disable the double-up exception.
    elite = [e for e in teams.values() if e["elite"]]
    assert 1 <= len(elite) <= 5
    assert max(e["csRate"] for e in teams.values()) == max(e["csRate"] for e in elite)
