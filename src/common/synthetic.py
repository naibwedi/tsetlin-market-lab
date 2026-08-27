"""Generate a synthetic raw-odds dataset with a *known* lead/lag structure.

Used by tests and for dry-running the pipeline before the real feed is wired up.

Ground truth injected:
    * "pinnacle" is the leader: it random-walks its fair prob.
    * Follower books copy pinnacle's move ~1-2 snapshots later, with noise.
    * One lazy book ("slowbook") updates rarely.
A model that learns "follower moves next iff pinnacle just moved" should beat
the majority-class baseline. That is the sanity target for the whole lab.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

LEADER = "pinnacle"
FOLLOWERS = ["bet365", "williamhill", "unibet", "marathonbet", "betfair_ex_eu"]
LAZY = "slowbook"
BOOKS = [LEADER, *FOLLOWERS, LAZY]


def _price_from_probs(p_home: float, p_draw: float, p_away: float, vig: float) -> tuple[float, float, float]:
    tot = 1.0 + vig
    return (tot / p_home, tot / p_draw, tot / p_away)


def make_raw(
    n_matches: int = 40,
    snapshots_per_match: int = 60,
    step_minutes: int = 5,
    seed: int = 0,
    sport: str = "soccer_epl",
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    base_ts = datetime(2026, 3, 1, tzinfo=timezone.utc)

    for m in range(n_matches):
        commence = base_ts + timedelta(days=m, hours=19)
        start = commence - timedelta(minutes=step_minutes * (snapshots_per_match - 1))
        match_id = f"m{m:04d}"

        # Leader's fair-prob path (home/draw/away), a bounded random walk.
        p_home = np.clip(0.30 + 0.30 * rng.random(), 0.15, 0.7)
        p_draw = np.clip(0.24 + 0.05 * rng.standard_normal(), 0.15, 0.34)
        leader_home = []
        for _ in range(snapshots_per_match):
            if rng.random() < 0.25:  # leader moves 25% of snapshots
                p_home = float(np.clip(p_home + 0.01 * rng.standard_normal(), 0.1, 0.8))
            leader_home.append(p_home)
        leader_home = np.array(leader_home)

        # Per-book state.
        book_home = {b: leader_home[0] for b in BOOKS}
        book_vig = {b: 0.04 + 0.03 * rng.random() for b in BOOKS}
        lag = {b: rng.integers(1, 3) for b in FOLLOWERS}

        for i in range(snapshots_per_match):
            ts = (start + timedelta(minutes=step_minutes * i)).strftime("%Y-%m-%dT%H:%M:%SZ")
            for b in BOOKS:
                if b == LEADER:
                    book_home[b] = leader_home[i]
                elif b == LAZY:
                    if rng.random() < 0.05:
                        book_home[b] = leader_home[i] + 0.004 * rng.standard_normal()
                else:
                    src = leader_home[max(0, i - lag[b])]
                    if abs(src - book_home[b]) > 0.004 and rng.random() < 0.8:
                        book_home[b] = float(src + 0.003 * rng.standard_normal())

                ph = float(np.clip(book_home[b], 0.08, 0.85))
                pd_ = float(np.clip(p_draw + 0.01 * rng.standard_normal(), 0.12, 0.36))
                pa = max(0.05, 1.0 - ph - pd_)
                oh, od, oa = _price_from_probs(ph, pd_, pa, book_vig[b])
                for name, price in (("Home", oh), ("Draw", od), ("Away", oa)):
                    rows.append(
                        {
                            "sport": sport,
                            "snapshot_ts": ts,
                            "match_id": match_id,
                            "commence_time": commence.strftime("%Y-%m-%dT%H:%M:%SZ"),
                            "home_team": f"Home{m}",
                            "away_team": f"Away{m}",
                            "bookmaker": b,
                            "book_last_update": ts,
                            "market": "h2h",
                            "outcome_name": name,
                            "price": round(price, 3),
                        }
                    )
    return pd.DataFrame(rows)
