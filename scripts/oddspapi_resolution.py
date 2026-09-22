"""De-duplicated resolution check for OddsPapi's /historical-odds.

The probe's raw point counts (e.g. "bet365: 368,591 points") sum every
market x outcome combination, so they overstate the real update rate several
times over. This finds the actual match-winner market (3-way: home/draw/away,
not a player prop) and computes the true gap between DISTINCT snapshot
instants - the number that answers "how often does the price we care about
actually change".

Prints a SUMMARY ONLY, same as the probe: nothing is written to disk, because
OddsPapi's terms ("may not resell, repackage, or redistribute our data as a
standalone product") mean raw price data stays out of this public repo.

    ODDSPAPI_KEY=... python -m scripts.oddspapi_resolution
"""
from __future__ import annotations

import os
import statistics
import time
from datetime import date, datetime, timedelta

import requests

BASE = "https://api.oddspapi.io/v4"
KEY = os.environ.get("ODDSPAPI_KEY", "")
BOOK = os.environ.get("ODDSPAPI_BOOK", "bet365")
REQUEST_DELAY_S = 3.0
# names that mean "the main 1x2 / match-winner market" across odds vendors
WINNER_MARKET_HINTS = ("moneyline", "match_winner", "matchwinner", "1x2", "3way",
                       "three_way", "full_time_result", "winner", "match_odds")
used = 0


def get(path: str, **params):
    global used
    time.sleep(REQUEST_DELAY_S)
    r = requests.get(f"{BASE}{path}", params={"apiKey": KEY, **params}, timeout=60)
    used += 1
    if r.status_code == 429:
        wait = float(r.headers.get("Retry-After", 15))
        print(f"  (429, retrying once after {wait:.0f}s)")
        time.sleep(wait)
        r = requests.get(f"{BASE}{path}", params={"apiKey": KEY, **params}, timeout=60)
        used += 1
    return r


def _fixtures(payload) -> list[dict]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for v in payload.values():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                return v
    return []


def find_epl_fixture() -> dict:
    for d in (3, 30, 180, 365, 730):
        start = date.today() - timedelta(days=d)
        r = get("/fixtures", sportId=10, **{"from": start.isoformat(),
                                            "to": (start + timedelta(days=2)).isoformat()})
        fx = _fixtures(r.json()) if r.ok else []
        epl = next((f for f in fx if str(f.get("tournamentName", "")).strip().lower() == "premier league"
                   and "england" in str(f.get("categoryName", "")).lower()), None)
        if epl:
            return epl
    raise SystemExit("no Premier League fixture found in any depth window")


def _outcome_timestamps(outcome: dict) -> list[datetime]:
    """An outcome's price series lives under 'players' even for team markets
    (a generic container key, not literally per-player) - flatten all of it."""
    ts: list[datetime] = []
    for series in (outcome.get("players") or {}).values():
        for p in series:
            if p.get("createdAt"):
                ts.append(datetime.fromisoformat(p["createdAt"].replace("Z", "+00:00")))
    return sorted(ts)


def main() -> None:
    if not KEY:
        raise SystemExit("set ODDSPAPI_KEY")

    pick = find_epl_fixture()
    print(f"fixture: {pick.get('participant1Name')} v {pick.get('participant2Name')} "
          f"({pick.get('tournamentName')})")

    r = get("/historical-odds", fixtureId=pick["fixtureId"], bookmakers=BOOK)
    if not r.ok:
        raise SystemExit(f"HTTP {r.status_code}: {r.text[:300]}")
    hist = r.json()
    book_data = (hist.get("bookmakers") or {}).get(BOOK)
    if not book_data:
        raise SystemExit(f"no data for {BOOK}. available: {list((hist.get('bookmakers') or {}).keys())}")

    markets = book_data.get("markets") or {}
    print(f"\n{len(markets)} markets available for {BOOK}:")
    for key, m in markets.items():
        n_out = len(m.get("outcomes") or {})
        print(f"  {key}: {n_out} outcomes")

    # pick the 3-way match-winner market: by name hint first, else the first
    # market with exactly 3 outcomes (home/draw/away has no player-prop analog)
    winner_key = next((k for k in markets if any(h in k.lower() for h in WINNER_MARKET_HINTS)), None)
    if winner_key is None:
        winner_key = next((k for k, m in markets.items() if len(m.get("outcomes") or {}) == 3), None)
    if winner_key is None:
        raise SystemExit("could not identify the match-winner market from the keys above")
    print(f"\nusing market: {winner_key}")

    outcomes = markets[winner_key].get("outcomes") or {}
    all_snapshot_instants: set[datetime] = set()
    for out_key, out in outcomes.items():
        ts = _outcome_timestamps(out)
        print(f"  outcome {out_key}: {len(ts)} raw points")
        all_snapshot_instants.update(ts)

    snaps = sorted(all_snapshot_instants)
    print(f"\n{len(snaps)} DISTINCT snapshot instants across all outcomes of {winner_key} "
          f"(this de-dupes simultaneous home/draw/away updates into one snapshot each)")
    if len(snaps) >= 2:
        gaps_s = [(b - a).total_seconds() for a, b in zip(snaps, snaps[1:], strict=False)]
        days_covered = (snaps[-1] - snaps[0]).total_seconds() / 86400
        print(f"  span: {snaps[0]:%Y-%m-%d %H:%M} .. {snaps[-1]:%Y-%m-%d %H:%M}  ({days_covered:.1f} days)")
        print(f"  median gap: {statistics.median(gaps_s):.0f}s  "
              f"mean gap: {statistics.mean(gaps_s):.0f}s  "
              f"min gap: {min(gaps_s):.0f}s  max gap: {max(gaps_s)/3600:.1f}h")
        print(f"  snapshots/day (avg): {len(snaps)/max(days_covered, 0.01):.1f}")
        under_1h = sum(1 for g in gaps_s if g < 3600) / len(gaps_s)
        print(f"  fraction of gaps under 1 hour: {under_1h:.1%}  "
              f"(this is the number that matters for 'is this sub-hourly')")

    print(f"\nrequests used: {used} (free tier: 250 a month)")


if __name__ == "__main__":
    main()
