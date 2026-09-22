"""Vet OddsPapi's free tier with a handful of requests. Prints a SUMMARY ONLY.

Answers, from real responses rather than the vendor's docs:
  1. how far back does the archive go (do old fixtures have odds history)?
  2. what is the real time resolution (gap between line moves, per bookmaker)?
  3. does the free tier's 250-request quota tick down on /historical-odds calls?

Nothing is written to disk: their terms on storing or redistributing data are
not stated, and this repo is public. Needs env ODDSPAPI_KEY (a GitHub secret in
the workflow). Costs about 8 of the 250 monthly requests.

    ODDSPAPI_KEY=... python -m scripts.oddspapi_probe
"""
from __future__ import annotations

import os
import statistics
import time
from datetime import date, datetime, timedelta

import requests

BASE = "https://api.oddspapi.io/v4"
KEY = os.environ.get("ODDSPAPI_KEY", "")
# Free-tier account (checked 2026-09-22): bookmakers access actually covers
# essentially every book, including the sharp references this project uses
# (pinnacle, bet365, betfair-ex, williamhill, marathonbet). The first probe's
# sample fixture (Brazilian Serie B) just happened not to have those specific
# three in its own coverage - that was per-fixture, not an account limit.
# betfair-ex is excluded here: the API rejects it in a multi-bookmaker call
# ("you must provide only one bookmaker and exactly one outcomeId" - it's an
# exchange, priced per outcome, not a fixed-odds book like the others).
BOOKS = os.environ.get("ODDSPAPI_BOOKS", "pinnacle,bet365")
DEPTHS_DAYS = [3, 30, 180, 365, 730]
REQUEST_DELAY_S = 3.0   # be gentle: 429s showed up after just 2 calls with no delay
used = 0


def get(path: str, **params):
    """GET with a fixed delay before each call and one retry-with-backoff on 429
    (the vendor's docs don't distinguish a monthly quota 429 from a burst-rate
    429; the dashboard's 5/250 usage after our first probe run showed the quota
    itself was nowhere near hit, so treat 429 as a transient rate limit)."""
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
    quota = {k: v for k, v in r.headers.items()
             if any(s in k.lower() for s in ("limit", "remaining", "quota", "credit"))}
    return r, quota


def _fixtures(payload) -> list[dict]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for v in payload.values():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                return v
    return []


def _walk_history(hist: dict) -> dict[str, list[datetime]]:
    """bookmaker -> sorted timestamps of every price point (all markets)."""
    out: dict[str, list[datetime]] = {}
    for book, b in (hist.get("bookmakers") or {}).items():
        ts: list[datetime] = []
        for m in (b.get("markets") or {}).values():
            for o in (m.get("outcomes") or {}).values():
                for series in (o.get("players") or {}).values():
                    for p in series:
                        if p.get("createdAt"):
                            ts.append(datetime.fromisoformat(p["createdAt"].replace("Z", "+00:00")))
        out[book] = sorted(ts)
    return out


def main() -> None:
    if not KEY:
        raise SystemExit("set ODDSPAPI_KEY")
    print("== 1. archive depth: soccer fixtures at increasing age, looking for a Premier League match ==")
    fallback = None
    epl = None
    first_fixture_raw = None
    for d in DEPTHS_DAYS:
        start = date.today() - timedelta(days=d)
        r, quota = get("/fixtures", sportId=10, **{"from": start.isoformat(),
                                                   "to": (start + timedelta(days=2)).isoformat()})
        fx = _fixtures(r.json()) if r.ok else []
        if fx and first_fixture_raw is None:
            first_fixture_raw = fx[0]
        with_odds = [f for f in fx if f.get("hasOdds", True)]
        print(f"  {d:>4}d ago  HTTP {r.status_code}  fixtures={len(fx)}  hasOdds~={len(with_odds)}"
              f"{'' if r.ok else '  body: ' + r.text[:200]}")
        # hasOdds is unreliable (confirmed 2026-09-22: a hasOdds=false fixture
        # still returned real historical-odds data) - scan ALL fixtures, not
        # just the ones flagged with_odds.
        if epl is None:
            epl = next((f for f in fx if str(f.get("tournamentName", "")).strip().lower() == "premier league"
                       and "england" in str(f.get("categoryName", "")).lower()), None)
        if fallback is None and fx:
            fallback = fx[0]

    if first_fixture_raw is not None:
        print(f"\nsample raw fixture (keys + values, to see what hasOdds means here):\n  {first_fixture_raw}")
    pick = epl or fallback
    if pick is None:
        raise SystemExit("no fixtures returned at any depth (not even without odds)")
    print(f"\n{'Premier League fixture found' if epl else 'no EPL fixture in range - using fallback'}: "
          f"{pick.get('participant1Name')} v {pick.get('participant2Name')} ({pick.get('tournamentName')})")

    print(f"\n== 2. resolution: /historical-odds for {pick.get('participant1Name')} v "
          f"{pick.get('participant2Name')} ({pick.get('tournamentName')}), bookmakers={BOOKS} ==")
    r, quota = get("/historical-odds", fixtureId=pick["fixtureId"], bookmakers=BOOKS)
    print(f"  HTTP {r.status_code}  quota headers: {quota or 'none exposed'}")
    if not r.ok:
        raise SystemExit(r.text[:300])
    per_book = _walk_history(r.json())
    if not per_book:
        print(f"  no bookmakers in {BOOKS} had data for this fixture. raw keys: "
              f"{list(r.json().get('bookmakers', {}).keys())[:20]}")
    for book, ts in per_book.items():
        if len(ts) < 2:
            print(f"  {book}: {len(ts)} price points")
            continue
        gaps = [(b - a).total_seconds() for a, b in zip(ts, ts[1:], strict=False)]
        print(f"  {book}: {len(ts)} points, {ts[0]:%Y-%m-%d %H:%M} .. {ts[-1]:%Y-%m-%d %H:%M}, "
              f"median gap {statistics.median(gaps):.0f}s, min {min(gaps):.0f}s, max {max(gaps)/3600:.1f}h")

    print("\n== 3. does the quota tick down? ==")
    _, q1 = get("/fixtures", sportId=10, **{"from": date.today().isoformat(),
                                            "to": (date.today() + timedelta(days=1)).isoformat()})
    print(f"  after the history call: {q1 or 'no quota headers exposed - check the dashboard'}")
    print(f"\nrequests used by this probe: {used} (free tier: 250 a month)")


if __name__ == "__main__":
    main()
