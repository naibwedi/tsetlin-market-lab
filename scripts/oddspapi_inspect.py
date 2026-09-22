"""One-off diagnostic: print the real field names inside one OddsPapi price
point, so the ingest parser (src/ingest/oddspapi.py) is built against the
actual response shape instead of a guess. Prints only, writes nothing.

    ODDSPAPI_KEY=... python -m scripts.oddspapi_inspect
"""
from __future__ import annotations

import os
import time
from datetime import date, timedelta

import requests

BASE = "https://api.oddspapi.io/v4"
KEY = os.environ.get("ODDSPAPI_KEY", "")


def get(path: str, **params):
    time.sleep(3)
    r = requests.get(f"{BASE}{path}", params={"apiKey": KEY, **params}, timeout=60)
    if r.status_code == 429:
        time.sleep(float(r.headers.get("Retry-After", 15)))
        r = requests.get(f"{BASE}{path}", params={"apiKey": KEY, **params}, timeout=60)
    return r


def _fixtures(payload) -> list[dict]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for v in payload.values():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                return v
    return []


def main() -> None:
    if not KEY:
        raise SystemExit("set ODDSPAPI_KEY")

    epl = None
    for d in (3, 30):
        start = date.today() - timedelta(days=d)
        r = get("/fixtures", sportId=10, **{"from": start.isoformat(),
                                            "to": (start + timedelta(days=2)).isoformat()})
        fx = _fixtures(r.json())
        epl = next((f for f in fx if str(f.get("tournamentName", "")).strip().lower() == "premier league"
                   and "england" in str(f.get("categoryName", "")).lower()), None)
        if epl:
            break
    if epl is None:
        raise SystemExit("no EPL fixture found")

    print("fixture:", epl.get("participant1Name"), "v", epl.get("participant2Name"), epl.get("fixtureId"))
    r = get("/historical-odds", fixtureId=epl["fixtureId"], bookmakers="bet365")
    hist = r.json()
    book = hist["bookmakers"]["bet365"]
    markets = book["markets"]
    mkey = next((k for k, m in markets.items() if len(m.get("outcomes") or {}) == 3), None)
    print("market key:", mkey)
    out = markets[mkey]["outcomes"]
    print("outcome keys:", list(out.keys()))
    first_out_key = list(out.keys())[0]
    players = out[first_out_key].get("players") or {}
    print("players keys:", list(players.keys()))
    pkey = list(players.keys())[0]
    series = players[pkey]
    print("series length:", len(series))
    print("FIRST POINT:", series[0])
    print("SECOND POINT:", series[1] if len(series) > 1 else None)
    print("outcome-level keys (excluding players):",
          {k: v for k, v in out[first_out_key].items() if k != "players"})
    print("market-level keys (excluding outcomes):",
          {k: v for k, v in markets[mkey].items() if k != "outcomes"})


if __name__ == "__main__":
    main()
