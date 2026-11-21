"""Check OddsPapi account usage via the free /account endpoint, so scaling up the
ingest is based on a real number instead of a guess.

This runs in a PUBLIC repo's Actions log, so it prints only plan, quota and book
names -- never the raw response (which contains the account email and key).

    ODDSPAPI_KEY=... python -m scripts.oddspapi_account
"""
from __future__ import annotations

import os

import requests

BASE = "https://api.oddspapi.io/v4"
KEY = os.environ.get("ODDSPAPI_KEY", "")
WATCH = ["pinnacle", "ps3838", "sbobet", "bet365", "williamhill", "paddypower", "ladbrokes",
         "coral", "skybet", "betfred", "unibet", "bwin", "betway", "888sport", "marathonbet",
         "betmgm", "boylesports", "betfair-ex"]


def main() -> None:
    if not KEY:
        raise SystemExit("set ODDSPAPI_KEY")
    r = requests.get(f"{BASE}/account", params={"apiKey": KEY}, timeout=30)
    print(f"HTTP {r.status_code}")
    if not r.ok:
        print(r.text[:200].replace(KEY, "***"))
        return
    j = r.json()
    for s in j.get("subscriptions", []):
        if not s.get("is_active"):
            continue
        books = s.get("bookmakers", {}) or {}
        print(f"plan={s.get('plan')} requests_used={s.get('request_count')}/{s.get('request_limit')} "
              f"valid_from={str(s.get('valid_from'))[:10]} valid_until={s.get('valid_until')} "
              f"auto_renew={s.get('auto_renew')}")
        print(f"bookmakers available: {len(books)}")
        print("watched books present:", ", ".join(b for b in WATCH if b in books) or "none")
        print("watched books missing:", ", ".join(b for b in WATCH if b not in books) or "none")


if __name__ == "__main__":
    main()
