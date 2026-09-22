"""One-off: check OddsPapi account usage via the free /account endpoint, so
scaling up the ingest is based on a real number instead of a guess.

    ODDSPAPI_KEY=... python -m scripts.oddspapi_account
"""
from __future__ import annotations

import os

import requests

BASE = "https://api.oddspapi.io/v4"
KEY = os.environ.get("ODDSPAPI_KEY", "")


def main() -> None:
    if not KEY:
        raise SystemExit("set ODDSPAPI_KEY")
    r = requests.get(f"{BASE}/account", params={"apiKey": KEY}, timeout=30)
    print(f"HTTP {r.status_code}")
    print(r.text)


if __name__ == "__main__":
    main()
