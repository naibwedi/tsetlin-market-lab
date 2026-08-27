"""Free self-collection poller.

Appends a live odds snapshot to ``data/raw/live/{sport}/{yyyy-mm-dd}.parquet``
in the same flat schema as the historical ingester. Designed to run unattended
on a schedule (GitHub Actions cron or a local task) so a real dataset accrues
for free while the rest of the lab is developed against synthetic data.

Adapters
--------
theoddsapi_free : The Odds API live endpoint, 500 req/month free tier.
betfair         : Betfair Exchange (free *delayed* app key). Stub - see
                  ``src/ingest/betfair.py``; enable in config once creds exist.

Run:  python -m src.ingest.collect --config config/collect.yaml
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

from src.common.config import load_yaml, resolve
from src.ingest.odds_api import flatten_snapshot

load_dotenv()

API_KEY = os.environ.get("ODDS_API_KEY", "")
BASE = os.environ.get("ODDS_API_BASE", "https://api.the-odds-api.com/v4")


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def theoddsapi_free(cfg: dict) -> list[dict]:
    if not API_KEY:
        print("  theoddsapi_free: ODDS_API_KEY not set - skipping")
        return []
    rows: list[dict] = []
    now = datetime.now(timezone.utc)
    for sport in cfg["sports"]:
        try:
            r = requests.get(
                f"{BASE}/sports/{sport}/odds",
                params={
                    "apiKey": API_KEY,
                    "regions": cfg["regions"],
                    "markets": cfg["markets"],
                    "oddsFormat": cfg.get("odds_format", "decimal"),
                },
                timeout=30,
            )
            r.raise_for_status()
        except requests.HTTPError as e:
            print(f"  theoddsapi_free {sport}: {e}")
            continue
        remaining = r.headers.get("x-requests-remaining")
        events = r.json()
        kept = 0
        for ev in events:
            ct = datetime.fromisoformat(ev["commence_time"].replace("Z", "+00:00"))
            hrs = (ct - now).total_seconds() / 3600.0
            if not (cfg["min_hours_to_commence"] <= hrs <= cfg["max_hours_to_commence"]):
                continue
            ev["_snapshot_ts"] = _now_iso()
            rows.extend(flatten_snapshot(sport, ev))
            kept += 1
        print(f"  theoddsapi_free {sport}: {kept} events, credits remaining~={remaining}")
    return rows


def betfair(cfg: dict) -> list[dict]:
    try:
        from src.ingest.betfair import collect_snapshot
    except Exception as e:  # noqa: BLE001
        print(f"  betfair adapter unavailable: {e}")
        return []
    return collect_snapshot(cfg)


ADAPTERS = {"theoddsapi_free": theoddsapi_free, "betfair": betfair}


def _append(out_dir: Path, rows: list[dict]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    for (sport, day), part in df.assign(
        _day=pd.to_datetime(df["snapshot_ts"]).dt.strftime("%Y-%m-%d")
    ).groupby(["sport", "_day"]):
        d = out_dir / sport
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{day}.parquet"
        part = part.drop(columns="_day")
        if path.exists():
            part = pd.concat([pd.read_parquet(path), part], ignore_index=True)
        part.drop_duplicates(
            subset=["match_id", "snapshot_ts", "bookmaker", "outcome_name"]
        ).to_parquet(path, index=False)
        print(f"  -> {path}  (+{len(rows)} rows this run)")


def run(config_path: str = "config/collect.yaml") -> None:
    cfg = load_yaml(config_path)
    out_dir = resolve(cfg["out_dir"])
    all_rows: list[dict] = []
    for name in cfg["adapters"]:
        fn = ADAPTERS.get(name)
        if fn is None:
            print(f"  unknown adapter: {name}")
            continue
        all_rows.extend(fn(cfg.get(name, {})))
    _append(out_dir, all_rows)
    print(f"collect done: {len(all_rows)} rows at {_now_iso()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/collect.yaml")
    run(ap.parse_args().config)
