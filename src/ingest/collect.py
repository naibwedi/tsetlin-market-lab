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
import calendar
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


def _hdr_int(resp: requests.Response, name: str) -> int | None:
    try:
        return int(float(resp.headers.get(name)))
    except (TypeError, ValueError):
        return None


def within_pace(remaining: int | None, quota: int, reserve: int, now: datetime) -> bool:
    """May we spend a credit now?

    The free tier resets on the 1st of each month (seen: 480 -> 496 across
    2026-08-31 -> 09-01). We spend only while at or ahead of a linear burn-down of
    the monthly quota, so dropped GitHub runs and extra leagues can't drain it early.
    """
    if remaining is None:
        return True
    if remaining <= reserve:
        return False
    days = calendar.monthrange(now.year, now.month)[1]
    elapsed = (now.day - 1) + now.hour / 24 + now.minute / 1440
    return remaining >= quota * (1 - elapsed / days)


def has_event_in_window(events: list[dict], now: datetime, lo_h: float, hi_h: float) -> bool:
    for ev in events:
        ct = datetime.fromisoformat(ev["commence_time"].replace("Z", "+00:00"))
        if lo_h <= (ct - now).total_seconds() / 3600.0 <= hi_h:
            return True
    return False


def theoddsapi_free(cfg: dict) -> list[dict]:
    if not API_KEY:
        print("  theoddsapi_free: ODDS_API_KEY not set - skipping")
        return []
    rows: list[dict] = []
    now = datetime.now(timezone.utc)
    quota = int(cfg.get("monthly_quota", 500))
    reserve = int(cfg.get("reserve", 15))

    # /sports is a free endpoint: read the live quota from its header first.
    remaining: int | None = None
    try:
        r0 = requests.get(f"{BASE}/sports", params={"apiKey": API_KEY}, timeout=30)
        remaining = _hdr_int(r0, "x-requests-remaining")
    except requests.RequestException as e:
        print(f"  theoddsapi_free: quota probe failed ({e}); continuing without pacing")
    print(f"  theoddsapi_free: quota remaining={remaining} (before any paid call)")

    for sport in cfg["sports"]:
        if cfg.get("pace", True) and not within_pace(remaining, quota, reserve, now):
            print(f"  theoddsapi_free {sport}: skipped - ahead of monthly pace (remaining={remaining})")
            continue
        # /events is free: don't pay for a snapshot when nothing kicks off in the window.
        try:
            re_ = requests.get(f"{BASE}/sports/{sport}/events", params={"apiKey": API_KEY}, timeout=30)
            re_.raise_for_status()
            if not has_event_in_window(re_.json(), now, cfg["min_hours_to_commence"],
                                       cfg["max_hours_to_commence"]):
                print(f"  theoddsapi_free {sport}: skipped - no events in window "
                      f"(remaining={_hdr_int(re_, 'x-requests-remaining')})")
                continue
        except (requests.RequestException, ValueError, KeyError) as e:
            print(f"  theoddsapi_free {sport}: events check failed ({e}); collecting anyway")
        params = {
            "apiKey": API_KEY,
            "markets": cfg["markets"],
            "oddsFormat": cfg.get("odds_format", "decimal"),
        }
        # An explicit bookmaker list (<= 10) is billed as ONE region, so we can keep
        # the sharp references (Pinnacle, Betfair) without paying for a second region.
        if cfg.get("bookmakers"):
            params["bookmakers"] = ",".join(cfg["bookmakers"])
        else:
            params["regions"] = cfg["regions"]
        try:
            r = requests.get(f"{BASE}/sports/{sport}/odds", params=params, timeout=30)
            r.raise_for_status()
        except requests.HTTPError as e:
            print(f"  theoddsapi_free {sport}: {e}")
            continue
        remaining = _hdr_int(r, "x-requests-remaining")
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
        n_new = len(part)
        if path.exists():
            part = pd.concat([pd.read_parquet(path), part], ignore_index=True)
        part.drop_duplicates(
            subset=["match_id", "snapshot_ts", "bookmaker", "outcome_name"]
        ).to_parquet(path, index=False)
        print(f"  -> {path}  (+{n_new} rows this run)")


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
