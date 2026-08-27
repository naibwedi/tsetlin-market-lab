"""Pull historical multi-bookmaker odds snapshots from The Odds API (v4).

Docs: https://the-odds-api.com/liveapi/guides/v4/#historical-odds

Design notes
------------
* We walk each match's pre-kickoff window at a fixed step (default 5 min) and
  request that single event's odds snapshot (``eventIds`` filter keeps payloads
  small and lets us resume per-event).
* Raw rows are flattened to one record per (snapshot_ts, match, bookmaker,
  outcome) and written to ``data/raw/{sport}/{yyyy-mm}.parquet``.
* Idempotent: a (sport, event_id, snapshot_ts) already present is skipped.

Quota math (read before you run against the Business plan!)
    6 leagues x ~40 matches/mo x (24h / 5min = 288 snapshots) ~= 69k requests
    per month of history. The Business plan allows 200k requests/month. Narrow
    ``window_hours_before_kickoff`` or ``leagues`` if that is too much.
"""
from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

from src.common.config import load_yaml, resolve

load_dotenv()

API_KEY = os.environ.get("ODDS_API_KEY", "")
BASE = os.environ.get("ODDS_API_BASE", "https://api.the-odds-api.com/v4")


@dataclass
class Budget:
    """Track request count and the provider's remaining-quota headers."""

    max_requests: int
    used: int = 0
    last_remaining: int | None = None

    def check(self) -> None:
        if self.used >= self.max_requests:
            raise RuntimeError(f"hit max_requests_per_run={self.max_requests}")

    def note(self, resp: requests.Response) -> None:
        self.used += 1
        rem = resp.headers.get("x-requests-remaining")
        if rem is not None:
            self.last_remaining = int(float(rem))


def _get(path: str, params: dict, budget: Budget, pause: float) -> requests.Response:
    budget.check()
    p = {"apiKey": API_KEY, **params}
    resp = requests.get(f"{BASE}{path}", params=p, timeout=30)
    budget.note(resp)
    time.sleep(pause)
    resp.raise_for_status()
    return resp


def discover_events(sport: str, date_iso: str, budget: Budget, pause: float) -> list[dict]:
    """Historical events snapshot for a sport at a point in time."""
    resp = _get(f"/historical/sports/{sport}/events", {"date": date_iso}, budget, pause)
    payload = resp.json()
    return payload.get("data", payload if isinstance(payload, list) else [])


def event_odds_snapshot(
    sport: str, event_id: str, date_iso: str, cfg: dict, budget: Budget, pause: float
) -> dict | None:
    resp = _get(
        f"/historical/sports/{sport}/odds",
        {
            "date": date_iso,
            "eventIds": event_id,
            "regions": cfg["regions"],
            "markets": cfg["markets"],
            "oddsFormat": cfg.get("odds_format", "decimal"),
        },
        budget,
        pause,
    )
    data = resp.json().get("data", [])
    if not data:
        return None
    snap = data[0]
    snap["_snapshot_ts"] = resp.json().get("timestamp", date_iso)
    return snap


def flatten_snapshot(sport: str, snap: dict) -> list[dict]:
    rows: list[dict] = []
    snapshot_ts = snap.get("_snapshot_ts")
    for bm in snap.get("bookmakers", []):
        for mkt in bm.get("markets", []):
            if mkt.get("key") != "h2h":
                continue
            for outcome in mkt.get("outcomes", []):
                rows.append(
                    {
                        "sport": sport,
                        "snapshot_ts": snapshot_ts,
                        "match_id": snap.get("id"),
                        "commence_time": snap.get("commence_time"),
                        "home_team": snap.get("home_team"),
                        "away_team": snap.get("away_team"),
                        "bookmaker": bm.get("key"),
                        "book_last_update": bm.get("last_update"),
                        "market": mkt.get("key"),
                        "outcome_name": outcome.get("name"),
                        "price": outcome.get("price"),
                    }
                )
    return rows


def _snapshot_times(commence: datetime, hours_before: int, step_min: int) -> list[datetime]:
    start = commence - timedelta(hours=hours_before)
    out, t = [], start
    while t <= commence:
        out.append(t)
        t += timedelta(minutes=step_min)
    return out


def _month_key(ts: str) -> str:
    return ts[:7]  # yyyy-mm


def run(config_path: str = "config/ingest.yaml") -> None:
    if not API_KEY:
        raise SystemExit("ODDS_API_KEY not set (copy .env.example -> .env).")
    cfg = load_yaml(config_path)
    out_dir = resolve(cfg["out_dir"])
    budget = Budget(max_requests=int(cfg.get("max_requests_per_run", 20000)))
    pause = float(cfg.get("request_pause_seconds", 0.3))

    d_from = datetime.fromisoformat(cfg["date_from"].replace("Z", "+00:00"))
    d_to = datetime.fromisoformat(cfg["date_to"].replace("Z", "+00:00"))

    for sport in cfg["leagues"]:
        sport_dir = out_dir / sport
        sport_dir.mkdir(parents=True, exist_ok=True)
        seen = _load_seen(sport_dir)
        buffers: dict[str, list[dict]] = {}

        # Discover matches by probing the events endpoint once per day in range.
        day = d_from
        events: dict[str, datetime] = {}
        while day <= d_to:
            for ev in discover_events(sport, _iso(day), budget, pause):
                ct = datetime.fromisoformat(ev["commence_time"].replace("Z", "+00:00"))
                if d_from <= ct <= d_to:
                    events[ev["id"]] = ct
            day += timedelta(days=1)
        print(f"[{sport}] {len(events)} matches in range")

        for event_id, commence in sorted(events.items(), key=lambda kv: kv[1]):
            for t in _snapshot_times(
                commence,
                int(cfg["window_hours_before_kickoff"]),
                int(cfg["snapshot_step_minutes"]),
            ):
                key = (event_id, _iso(t))
                if key in seen:
                    continue
                try:
                    snap = event_odds_snapshot(sport, event_id, _iso(t), cfg, budget, pause)
                except requests.HTTPError as e:
                    print(f"  ! {event_id} {_iso(t)}: {e}")
                    continue
                seen.add(key)
                if snap is None:
                    continue
                for row in flatten_snapshot(sport, snap):
                    buffers.setdefault(_month_key(row["snapshot_ts"]), []).append(row)
            _flush(sport_dir, buffers)
            print(f"  {sport} {event_id} done (requests used={budget.used}, "
                  f"remaining~={budget.last_remaining})")
        _flush(sport_dir, buffers, force=True)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_seen(sport_dir: Path) -> set[tuple[str, str]]:
    seen: set[tuple[str, str]] = set()
    for f in sport_dir.glob("*.parquet"):
        df = pd.read_parquet(f, columns=["match_id", "snapshot_ts"])
        seen.update(map(tuple, df.drop_duplicates().to_numpy()))
    return seen


def _flush(sport_dir: Path, buffers: dict[str, list[dict]], force: bool = False) -> None:
    for month, rows in list(buffers.items()):
        if not rows or (len(rows) < 5000 and not force):
            continue
        path = sport_dir / f"{month}.parquet"
        new = pd.DataFrame(rows)
        if path.exists():
            new = pd.concat([pd.read_parquet(path), new], ignore_index=True)
        new.drop_duplicates(
            subset=["match_id", "snapshot_ts", "bookmaker", "outcome_name"]
        ).to_parquet(path, index=False)
        buffers[month] = []


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/ingest.yaml")
    run(ap.parse_args().config)
