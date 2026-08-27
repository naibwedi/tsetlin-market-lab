"""Betfair Exchange adapter (FREE *delayed* application key).

The delayed key is issued automatically and free of charge, with a 1-180 s data
delay - fine for research. A live key costs a one-off GBP 499 and is NOT needed
here. Note: Betfair is not licensed in every country (e.g. Norway); account
creation may be unavailable.

Setup
-----
1. Open a Betfair account, then create app keys via the Accounts API demo tool
   (createDeveloperAppKeys). Use the Delayed key.
2. Set env vars / GitHub secrets:
       BETFAIR_USER, BETFAIR_PASS, BETFAIR_APP_KEY
3. `pip install betfairlightweight`
4. Enable the `betfair` adapter in config/collect.yaml.

This module returns rows in the shared flat schema (one per
snapshot x match x "bookmaker" x outcome). For the exchange we emit a synthetic
bookmaker key ``betfair_ex`` with the back price (best available to back) per
runner, so it slots straight into the existing panel builder. Richer fields
(lay price, traded volume, ladder depth) are captured in extra columns for the
microstructure features added in later phases.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def collect_snapshot(cfg: dict) -> list[dict]:
    user = os.environ.get("BETFAIR_USER")
    pw = os.environ.get("BETFAIR_PASS")
    app_key = os.environ.get("BETFAIR_APP_KEY")
    if not (user and pw and app_key):
        raise RuntimeError("BETFAIR_USER / BETFAIR_PASS / BETFAIR_APP_KEY not set")

    import betfairlightweight
    from betfairlightweight import filters

    client = betfairlightweight.APIClient(user, pw, app_key=app_key)
    client.login_interactive()
    try:
        mf = filters.market_filter(
            event_type_ids=[cfg.get("event_type_id", "1")],
            market_type_codes=cfg.get("market_types", ["MATCH_ODDS"]),
            competition_ids=cfg.get("competitions") or None,
        )
        catalogues = client.betting.list_market_catalogue(
            filter=mf,
            market_projection=["EVENT", "RUNNER_DESCRIPTION", "MARKET_START_TIME"],
            max_results=100,
        )
        rows: list[dict] = []
        ts = _now_iso()
        for cat in catalogues:
            book = client.betting.list_market_book(
                market_ids=[cat.market_id],
                price_projection=filters.price_projection(price_data=["EX_BEST_OFFERS", "EX_TRADED"]),
            )
            if not book:
                continue
            b = book[0]
            runners = {r.selection_id: r for r in b.runners}
            for rc in cat.runners:
                r = runners.get(rc.selection_id)
                if r is None:
                    continue
                back = r.ex.available_to_back[0].price if r.ex.available_to_back else None
                lay = r.ex.available_to_lay[0].price if r.ex.available_to_lay else None
                rows.append(
                    {
                        "sport": "betfair_soccer",
                        "snapshot_ts": ts,
                        "match_id": cat.event.id,
                        "commence_time": (
                            cat.market_start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
                            if cat.market_start_time else None
                        ),
                        "home_team": cat.event.name.split(" v ")[0] if cat.event.name else None,
                        "away_team": cat.event.name.split(" v ")[-1] if cat.event.name else None,
                        "bookmaker": "betfair_ex",
                        "book_last_update": ts,
                        "market": "h2h",
                        "outcome_name": rc.runner_name,
                        "price": back,
                        "bf_lay": lay,
                        "bf_ltp": r.last_price_traded,
                        "bf_traded_volume": r.total_matched,
                    }
                )
        return rows
    finally:
        client.logout()
