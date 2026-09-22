import pandas as pd

from src.panel.resample_ticks import resample


def _row(ts, price, outcome="Home", book="bet365"):
    return {
        "sport": "oddspapi_premier-league", "snapshot_ts": ts, "match_id": "m1",
        "commence_time": "2026-09-19T14:00:00.000Z", "bookmaker": book,
        "home_team": "A", "away_team": "B", "market": "h2h", "result": "H",
        "outcome_name": outcome, "price": price,
    }


def test_two_books_never_sharing_a_timestamp_end_up_aligned():
    # bet365 and pinnacle update at different instants, like the real API
    raw = pd.DataFrame([
        _row("2026-09-19T10:00:00Z", 2.0, book="bet365"),
        _row("2026-09-19T10:07:00Z", 2.1, book="bet365"),
        _row("2026-09-19T10:03:00Z", 2.2, book="pinnacle"),
        _row("2026-09-19T10:12:00Z", 2.3, book="pinnacle"),
    ])
    out = resample(raw, grid_minutes=5)
    grid_ts = sorted(out["snapshot_ts"].unique())
    # every grid point should have BOTH books present after forward-filling
    for ts in grid_ts:
        books = out[out["snapshot_ts"] == ts]["bookmaker"].tolist()
        assert set(books) <= {"bet365", "pinnacle"}
    counts = out.groupby("snapshot_ts")["bookmaker"].nunique()
    assert (counts == 2).sum() >= 1  # at least one grid point sees both books


def test_forward_fills_the_last_known_price_not_the_next_one():
    raw = pd.DataFrame([
        _row("2026-09-19T10:00:00Z", 2.0),
        _row("2026-09-19T10:20:00Z", 3.0),
    ])
    out = resample(raw, grid_minutes=5).sort_values("snapshot_ts")
    # a grid point between the two real observations must carry the EARLIER price
    mid = out[out["snapshot_ts"] == pd.Timestamp("2026-09-19T10:10:00Z")]
    assert not mid.empty
    assert mid["price"].iloc[0] == 2.0


def test_a_single_observation_cannot_be_resampled_onto_a_grid():
    # grid_start is ceil'd to the next boundary, so with only one point there
    # is no grid instant that both follows the ceil and has data behind it
    raw = pd.DataFrame([_row("2026-09-19T10:07:00Z", 2.0)])
    out = resample(raw, grid_minutes=5)
    assert out.empty


def test_no_grid_points_before_the_first_real_observation():
    raw = pd.DataFrame([
        _row("2026-09-19T10:07:00Z", 2.0),
        _row("2026-09-19T10:22:00Z", 2.5),
    ])
    out = resample(raw, grid_minutes=5)
    assert not out.empty
    assert (out["price"].notna()).all()
    # ceil(10:07, 5min) = 10:10 is the earliest a grid point can appear
    assert out["snapshot_ts"].min() >= pd.Timestamp("2026-09-19T10:10:00Z")


def test_grid_stops_at_kickoff():
    raw = pd.DataFrame([
        _row("2026-09-19T13:00:00Z", 2.0),
        _row("2026-09-19T15:00:00Z", 2.5),   # after the 14:00 kickoff
    ])
    out = resample(raw, grid_minutes=5)
    commence = pd.Timestamp("2026-09-19T14:00:00Z")
    assert (out["snapshot_ts"] <= commence).all()
