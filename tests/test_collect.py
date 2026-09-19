from datetime import datetime, timezone

from src.ingest.collect import has_event_in_window, within_pace

Q, RESERVE = 500, 15


def _t(day, hour=0):
    return datetime(2026, 9, day, hour, tzinfo=timezone.utc)


def test_pace_allows_when_ahead_of_burn_down():
    # 19 Sep noon: 61.7% of a 30-day month gone, linear target ~191 left
    assert within_pace(408, Q, RESERVE, _t(19, 12))


def test_pace_blocks_when_overspent():
    assert not within_pace(150, Q, RESERVE, _t(19, 12))


def test_pace_never_dips_below_reserve():
    assert not within_pace(10, Q, RESERVE, _t(29, 23))


def test_pace_unknown_quota_fails_open():
    assert within_pace(None, Q, RESERVE, _t(19, 12))


def test_month_start_allows_first_call():
    assert within_pace(500, Q, RESERVE, _t(1, 0))


def test_event_window():
    now = _t(19, 12)
    inside = [{"commence_time": "2026-09-19T15:00:00Z"}]
    outside = [{"commence_time": "2026-09-25T15:00:00Z"}, {"commence_time": "2026-09-19T01:00:00Z"}]
    assert has_event_in_window(inside, now, -3, 72)
    assert not has_event_in_window(outside, now, -3, 72)
    assert not has_event_in_window([], now, -3, 72)
