from datetime import datetime, timezone

from src.ingest.footballdata import opening_time


def _k(y, m, d, h):
    return datetime(y, m, d, h, tzinfo=timezone.utc)


def test_weekend_uses_previous_friday_afternoon():
    # Sat 20 Sep 2025 15:00 -> Fri 19 Sep 15:00
    assert opening_time(_k(2025, 9, 20, 15)) == _k(2025, 9, 19, 15)
    # Sun 21 Sep -> Fri 19 Sep; Mon 22 Sep -> Fri 19 Sep
    assert opening_time(_k(2025, 9, 21, 16)) == _k(2025, 9, 19, 15)
    assert opening_time(_k(2025, 9, 22, 20)) == _k(2025, 9, 19, 15)


def test_midweek_uses_tuesday_afternoon():
    # Wed 24 Sep 2025 19:45 -> Tue 23 Sep 15:00
    assert opening_time(_k(2025, 9, 24, 19)) == _k(2025, 9, 23, 15)


def test_open_is_always_at_least_3h_before_kickoff():
    # Tue 23 Sep 16:00 kickoff cannot have a 15:00-same-day capture with a 3h gap
    ko = _k(2025, 9, 23, 16)
    assert (ko - opening_time(ko)).total_seconds() >= 3 * 3600
