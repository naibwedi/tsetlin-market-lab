from scripts.oddspapi_probe import _fixtures, _walk_history

SAMPLE = {  # shape copied from OddsPapi's published example
    "fixtureId": "id1",
    "bookmakers": {"pinnacle": {"markets": {"101": {"outcomes": {"101": {"players": {"0": [
        {"createdAt": "2026-04-05T14:58:17Z", "price": 1.85, "limit": None},
        {"createdAt": "2026-04-05T15:02:17Z", "price": 1.87, "limit": None}]}}}}}}},
}


def test_walk_history_counts_and_sorts_points():
    ts = _walk_history(SAMPLE)["pinnacle"]
    assert len(ts) == 2 and ts[0] < ts[1]
    assert (ts[1] - ts[0]).total_seconds() == 240


def test_fixtures_accepts_list_or_wrapped_dict():
    fx = [{"fixtureId": "a"}]
    assert _fixtures(fx) == fx
    assert _fixtures({"data": fx}) == fx
    assert _fixtures({"error": "x"}) == []
