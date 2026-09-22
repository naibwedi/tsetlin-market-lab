from src.ingest.oddspapi import _result_from_scores_body, parse_history

FIXTURE = {
    "fixtureId": "id123", "participant1Name": "Tottenham Hotspur",
    "participant2Name": "Aston Villa", "startTime": "2026-09-19T14:00:00.000Z",
    "tournamentSlug": "premier-league",
}

# shape confirmed live 2026-09-22 (scripts/oddspapi_inspect.py): a price point is
# {createdAt, price, limit, active, exchangeMeta}; market "101" = Full Time
# Result, outcomes 101/102/103 = home/draw/away (OddsPapi's GET /markets ref).
HIST = {
    "bookmakers": {
        "bet365": {
            "markets": {
                "101": {
                    "outcomes": {
                        "101": {"players": {"0": [
                            {"createdAt": "2026-09-15T10:00:00.000Z", "price": 2.3},
                            {"createdAt": "2026-09-15T10:20:00.000Z", "price": 2.25},
                        ]}},
                        "102": {"players": {"0": [
                            {"createdAt": "2026-09-15T10:00:00.000Z", "price": 3.4},
                        ]}},
                        "103": {"players": {"0": [
                            {"createdAt": "2026-09-15T10:00:00.000Z", "price": 3.1},
                        ]}},
                    },
                },
                # a market that isn't the 3-way winner market - must be ignored
                "999": {"outcomes": {"1": {"players": {"0": [
                    {"createdAt": "2026-09-15T10:00:00.000Z", "price": 1.9},
                ]}}}},
            },
        },
    },
}


def test_parses_all_three_outcomes_for_a_requested_book():
    df = parse_history(HIST, FIXTURE, books=["bet365"], result="H")
    assert set(df["outcome_name"]) == {"Home", "Draw", "Away"}
    assert len(df) == 4  # 2 home points + 1 draw + 1 away


def test_ignores_books_not_in_the_response():
    df = parse_history(HIST, FIXTURE, books=["pinnacle"], result="H")
    assert df.empty


def test_carries_fixture_metadata_and_result_onto_every_row():
    df = parse_history(HIST, FIXTURE, books=["bet365"], result="H")
    assert (df["home_team"] == "Tottenham Hotspur").all()
    assert (df["away_team"] == "Aston Villa").all()
    assert (df["result"] == "H").all()
    assert (df["match_id"] == "20260919_premier-league_id123").all()


def test_price_is_the_raw_decimal_odds():
    df = parse_history(HIST, FIXTURE, books=["bet365"], result="H")
    home_prices = sorted(df[df["outcome_name"] == "Home"]["price"])
    assert home_prices == [2.25, 2.3]


# real /scores shape, confirmed live 2026-09-22 (Fulham 1-1 Manchester United)
SCORES_BODY = {
    "fixtureId": "id1", "scores": {"periods": {
        "result": {"participant1Score": 1, "participant2Score": 1},
        "p1": {"participant1Score": 0, "participant2Score": 0},
        "fulltime": {"participant1Score": 1, "participant2Score": 1},
    }},
}


def test_scores_draw_from_fulltime_period():
    assert _result_from_scores_body(SCORES_BODY) == "D"


def test_scores_home_win():
    body = {"scores": {"periods": {"fulltime": {"participant1Score": 2, "participant2Score": 0}}}}
    assert _result_from_scores_body(body) == "H"


def test_scores_away_win():
    body = {"scores": {"periods": {"fulltime": {"participant1Score": 0, "participant2Score": 1}}}}
    assert _result_from_scores_body(body) == "A"


def test_scores_falls_back_to_result_period_without_fulltime():
    body = {"scores": {"periods": {"result": {"participant1Score": 3, "participant2Score": 1}}}}
    assert _result_from_scores_body(body) == "H"


def test_scores_missing_periods_returns_none():
    assert _result_from_scores_body({"scores": {"periods": {}}}) is None
    assert _result_from_scores_body({}) is None
