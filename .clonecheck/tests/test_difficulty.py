from cognitive_load.difficulty import (
    build_difficulty_features,
    compute_task_difficulty,
    difficulty_multiplier,
    expected_recall_rate,
)

# expected_recall_rate ---------------------------------------------------------


def test_expected_recall_decreases_with_span():
    easy = expected_recall_rate(3)
    hard = expected_recall_rate(7)
    assert easy > hard
    assert 0.0 < hard < easy <= 1.0


def test_expected_recall_clamps_out_of_range():
    assert expected_recall_rate(2) == expected_recall_rate(3)
    assert expected_recall_rate(9) == expected_recall_rate(7)


# compute_task_difficulty ------------------------------------------------------


def test_canonical_config_has_low_difficulty(session_factory):
    session = session_factory(profile="low")
    difficulty = compute_task_difficulty(session)
    assert difficulty["score"] < 20.0


def test_harder_config_scores_higher(session_factory):
    easy_session = session_factory(profile="low")
    hard_session = session_factory(profile="low")
    # Make the config genuinely harder: long spans only, no congruent Stroop
    # trials, maximally imbalanced switch/repeat mix.
    for trial in hard_session["tasks"]["working_memory"]["trials"]:
        trial["sequence_length"] = 7
    for trial in hard_session["tasks"]["stroop"]["trials"]:
        trial["congruent"] = False
    for trial in hard_session["tasks"]["task_switching"]["trials"]:
        trial["is_switch"] = True

    easy = compute_task_difficulty(easy_session)
    hard = compute_task_difficulty(hard_session)
    assert hard["score"] > easy["score"]


# multiplier & adjusted error rate ---------------------------------------------


def test_multiplier_harder_than_one_hard_sessions_below_one():
    assert difficulty_multiplier(80.0) > 1.0
    assert difficulty_multiplier(20.0) < 1.0
    assert difficulty_multiplier(50.0) == 1.0


def test_adjusted_error_rate_compensates_for_hardness(session_factory):
    session = session_factory(profile="elevated")
    from cognitive_load.analysis import build_session_summary

    summary = build_session_summary(session)
    features = summary["features"]
    built = build_difficulty_features(session, features)

    raw = features["overall_error_rate"]
    assert built["task_difficulty_score"] >= 0.0
    assert 0.0 <= built["difficulty_adjusted_error_rate"] <= 1.0
    # A hard session (score > 50) should see its error rate divided down.
    if built["task_difficulty_score"] > 50:
        assert built["difficulty_adjusted_error_rate"] < raw


def test_missing_tasks_yield_zero_difficulty():
    difficulty = compute_task_difficulty({"tasks": {}})
    assert difficulty["score"] == 0.0
    assert difficulty["memory"] == 0.0
