from __future__ import annotations

from p2c.agents.phase3.score_and_diagnose import ScoreAndDiagnoseAgent, _failure_entries


def test_failure_entries_accepts_list_and_legacy_dict() -> None:
    list_shape = [{"step_failures": [{"step_id": "s1"}]}]
    dict_shape = {"failures": [{"step_failures": [{"step_id": "s2"}]}]}

    assert _failure_entries([]) == []
    assert _failure_entries(list_shape) == list_shape
    assert _failure_entries(dict_shape) == dict_shape["failures"]


def test_classify_gaps_accepts_empty_failure_list() -> None:
    agent = ScoreAndDiagnoseAgent.__new__(ScoreAndDiagnoseAgent)

    gaps = agent._classify_gaps(
        verdict={"claim_verdicts": []},
        manifest={"runs": []},
        env_result={"validation_passed": True},
        failures=[],
    )

    assert gaps == []
