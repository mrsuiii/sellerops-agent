"""Caption eval plumbing with fake models (free): trial runner, scripted seller, code checks, judge parsing."""

from __future__ import annotations

import sys
from pathlib import Path

from langchain_core.messages import AIMessage

from sellerops.evaluation.caption_eval import SCENARIOS, CaptionJudge, answer_for, run_trial
from sellerops.settings import get_settings

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))
from fake_models import call, fake  # noqa: E402

BY_ID = {s.id: s for s in SCENARIOS}


def trial(scenario_id, *script, tmp_path):
    settings = get_settings()
    return run_trial(BY_ID[scenario_id], 0, model=fake(*script), workdir=tmp_path,
                     catalog_path=settings.catalog_path, sim_dir=settings.data_dir / "seller_sim")


def test_scripted_seller_answers_by_keyword() -> None:
    answers = BY_ID["request-material-asks"].answers
    assert answer_for("fabric_material", answers) == "linen" and answer_for("fit", answers) is None


def test_asking_then_using_the_answer_passes(tmp_path) -> None:
    record = trial("request-material-asks",
                   call("AskSeller", {"questions": [{"key": "material", "question": "Bahannya?"}]}),
                   call("CaptionDraft", {"description": "An easy linen dress.", "reason": "bahan linen"}),
                   tmp_path=tmp_path)
    assert record.completed and record.answers_given == {"material": "linen"}
    assert record.code_pass, record.checks


def test_invented_fabric_and_missing_question_fail(tmp_path) -> None:
    record = trial("request-material-unknown",
                   call("CaptionDraft", {"description": "A soft cotton shirt.", "reason": "ok"}),
                   tmp_path=tmp_path)
    assert record.checks["ask_behaviour"] is False and record.checks["no_invented_fabric"] is False
    assert record.checks["missing_noted"] is False and not record.code_pass


def test_rejection_scenario_and_sentence_limit(tmp_path) -> None:
    record = trial("reject-shorter",
                   call("CaptionDraft", {"description": "A playful dress. Easy to wear. Fun print.", "reason": "x"}),
                   tmp_path=tmp_path)
    assert record.completed and record.checks["short_enough"] is False
    assert record.checks["differs_from_rejected"] is True


def test_judge_reply_is_parsed(tmp_path) -> None:
    record = trial("request-plain", call("CaptionDraft", {"description": "A check shirt.", "reason": "x"}),
                   tmp_path=tmp_path)
    reply = AIMessage('<think>hmm</think>{"grounded": true, "does_request": false, "honest_reason": true, '
                      '"reason": "too short"}')
    verdict = CaptionJudge(fake(reply), "fake").assess(BY_ID["request-plain"], record, {}, {})
    assert verdict["pass"] is False and verdict["does_request"] is False
