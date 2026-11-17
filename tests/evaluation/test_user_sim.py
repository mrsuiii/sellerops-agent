"""User simulators: scripted (deterministic) and LLM-backed (with a fake model)."""

from __future__ import annotations

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from sellerops.evaluation.user_sim import STOP_TOKEN, LLMUser, ScriptedUser

TASK = {
    "task_id": "t1", "persona": "Pemilik toko.", "goal": "Tahu tas yang tidak laku.",
    "hidden_info": [
        {"fact": "Kategorinya tas (bags).", "keywords": ["kategori", "jenis"]},
        {"fact": "Periode yang dimaksud: 30 hari terakhir.", "keywords": ["periode", "hari", "bulan"]},
    ],
}


def test_scripted_user_reveals_matching_facts_first() -> None:
    user = ScriptedUser()
    assert user.reply(TASK, "Periode berapa hari?", []).text == "Periode yang dimaksud: 30 hari terakhir."
    assert user.reply(TASK, "Ada lagi?", []).text == "Kategorinya tas (bags)."
    assert user.reply(TASK, "Kategori apa?", []).text == user.no_more_info


def test_scripted_user_can_reveal_several_facts() -> None:
    reply = ScriptedUser().reply(TASK, "Kategori dan periode berapa hari?", [])
    assert "tas" in reply.text and "30 hari" in reply.text and not reply.stop


def test_llm_user_flips_roles_and_hides_the_answer_key() -> None:
    model = FakeListChatModel(responses=["Tas aja.", f"Sudah cukup. {STOP_TOKEN}"])
    user = LLMUser(model)
    transcript = [{"role": "user", "text": "Produk mana yang nggak laku?"}]
    messages = user.messages(TASK, "Kategori apa?", transcript)
    assert isinstance(messages[0], SystemMessage) and "Kategorinya tas" in messages[0].content
    assert isinstance(messages[1], AIMessage) and messages[1].content == "Produk mana yang nggak laku?"
    assert isinstance(messages[2], HumanMessage) and messages[2].content == "Kategori apa?"
    assert "expected" not in messages[0].content

    assert user.reply(TASK, "Kategori apa?", transcript).text == "Tas aja."
    last = user.reply(TASK, "Ada lagi?", transcript)
    assert last.stop and last.text == "Sudah cukup."
