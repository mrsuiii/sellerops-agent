"""C3b in the real approval graph: first draft from a request, asking the seller, photos."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from sellerops.workflows.approval.approval_graph import build_approval_graph

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))
from fake_models import FakeModel, call, fake  # noqa: E402

HEADERS = {"X-Seller-User": "seller-1"}
DRAFT = lambda text, **extra: call("CaptionDraft", {"description": text, "reason": "draf", **extra})  # noqa: E731
ASK = lambda *keys: call("AskSeller", {"questions": [{"key": k, "question": f"{k}?"} for k in keys]})  # noqa: E731


class VisionFake(FakeModel):
    supports_vision: bool = True


def run_request(service, model, *, images=None, **request):
    created = service.request_caption("FS180", actor="seller:s1", **({"instruction": "sebut bahannya"} | request))
    graph = build_approval_graph(service, model, checkpointer=InMemorySaver(), images=images)
    config = {"configurable": {"thread_id": created["request_id"]}}
    return graph, config, created["request_id"], graph.invoke({"request_id": created["request_id"]}, config)


def answer(graph, config, **answers):
    return graph.invoke(Command(resume={"answers": answers, "user": "seller-1"}), config)


def human_text(message) -> str:
    content = message.content
    return content if isinstance(content, str) else content[0]["text"]


def test_request_drafts_the_first_version(service) -> None:
    model = fake(DRAFT("A pea-print dress."))
    graph, config, rid, out = run_request(service, model)
    request = out["__interrupt__"][0].value
    assert request["kind"] == "decision"
    proposal = request["proposal"]
    assert proposal["request_id"] == rid and proposal["revision_of"] is None
    assert proposal["changes"]["description"]["new"] == "A pea-print dress."
    assert service.get_request(rid)["status"] == "proposed"
    prompt = human_text(model.calls[0][1])
    assert "Seller's request: sebut bahannya" in prompt and "first version" in prompt
    assert "Rejected versions" not in prompt and "You may call AskSeller" in prompt


def test_ask_then_draft_with_the_answers(service) -> None:
    model = fake(ASK("material", "fit"), DRAFT("A cotton dress."))
    graph, config, rid, out = run_request(service, model)
    request = out["__interrupt__"][0].value
    assert request["kind"] == "question" and request["request_id"] == rid
    assert [q["key"] for q in request["questions"]] == ["material", "fit"]

    bad = answer(graph, config, material="katun")                       # fit missing: asks again
    assert bad["__interrupt__"][0].value["error"]
    out = answer(graph, config, material="katun", fit=None)              # the graph works without the runner too
    proposal = out["__interrupt__"][0].value["proposal"]
    assert proposal["inputs"]["seller_facts"] == {"material": "katun", "fit": None}
    second = human_text(model.calls[1][1])
    assert '"material": "katun"' in second and '"fit": null' in second
    assert "Asking is not available now" in second                      # one round of questions per version


def test_known_facts_are_not_asked_again(service) -> None:
    service.remember_facts("FS180", {"material": "katun", "care": ""}, actor="seller:s1")
    model = fake(ASK("material", "care"), DRAFT("A cotton dress."))
    graph, config, _, out = run_request(service, model)
    assert out["__interrupt__"][0].value["kind"] == "decision"           # no question: both already known
    retry_prompt = human_text(model.calls[1][1])
    assert "already in the seller facts" in retry_prompt and '"care": null' in retry_prompt


def test_rejection_may_ask_again(service) -> None:
    model = fake(DRAFT("Short."), ASK("material"), DRAFT("Short, cotton."))
    graph, config, _, out = run_request(service, model)
    out = graph.invoke(Command(resume={"decision": "reject", "feedback": "sebut bahannya", "user": "s1"}), config)
    request = out["__interrupt__"][0].value
    assert request["kind"] == "question" and request["proposal_id"] == out["proposal_id"]
    out = answer(graph, config, material="katun")
    assert out["__interrupt__"][0].value["proposal"]["revision_of"] is not None


def test_cancelled_request_stops(service) -> None:
    model = fake(ASK("material"), DRAFT("x"))
    graph, config, rid, _ = run_request(service, model)
    service.cancel_request(rid, actor="seller:s1")
    out = answer(graph, config, material="katun")
    assert out["stop_reason"] == "cancelled"


# ------------------------------------------------------------------ photos


@pytest.fixture
def photos(tmp_path):
    pytest.importorskip("PIL")
    from PIL import Image

    from sellerops.seller.images import LocalImageStore, ProductImages

    def fetch(url: str) -> bytes:
        fetch.urls.append(url)
        out = io.BytesIO()
        Image.new("RGB", (1000, 1000), (20, 40, 160)).save(out, "PNG")
        return out.getvalue()

    fetch.urls = []
    images = ProductImages(LocalImageStore(tmp_path / "img", store_id="kala-studio"), fetch=fetch)
    return images, fetch


def test_vision_model_sees_one_photo_and_the_proposal_records_it(service, photos) -> None:
    images, fetch = photos
    model = VisionFake(script=[DRAFT("A navy dress.")], calls=[], kwargs=[])
    _, _, _, out = run_request(service, model, images=images)
    content = model.calls[0][1].content
    assert [block["type"] for block in content] == ["text", "image"] and "1 attached" in content[0]["text"]
    assert len(fetch.urls) == 1 and not fetch.urls[0].endswith("thumbnail.webp")   # a full photo, not the thumbnail
    refs = out["__interrupt__"][0].value["proposal"]["inputs"]["images"]
    assert refs[0]["source_url"] == fetch.urls[0] and len(refs[0]["sha256"]) == 64
    assert "base64" not in str(out)                                       # no image bytes in the state


def test_text_only_model_gets_no_photo(service, photos) -> None:
    images, fetch = photos
    model = fake(DRAFT("A dress."))
    _, _, _, out = run_request(service, model, images=images)
    assert isinstance(model.calls[0][1].content, str) and fetch.urls == []
    assert out["__interrupt__"][0].value["proposal"]["inputs"] is None


def test_broken_photos_fall_back_to_text(service, tmp_path) -> None:
    pytest.importorskip("PIL")
    from sellerops.seller.images import LocalImageStore, ProductImages

    images = ProductImages(LocalImageStore(tmp_path, store_id="kala-studio"), fetch=lambda url: b"broken")
    model = VisionFake(script=[DRAFT("A dress.")], calls=[], kwargs=[])
    _, _, _, out = run_request(service, model, images=images)
    assert isinstance(model.calls[0][1].content, str)
    assert out["__interrupt__"][0].value["kind"] == "decision"


# ------------------------------------------------------------------ over HTTP


def test_request_question_answer_decision_over_http(service) -> None:
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from sellerops.seller import api as seller_api
    from sellerops.store.api import app
    from sellerops.workflows.approval import api as approval_api
    from sellerops.workflows.approval.runner import ApprovalRunner

    runner = ApprovalRunner(service, fake(ASK("material"), DRAFT("A cotton dress.")), InMemorySaver())
    app.dependency_overrides[seller_api.get_service] = lambda: service
    app.dependency_overrides[approval_api.get_runner] = lambda: runner
    try:
        client = TestClient(app)
        view = client.post("/api/seller/caption-requests", headers=HEADERS,
                           json={"product_id": "FS180", "instruction": "sebut bahannya"}).json()
        assert (view["state"], view["waiting_for"]) == ("waiting", "answer")
        view = client.post(f"/api/seller/caption-requests/{view['request_id']}/approval/answer", headers=HEADERS,
                           json={"answers": {"material": "katun"}}).json()
        assert view["waiting_for"] == "decision"
        pid = view["request"]["proposal"]["proposal_id"]
        assert service.product_facts("FS180")["material"]["source"] == "CR-0001"
        done = client.post(f"/api/seller/caption-proposals/{pid}/approval/decision", headers=HEADERS,
                           json={"decision": "approve"}).json()
        assert done["stop_reason"] == "applied"
        assert service.analytics().product("FS180")["description"] == "A cotton dress."
    finally:
        app.dependency_overrides.clear()
