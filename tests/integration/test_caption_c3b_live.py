"""C3b with a real LLM (opt-in, costs money): asking the seller, and the product photo.

Two seller sessions, about 2-4 model calls in total. The photo is fetched from
the catalog CDN for real.
"""

from __future__ import annotations

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from sellerops.workflows.approval.approval_graph import build_approval_graph

pytestmark = pytest.mark.live


def run(service, model, images=None, **request):
    created = service.request_caption("FS180", actor="seller:tester", **request)
    graph = build_approval_graph(service, model, checkpointer=InMemorySaver(), images=images)
    config = {"configurable": {"thread_id": created["request_id"]}}
    return graph, config, graph.invoke({"request_id": created["request_id"]}, config)


def show(out) -> dict:
    value = out["__interrupt__"][0].value
    print(f"\n[live] {value['kind']}: {value.get('questions') or value['proposal']['changes']}")
    if value["kind"] == "decision":
        print(f"[live] reason: {value['proposal']['reason']}")
    return value


def test_agent_asks_for_the_material_then_uses_it(service, live_model) -> None:
    graph, config, out = run(service, live_model, fields=["description"],
                             instruction="Sebutkan bahan kainnya di deskripsi.")
    asked = show(out)
    assert asked["kind"] == "question", "the material is in no source: the agent should ask"
    keys = [q["key"] for q in asked["questions"]]
    assert any("material" in key or "fabric" in key or "bahan" in key for key in keys)

    out = graph.invoke(Command(resume={"answers": {k: ("linen" if k == keys[0] else None) for k in keys},
                                       "user": "tester"}), config)
    proposal = show(out)["proposal"]
    assert "linen" in proposal["changes"]["description"]["new"].lower()


def test_photo_grounds_the_colour(service, live_model, tmp_path) -> None:
    pytest.importorskip("PIL")
    from sellerops.harness.models import supports_vision
    from sellerops.seller.images import LocalImageStore, ProductImages

    if not supports_vision(live_model):
        pytest.skip("the configured model has no vision")
    images = ProductImages(LocalImageStore(tmp_path, store_id="kala-studio"))
    graph, config, out = run(service, live_model, images=images, fields=["description"],
                             instruction="Sebutkan warna dan motifnya. Jangan tanya apa pun, tulis saja.")
    value = show(out)
    if value["kind"] == "question":   # allowed, but answer "unknown" and go on
        out = graph.invoke(Command(resume={"answers": {q["key"]: None for q in value["questions"]},
                                           "user": "tester"}), config)
        value = show(out)
    proposal = value["proposal"]
    assert proposal["inputs"]["images"], "the proposal records the photo the model saw"
    text = proposal["changes"]["description"]["new"].lower()
    print(f"[live] photo: {proposal['inputs']['images'][0]['source_url']}")
    # the photo: a white strapless dress with small red-pink polka dots and a bow (not in the text data)
    assert any(word in text for word in ("white", "polka", "dot", "bow", "strapless")), text
