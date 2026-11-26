# Tutorial — membangun graph approval caption (C3)

Untuk Nigka. Desainnya ada di `docs/c3-approval-graph-design.md`; tutorial
ini urutan mengerjakannya. Setiap langkah berisi **konsep**, **kerangka**
(dengan `TODO` untuk kamu isi), **test**, dan **cek**. Kerjakan berurutan:
jalankan test tiap langkah sampai hijau sebelum lanjut.

Kode test boleh disalin apa adanya, karena test adalah spesifikasi. Kode
node ditulis sendiri; kerangka hanya menunjukkan bentuknya.

File yang dibuat:

```
src/sellerops/workflows/__init__.py        (kosong)
src/sellerops/workflows/caption_approval.py
tests/workflows/test_caption_approval.py
```

Jalankan test satu file: `uv run pytest -q tests/workflows -x`.

---

## Langkah 0 — Persiapan

- Baca docstring `src/sellerops/seller/service.py` (state machine proposal)
  dan `docs/c3-approval-graph-design.md` bagian 2–4.
- Fixture `service` (database SQLite baru per test) sudah ada di
  `tests/conftest.py`, jadi bisa langsung dipakai di `tests/workflows/`.
- Fake model ada di `tests/harness/fake_models.py`. Pytest tidak otomatis
  mencarinya dari folder lain, jadi di awal file test tambahkan:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))
from fake_models import call, fake  # noqa: E402
```

`fake(call("CaptionDraft", {...}))` membuat model yang menjawab satu tool
call. `model.with_structured_output(CaptionDraft)` mengubah tool call itu
menjadi objek `CaptionDraft` (sudah dicek).

---

## Langkah 1 — Kerangka: state, builder, compile

**Konsep.**

- `StateGraph(ApprovalState)`: setiap node menerima state dan mengembalikan
  dict berisi field yang **diubah** saja; LangGraph menggabungkannya ke state.
- Tanpa reducer, field list ditimpa. Untuk `feedback_history` ada dua
  pilihan:
  - kembalikan list lengkap yang baru, atau
  - pakai `Annotated[list[str], operator.add]` supaya LangGraph
    menambahkan, seperti `add_messages`.

  Pilih salah satu dan pahami bedanya.
- `interrupt()` **wajib** checkpointer. Tanpa checkpointer, tidak ada tempat
  menyimpan state saat graph berhenti.
- Node didefinisikan di dalam `build_approval_graph(...)` sebagai closure,
  supaya bisa memakai `service` dan `model` (pola yang sama dengan
  `build_agent_graph`).

**Kerangka.**

```python
from typing import Literal, TypedDict
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

StopReason = Literal["applied", "stale", "cancelled", "max_rejections", "revision_failed"]

class ApprovalState(TypedDict, total=False):
    proposal_id: str
    product_id: str
    fields: list[str]
    # TODO: field lain dari desain bagian 3

def build_approval_graph(service, model, *, checkpointer):
    # TODO: definisikan node di sini (langkah 2 dst.)
    builder = StateGraph(ApprovalState)
    # TODO: add_node / add_edge
    return builder.compile(checkpointer=checkpointer)

def thread_config(root_proposal_id: str) -> dict:
    return {"configurable": {"thread_id": f"caption-approval-{root_proposal_id}"}}
```

Satu tambahan kecil dari desain: input graph **cukup `proposal_id`**. Node
pertama, `load`, membaca proposal dari service lalu mengisi `product_id`,
`fields` (= kunci `changes` proposal), dan `rejections`. Node ini hanya
jalan sekali (bukan node interrupt), jadi aman.

**Cek.** `build_approval_graph(service, fake(), checkpointer=InMemorySaver())`
tidak error.

---

## Langkah 2 — `load` + `await_decision` (interrupt)

**Konsep.** Lihat penjelasan `interrupt()` sebelumnya. Ringkasnya:

- invoke pertama berhenti di `interrupt(value)`; hasil `invoke` punya
  `"__interrupt__"`;
- `invoke(Command(resume=x), config)` menjalankan **ulang** node dari awal,
  dan kali ini `interrupt()` mengembalikan `x`.

Jadi semua kode sebelum `interrupt()` harus aman dijalankan dua kali (hanya
membaca).

**Kerangka.**

```python
def load(state):
    proposal = service.get_proposal(state["proposal_id"])
    return {...}  # TODO: product_id, fields, rejections

def await_decision(state):
    proposal = service.get_proposal(state["proposal_id"])   # baca saja
    answer = interrupt({"proposal": proposal})              # TODO: apa lagi yang UI perlu?
    # TODO (langkah 3): validasi answer, kembalikan decision/feedback/decided_by/fields
    return {}
```

Edge: `START → load → await_decision`.

**Test.**

```python
from langgraph.checkpoint.memory import InMemorySaver
from sellerops.workflows.caption_approval import build_approval_graph, thread_config

def start(service, model=None, **fields):
    proposal = service.propose_caption("FS180", "clearer", actor="agent",
                                       **(fields or {"description": "Clearer."}))
    graph = build_approval_graph(service, model or fake(), checkpointer=InMemorySaver())
    config = thread_config(proposal["proposal_id"])
    out = graph.invoke({"proposal_id": proposal["proposal_id"]}, config)
    return graph, config, proposal, out

def test_graph_waits_for_the_seller(service):
    before = service.analytics().product("FS180")
    graph, config, proposal, out = start(service)
    assert out["__interrupt__"][0].value["proposal"]["proposal_id"] == proposal["proposal_id"]
    assert graph.get_state(config).next == ("await_decision",)
    assert service.analytics().product("FS180") == before   # belum berubah
```

---

## Langkah 3 — Payload resume dan routing

**Konsep.**

- Payload resume datang dari luar (API/UI), jadi divalidasi seperti input
  lain. Pydantic dengan `extra="forbid"`.
- Conditional edge hanya **membaca** state dan mengembalikan nama node.
  Keputusan dibuat node, bukan fungsi routing (pola yang sama dengan
  `graph.py`).

**Kerangka.**

```python
class Decision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    decision: Literal["approve", "reject", "cancel"]
    feedback: str = ""
    add_fields: list[Literal["name", "description", "tags"]] = []
    user: str

def route_decision(state) -> str:
    return ...  # TODO: "apply" | "record_rejection" | "cancel"

builder.add_conditional_edges("await_decision", route_decision, ["apply", "record_rejection", "cancel"])
```

Di `await_decision`: `Decision.model_validate(answer)`. Lalu putuskan apa
yang terjadi bila tidak valid: lempar error, atau `interrupt()` lagi untuk
bertanya ulang? Tulis alasanmu di docstring. `add_fields` digabung ke
`fields` (tanpa duplikat), dan `decided_by = f"seller:{user}"`.

**Cek.** Sementara, buat `apply` / `record_rejection` / `cancel` sebagai
node kosong yang langsung ke `END`, supaya graph bisa di-compile.

---

## Langkah 4 — `apply` (+ stale)

**Konsep.**

- Node berefek samping, jadi **tanpa** `RetryPolicy`.
- `SellerError` punya `.code`. Tangkap kode yang kamu harapkan; biarkan
  error lain naik.
- Idempoten: kalau node dijalankan ulang setelah approve sempat sukses
  (crash sebelum checkpoint), service melempar `invalid_transition`. Baca
  status proposal; kalau sudah `applied`, anggap sukses.
- Stale datang lewat **dua jalan**:
  - `stale_proposal`: caption berubah tanpa lewat proposal lain;
  - `invalid_transition` dengan status `stale`: jalan yang paling sering.
    Begitu proposal lain di-approve, backend langsung menandai proposal ini
    `stale`, jadi saat seller menekan approve statusnya sudah bukan
    `pending`.

  Jadi setelah `invalid_transition`, baca statusnya: `applied` → sukses,
  `stale` → stop `"stale"`, selain itu biarkan error naik.

**Kerangka.**

```python
def apply(state):
    try:
        service.approve(state["proposal_id"], actor=state["decided_by"])
    except SellerError as error:
        ...  # TODO: stale_proposal → "stale"; invalid_transition → baca status (applied / stale)
    return {"stop_reason": "applied"}
```

Edge: `apply → END`.

**Test.**

```python
def resume(graph, config, **answer):
    return graph.invoke(Command(resume={"user": "seller-1", **answer}), config)

def test_approve_applies_the_caption(service):
    graph, config, proposal, _ = start(service)
    out = resume(graph, config, decision="approve")
    assert out["stop_reason"] == "applied"
    assert service.analytics().product("FS180")["description"] == "Clearer."
    assert service.get_proposal(proposal["proposal_id"])["decided_by"] == "seller:seller-1"

def test_stale_proposal_is_not_applied(service):
    graph, config, proposal, _ = start(service)
    other = service.propose_caption("FS180", "x", actor="agent", tags=["dress"])
    service.approve(other["proposal_id"], actor="seller:seller-2")   # caption berubah duluan
    out = resume(graph, config, decision="approve")
    assert out["stop_reason"] == "stale"
    assert service.analytics().product("FS180")["description"] != "Clearer."

def test_apply_is_idempotent_after_a_crash(service):
    graph, config, proposal, _ = start(service)
    service.approve(proposal["proposal_id"], actor="seller:seller-1")  # "node sempat jalan, lalu crash"
    assert resume(graph, config, decision="approve")["stop_reason"] == "applied"
```

---

## Langkah 5 — `cancel`

Paling mudah: `service.cancel(...)` → `stop_reason="cancelled"` → `END`.
Tambahkan juga penanganan idempoten yang sama seperti `apply`.

```python
def test_cancel_stops_without_changes(service):
    before = service.analytics().product("FS180")
    graph, config, proposal, _ = start(service)
    assert resume(graph, config, decision="cancel")["stop_reason"] == "cancelled"
    assert service.get_proposal(proposal["proposal_id"])["status"] == "cancelled"
    assert service.analytics().product("FS180") == before
```

---

## Langkah 6 — `record_rejection` + batas 5

**Konsep.** Jumlah reject diambil dari `rejections_in_chain` hasil
`service.reject`, bukan dihitung sendiri. Routing sesudahnya:
`>= MAX_REJECTIONS` → `END` dengan `max_rejections`, selain itu ke
`draft_revision`.

Pertanyaan untukmu: siapa yang menulis `stop_reason="max_rejections"`?
Node `record_rejection` (lalu routing membaca `stop_reason`), atau fungsi
routing? Ingat aturan di `graph.py`: routing hanya membaca.

```python
from sellerops.seller import MAX_REJECTIONS
```

Test untuk langkah ini menyusul di langkah 9, karena butuh revisi.

---

## Langkah 7 — `draft_revision` (LLM)

**Konsep.**

- `model.with_structured_output(CaptionDraft)` meminta model mengembalikan
  objek sesuai skema, bukan teks bebas. Di balik layar, LangChain mengirim
  skema sebagai tool dan mem-parse tool call-nya (itulah sebabnya fake model
  dengan `call("CaptionDraft", ...)` bisa dipakai).
- Node ini **tanpa efek samping**, jadi aman diberi
  `retry_policy=DEFAULT_MODEL_RETRY` (dari `sellerops.harness.graph`) untuk
  error API sementara.
- Prompt disusun ulang dari state setiap kali (tanpa `messages`).

**Skema.**

```python
class CaptionDraft(BaseModel):
    name: str | None = None
    description: str | None = None
    tags: list[str] | None = None
    reason: str = Field(description="Apa yang diubah dan kenapa, untuk seller.")
```

**Yang harus ada di prompt** (tulis sendiri):

1. Caption **sekarang** (`service.analytics().product(product_id)`): name,
   description, tags, dan fakta produk yang boleh disebut.
2. Field yang boleh diubah (`fields`). Field lain harus `null`.
3. Semua versi yang ditolak beserta alasannya (`feedback_history`). Ambil
   versinya dari `service.get_proposal` mengikuti `revision_of`.
4. Bila alasan kosong: minta varian yang jelas berbeda dari semua versi yang
   ditolak.
5. `draft_error` dari percobaan sebelumnya, kalau ada (langkah 8).
6. Aturan lama: jangan menambah klaim (bahan, ukuran, garansi) yang tidak
   ada di data produk; batas panjang (name 80, description 600, tags 8).

Hasil node: `{"draft": draft.model_dump(), "draft_attempts": ... + 1}`.

---

## Langkah 8 — `submit_revision`

**Konsep.** Node yang menulis ke DB. Dua lapis cek:

1. **Cek cakupan field (kode kita):** draft yang mengisi field di luar
   `fields` ditolak sebelum memanggil service.
2. **Validasi service:** `service.propose_caption(..., revision_of=state["proposal_id"])`
   bisa melempar `no_changes`, `field_too_long`, atau `invalid_tags`.

Kalau gagal: simpan pesan error di `draft_error`, lalu kembali ke
`draft_revision`. Setelah 2 percobaan, `stop_reason="revision_failed"`. Kalau
sukses: `proposal_id` baru, `draft_error=None`, `draft_attempts=0`, lalu
kembali ke `await_decision`.

Idempoten: kalau node dijalankan ulang setelah revisi sempat dibuat, service
melempar `invalid_revision` ("already revised"). Cari revisinya dengan
`service.list_proposals(product_id=...)` (yang `revision_of` = proposal ini)
dan pakai itu.

---

## Langkah 9 — Loop lengkap

Sambungkan: `draft_revision → submit_revision → (await_decision | draft_revision | END)`.

```python
DRAFT = lambda text: call("CaptionDraft", {"description": text, "reason": "lebih pendek"})  # noqa: E731

def test_reject_then_revision_then_approve(service):
    graph, config, first, _ = start(service, model=fake(DRAFT("Short.")))
    out = resume(graph, config, decision="reject", feedback="terlalu panjang")
    revision = out["__interrupt__"][0].value["proposal"]
    assert revision["revision_of"] == first["proposal_id"]
    assert revision["changes"]["description"]["new"] == "Short."
    assert resume(graph, config, decision="approve")["stop_reason"] == "applied"
    assert service.analytics().product("FS180")["description"] == "Short."

def test_five_rejections_stop_the_flow(service):
    drafts = [DRAFT(f"Try {i}.") for i in range(5)]
    graph, config, _, _ = start(service, model=fake(*drafts))
    for _ in range(4):
        assert "__interrupt__" in resume(graph, config, decision="reject")
    out = resume(graph, config, decision="reject")
    assert out["stop_reason"] == "max_rejections"
    assert len(service.list_proposals(product_id="FS180")) == 5      # tidak ada proposal ke-6

def test_revision_cannot_touch_fields_that_were_not_asked(service):
    bad = call("CaptionDraft", {"name": "New Name", "reason": "x"})
    graph, config, _, _ = start(service, model=fake(bad, bad))
    assert resume(graph, config, decision="reject")["stop_reason"] == "revision_failed"

def test_seller_can_add_a_field_when_rejecting(service):
    draft = call("CaptionDraft", {"name": "Dress Pea Polka", "description": "Short.", "reason": "x"})
    graph, config, _, _ = start(service, model=fake(draft))
    out = resume(graph, config, decision="reject", feedback="ubah juga namanya", add_fields=["name"])
    assert out["__interrupt__"][0].value["proposal"]["changes"]["name"]["new"] == "Dress Pea Polka"

def test_reject_without_feedback_still_revises(service):
    graph, config, _, _ = start(service, model=fake(DRAFT("Other.")))
    assert "__interrupt__" in resume(graph, config, decision="reject")
```

Tambahkan sendiri: prompt revisi kedua memuat alasan reject pertama. Cek
`model.calls[-1]` (FakeModel merekam semua prompt).

---

## Langkah 10 — Bertahan setelah restart

**Konsep.** Approval bisa menunggu berhari-hari, jadi state harus selamat
saat server restart. `InMemorySaver` hilang saat proses mati; `SqliteSaver`
tidak.

```python
from sellerops.harness.trajectory import open_sqlite_checkpointer

def test_resume_survives_a_restart(service, tmp_path):
    proposal = service.propose_caption("FS180", "clearer", actor="agent", description="Clearer.")
    config = thread_config(proposal["proposal_id"])
    first = build_approval_graph(service, fake(), checkpointer=open_sqlite_checkpointer(tmp_path / "a.sqlite"))
    first.invoke({"proposal_id": proposal["proposal_id"]}, config)
    del first                                                       # "server mati"
    again = build_approval_graph(service, fake(), checkpointer=open_sqlite_checkpointer(tmp_path / "a.sqlite"))
    assert again.invoke(Command(resume={"decision": "approve", "user": "s"}), config)["stop_reason"] == "applied"
```

---

## Langkah 11 — Coba dengan model asli (manual, bayar)

Script kecil di scratchpad atau `scripts/try_caption_approval.py`:

1. `service` di file sementara;
2. `model = get_chat_model()`;
3. buat proposal, `invoke`, lalu reject dengan feedback sungguhan;
4. cetak revisi dari `__interrupt__`.

Periksa: hanya field yang diizinkan yang berubah? feedback benar-benar
dipakai? tidak ada klaim baru? Trace-nya akan muncul di LangSmith.

---

## Selesai

Checklist sebelum minta review:

- [ ] semua test di `tests/workflows/` hijau, `make lint test` bersih;
- [ ] tidak ada efek samping sebelum `interrupt()`;
- [ ] node berefek samping idempoten (approve, cancel, submit_revision);
- [ ] hanya `draft_revision` yang memanggil LLM, dan hanya node itu yang
      punya `RetryPolicy`;
- [ ] docstring modul menjelaskan alur dan keputusanmu (payload tidak valid,
      siapa yang menulis `stop_reason`).

Setelah itu, assistant menyambungkannya ke API:

- `start_approval` saat agent membuat proposal;
- endpoint approve/reject/cancel → `Command(resume=...)`;
- `SqliteSaver` di `var/approvals.sqlite`;
- `GET …/approval` untuk status thread.
