# Proposal desain — graph approval caption (C3)

Status: **disetujui dengan revisi** (20 Nov 2026; jawaban Nigka di bagian 9).
Implementasi graph tetap bagian Nigka (kontrak belajar); dokumen ini hanya
desain. Backend yang dipanggil sudah ada: `src/sellerops/seller/service.py`.

## 1. Posisi graph

```
chat agent (harness/graph.py)                graph approval (workflows/caption_approval.py)
  seller: "perbaiki deskripsi Dress Pea"
  → propose_caption_update ──► CP-0001 ──► start(CP-0001)
  → final: "usulan menunggu persetujuan"        └─► [interrupt: tunggu seller] …
  (chat selesai, seller bebas lanjut)
                                             UI/API: Approve / Reject + alasan
                                                └─► Command(resume=…) ke thread approval
```

- Satu **thread approval per rantai proposal**:
  `thread_id = "caption-approval-<proposal_id pertama>"`. Revisi tetap di
  thread yang sama
- Graph mulai dari proposal yang **sudah ada** (dibuat chat agent). Graph
  tidak membuat usulan pertama, hanya revisi.
- Usulan hanya reaktif (keputusan 18 Nov): satu-satunya pemicu adalah
  permintaan seller di chat.
- Backend tetap sumber kebenaran. Graph hanya mengatur urutan dan memanggil
  service. Revert tetap di luar graph.

## 2. Alur

```
START
  │
  ▼
await_decision ── interrupt({proposal}) ──►  (graph berhenti; bisa berhari-hari)
  │  resume {"decision", "feedback"?, "add_fields"?, "user"}
  ├── approve ──► apply ──┬── applied ───────────────────────► END  stop_reason="applied"
  │                       └── SellerError stale_proposal ─────► END  stop_reason="stale"
  ├── cancel ───► cancel (service.cancel) ────────────────────► END  stop_reason="cancelled"
  │
  └── reject ───► record_rejection
                     │  (rejections_in_chain dari service)
                     ├── >= 5 ────────────────────────────────► END  stop_reason="max_rejections"
                     ▼
                  draft_revision   (LLM: caption baru dari feedback; tanpa feedback → varian yang jelas beda)
                     ▼
                  submit_revision  (service.propose_caption(revision_of=…))
                     ├── ok ─────────────► await_decision  (proposal_id = revisi)
                     ├── error validasi ─► draft_revision  (maks 2 percobaan, error dikirim ke LLM)
                     └── tetap gagal ────────────────────────► END  stop_reason="revision_failed"
```

## 3. State

```python
class ApprovalState(TypedDict, total=False):
    product_id: str
    fields: list[str]          # field yang boleh diubah: awalnya field usulan pertama, + add_fields saat reject
    proposal_id: str           # proposal pending saat ini
    decision: Literal["approve", "reject", "cancel"] | None
    feedback: str              # alasan reject terakhir ("" = seller tidak memberi alasan)
    decided_by: str            # "seller:<user>" dari resume
    feedback_history: list[str]  # semua alasan reject di rantai ini, untuk prompt revisi
    rejections: int            # salinan rejections_in_chain dari service (service = kebenaran)
    draft: dict | None         # hasil draft_revision sebelum disubmit
    draft_error: str | None    # error validasi dari service, untuk percobaan berikutnya
    draft_attempts: int
    stop_reason: Literal["applied", "stale", "cancelled", "max_rejections", "revision_failed"] | None
```

Alasan pilihan:

- **`rejections` dibaca dari service**, bukan dihitung sendiri. Kalau dua
  sumber menghitung, keduanya bisa berbeda. State cuma menyimpan salinan
  untuk routing.
- **`fields`** menerapkan aturan RCA-1 (hanya ubah field yang diminta).
  Awalnya = field yang diubah usulan pertama. Saat reject, seller boleh
  menambah field lewat `add_fields` (mis. checkbox "ubah juga nama" di UI).
  Penambahan field harus **eksplisit di payload**, bukan ditebak LLM dari
  teks feedback: cakupan field itu izin, dan izin tidak boleh bergantung pada
  tafsiran model. Feedback tetap dikirim ke LLM sebagai arahan isi.
  `submit_revision` menolak draft yang mengubah field di luar `fields`.
- **`feedback_history`**, bukan hanya feedback terakhir: revisi ke-3 tidak
  boleh mengulang kesalahan yang ditolak di revisi ke-1.
- Tidak ada `messages`: graph ini bukan percakapan. Prompt revisi disusun
  ulang dari state setiap kali, sehingga pendek dan terkontrol.

## 4. Node

| Node | LLM? | Efek samping | Isi |
|---|---|---|---|
| `await_decision` | tidak | **tidak ada** | `interrupt({"proposal": service.get_proposal(id)})` → simpan `decision`, `feedback`, `decided_by`; `add_fields` digabung ke `fields` |
| `apply` | tidak | `service.approve` | sukses → `applied`; `stale_proposal` → `stale` |
| `cancel` | tidak | `service.cancel` | proposal → `cancelled`; `stop_reason="cancelled"` |
| `record_rejection` | tidak | `service.reject` | simpan `rejections_in_chain`, tambah `feedback_history` (alasan kosong dicatat sebagai "tanpa alasan") |
| `draft_revision` | **ya** | tidak ada | `with_structured_output(CaptionDraft)`; input = caption sekarang (`analytics().product`), `fields`, `feedback_history`, `draft_error` |
| `submit_revision` | tidak | `service.propose_caption(revision_of=…)` | sukses → `proposal_id` baru; `SellerError` → `draft_error` |

Hanya `draft_revision` yang memanggil LLM, dan perannya hanya menulis teks.
Semua keputusan (terapkan, tolak, berhenti) diambil kode dan seller.

**Kenapa LLM dan submit dipisah:** `draft_revision` tidak punya efek
samping, jadi aman diberi `RetryPolicy` untuk error API. `submit_revision`
menulis ke DB, jadi tidak boleh di-retry membabi buta.

## 5. Hal penting soal `interrupt()`

1. **Node yang berisi `interrupt()` dijalankan ulang dari awal saat resume.**
   Karena itu `await_decision` tidak boleh punya efek samping;
   `get_proposal` cuma membaca, jadi aman.
2. **Crash di tengah node berefek samping.** Kalau proses mati setelah
   `service.approve` tapi sebelum checkpoint tersimpan, node itu dijalankan
   ulang. Backend sudah menjaganya:
   - approve kedua → `invalid_transition` → baca status proposal; kalau
     `applied`, anggap sukses;
   - revisi kedua untuk proposal yang sama ditolak unique index
     (`invalid_revision: already revised`) → cari revisinya dan pakai itu.

   Jadi tiap node berefek samping perlu menangani "sudah terjadi" sebagai
   sukses, bukan error (idempoten).
3. **Payload resume divalidasi** dengan Pydantic
   (`{"decision": "approve"|"reject"|"cancel", "feedback": str = "", "add_fields": list[str] = [], "user": str}`;
   `add_fields` hanya berlaku untuk reject dan hanya `name`/`description`/`tags`). API
   sudah memvalidasi, tapi graph tetap memeriksa.

## 6. Keputusan yang diusulkan

| Pertanyaan | Usulan | Alasan |
|---|---|---|
| Proposal stale saat approve | **berhenti** (`stop_reason="stale"`), UI menampilkan "usulan kedaluwarsa, minta lagi lewat chat" | jarang terjadi; sederhana dan mudah diaudit; seller tetap memegang kendali |
| Siapa memulai thread approval | lapisan aplikasi tepat setelah proposal dibuat (bukan service, bukan LLM) | service tidak perlu tahu graph; agent tidak perlu tahu workflow |
| Revisi gagal validasi | kirim error ke LLM, maks 2 percobaan, lalu `revision_failed` | sama seperti pola error tool di `graph.py` |
| Seller tidak pernah menjawab | thread tetap menunggu; kedaluwarsa otomatis di luar scope | belum perlu untuk demo |
| Proposal diputuskan langsung lewat API (di luar graph) | endpoint approve/reject **selalu** lewat graph bila thread approval ada | satu jalur keputusan, audit konsisten |

## 7. Penyambungan (bagian assistant, setelah graph jadi)

- Checkpointer: `SqliteSaver` di `var/approvals.sqlite`, terpisah dari
  `seller.sqlite` (state workflow ≠ data bisnis).
- `start_approval(proposal_id)`: dipanggil setelah proposal agent dibuat;
  `invoke` sampai berhenti di interrupt.
- Endpoint `/approve`, `/reject` (+ `add_fields`), dan `/cancel`:
  `graph.invoke(Command(resume={...}), {"configurable": {"thread_id": ...}})`.
- Endpoint baru `GET /caption-proposals/{id}/approval`: status thread
  (menunggu / selesai + `stop_reason`).

## 8. Test yang dibutuhkan

1. approve → caption berubah, `stop_reason="applied"`, audit `approved`.
2. reject → revisi dibuat dengan `revision_of`, graph menunggu lagi.
3. reject 5× → `max_rejections`, caption tidak berubah, tidak ada proposal ke-6.
4. approve proposal stale → `stop_reason="stale"`, caption tidak berubah.
5. revisi tidak boleh mengubah field di luar `fields`.
6. draft tidak valid 2× (fake LLM) → `revision_failed`.
6b. reject dengan `add_fields=["name"]` → revisi boleh mengubah nama; tanpa itu, draft yang mengubah nama ditolak.
6c. reject tanpa feedback → tetap direvisi.
6d. cancel → proposal `cancelled`, caption tidak berubah, tidak ada revisi.
7. resume dua kali untuk keputusan yang sama → hasil tetap sekali (idempoten).
8. graph dibangun ulang dengan checkpointer yang sama (simulasi restart)
   → resume tetap jalan.

Semua memakai fixture `service` (`tests/seller/conftest.py`), `InMemorySaver`
atau `SqliteSaver` di `tmp_path`, dan fake model untuk `draft_revision`.

## 9. Keputusan Nigka (20 Nov 2026)

1. **Seller boleh menambah field saat reject.** → `add_fields` di payload
   resume (eksplisit, lihat bagian 3).
2. **Feedback tidak wajib.** → reject tanpa alasan tetap direvisi; prompt
   revisi meminta varian yang jelas berbeda dari semua versi yang ditolak.
3. **Perlu tombol batal.** → keputusan `cancel`, status backend baru
   `cancelled` (sudah ditambahkan: `service.cancel`, `POST …/cancel`).
4. **Graph langsung dengan LLM** (tanpa tahap fungsi palsu). Test tetap
   memakai fake model untuk `draft_revision`, supaya deterministik dan
   gratis; panggilan model asli hanya untuk uji manual / eval.
