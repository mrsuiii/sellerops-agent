# Step-by-Step Plan — SellerOps Agent

Terakhir diperbarui: 17 November 2026
Versi sebelumnya (19 Agustus 2026, Session 1–14 untuk jalur SFT → DPO) ada di
git history dan di `../../archive/PLAN-sellerops-2026-08-19.md`.

Sasaran dan scope ada di `../../TARGET.md`. Konteks ringkas untuk sesi AI ada
di `CLAUDE.md`. Dokumen ini mengatur **cara mengerjakannya bertahap**.

## Keputusan stack (17 November 2026)

Seluruh agent dan workflow dibangun di atas **LangGraph / LangChain v1**,
tetapi **tanpa `create_agent`**: isi `create_agent` (model node ⇄ tools node,
conditional edge, stop condition, batas langkah) disusun sendiri sebagai
`StateGraph`. Komponen framework di luar itu dipakai bila ada (chat model,
`@tool`, reducer `add_messages`, `RetryPolicy`, checkpointer, `interrupt`,
`Store`, streaming, LangSmith). Middleware LangChain tidak dipakai karena
hanya berlaku di `create_agent`; padanannya ditulis sebagai node / edge /
field state. `ToolNode` dan source `create_agent` boleh dibaca sebagai
referensi pembanding.

| Kebutuhan | Fitur framework yang dipakai |
| --- | --- |
| Model Claude / NVIDIA | `init_chat_model` → `ChatAnthropic` (`langchain-anthropic`), `ChatOpenAI` dengan `base_url` (`langchain-openai`) |
| Agent loop (tool calling) | `StateGraph` sendiri: `model` node → conditional edge → `tools` node → `model` …; state berbasis `add_messages` |
| Tool | `@tool` / `StructuredTool` dari `langchain_core`, `model.bind_tools(...)`; eksekusi di `tools` node sendiri (`ToolNode` hanya referensi) |
| Jawaban akhir terstruktur (`final` / `ask_user`) | Tool `final` / `ask_user` dengan skema Pydantic; edge mengakhiri graph saat tool itu dipanggil |
| Max steps | Counter di state + cek di conditional edge; `recursion_limit` sebagai pengaman terakhir |
| Retry API, fallback model | `retry_policy=RetryPolicy(...)` pada `add_node`, `max_retries` di chat model, `model.with_fallbacks([...])` |
| Prompt caching | `cache_control` pada system prompt / tools lewat `ChatAnthropic`; cek `usage_metadata` |
| Ringkasan konteks | Node `summarize` sendiri yang dipicu edge saat melewati budget token |
| Approval manusia | `interrupt()` + `Command(resume=...)` (agent maupun workflow) |
| Workflow pasti (approval caption, KPI) | `StateGraph` custom + checkpointer |
| Simpan state, resume, replay | Checkpointer (`InMemorySaver` untuk test, `SqliteSaver` lokal), `get_state_history`, replay dari `checkpoint_id` |
| Memory jangka panjang | LangGraph `Store` |
| Tracing & eval | LangSmith (sudah dipakai di `evaluation/`) |
| Streaming ke UI | `graph.stream(..., stream_mode="messages" / "updates")` |

Parameter `ChatAnthropic` (effort, cache, refusal fallback) dan API
LangGraph **diverifikasi ke dokumentasi versi terpasang sebelum dipakai**.
Versi terpasang di env conda (17 Nov 2026): `langchain` 1.4.2,
`langchain-core` 1.6.4, `langgraph` 1.2.12, `langgraph-checkpoint` 4.2.0.

Dependency baru yang dibutuhkan (dijelaskan dulu sebelum di-install, lalu
ditambahkan ke extra `agent` di `pyproject.toml`): `langchain-anthropic`,
`langchain-openai`, `langgraph-checkpoint-sqlite`.

## Kontrak belajar untuk AI assistant (Claude, Codex, dsb.)

Arahan user (13 September 2026, diperbarui 17 November 2026): bagian agentic
(pemilihan dan konfigurasi model, desain state, graph/node/edge, routing,
stop condition, retry, approval, orkestrasi) **disusun user sendiri dengan
primitif LangGraph** (`StateGraph`, node, edge), bukan `create_agent`. Assistant menjelaskan konsep
dan fitur framework yang relevan, mereview, dan memberi hint bertahap; tidak
menulis solusi kecuali diminta eksplisit. Jika user sedang mengejar deadline
dan meminta implementasi penuh, instruksi terbaru user menang.

Pada setiap sesi, assistant harus:

1. Membaca `CLAUDE.md`, `../../AGENTS.md`, `../../TARGET.md`, `PLAN.md`, dan
   bagian terbaru `../../LEARNING_LOG.md` sebelum mengubah proyek.
2. Melanjutkan dari session pertama yang belum selesai; jangan mengulang yang
   sudah lolos test.
3. Mengerjakan **satu session** dalam satu waktu kecuali diminta lebih.
4. Menjelaskan tujuan, teori minimum, fitur framework yang dipakai (dan apa
   yang disembunyikannya), bentuk input/output, dan definition of done sebelum
   menulis kode, serta alasan setiap keputusan desain.
5. Boleh membuat scaffold, fixture, boilerplate, plumbing, dan test, tetapi
   menyisakan bagian inti sebagai `TODO` / `FILL_ME` (maksimal 3–5 per sesi).
6. Tidak mengisi TODO user; review percobaan user, tunjukkan invariant yang
   gagal, beri hint bertahap.
7. Menjalankan test relevan (`make lint test`) dan menjelaskan hubungan
   failure dengan konsep.
8. Mencentang checkbox hanya setelah definition of done terverifikasi.
9. Menutup sesi dengan ringkasan: yang selesai, bukti test/metric, TODO user
   berikutnya, session selanjutnya; catat di `../../LEARNING_LOG.md`.

Bagian mekanis (konfigurasi, serialization, CLI plumbing, fixtures, Docker,
frontend, visualisasi) boleh dikerjakan penuh oleh assistant.

## Aturan environment, data, dan rahasia

- Environment: Conda `llm-post-training` atau `uv sync --extra agent`
  (`pip install -e ".[dev]"`). Jelaskan sebelum menambah/menghapus dependency.
- Jangan pernah membaca, mencetak, atau meng-commit `.env` / API key.
- Frozen test tasks tidak dipakai untuk training, prompt tuning, atau
  patching spesifik-failure.
- Simpan `task_id`, seed, split, environment version, model id, dan biaya
  (token, latency) di setiap eval run.
- Jangan simpan hidden chain-of-thought di trajectory; simpan action,
  observation, hasil evaluator, dan alasan singkat.
- Tanpa GitHub Actions CI. Nama branch bermakna (`feat/...`, `fix/...`).
  User lebih suka perubahan langsung di lokal kecuali diminta PR.

## Arsitektur yang dituju

```text
                 ┌────────────── LangGraph StateGraph (disusun sendiri) ──────────────┐
user / task ──▶  │ START → model ──(tool call)──▶ tools ──▶ model …                   │
                 │           └──(final / ask_user / step habis / invalid)──▶ END      │
                 │ state: messages (add_messages), steps, final_action, stop_reason   │
                 │ RetryPolicy, with_fallbacks, node summarize, interrupt() approval  │
                 │ checkpointer: state history, resume, replay                        │
                 └──────────┬───────────────────────────────┬─────────────────────────┘
                            │                               │
                     init_chat_model                  @tool + bind_tools
          (ChatAnthropic | ChatOpenAI + base_url)    (SellerEpisode / Kala Studio API)

Workflow pasti (approval caption, KPI) = StateGraph custom + interrupt().
Tracing + eval = LangSmith.
```

Struktur kode tumbuh bertahap di `src/sellerops/harness/`, misalnya
`models.py` (chat model dari settings), `state.py`, `graph.py` (node + edge
agent), `tools/` (tool production: seller di Track C, `store.py` di B1),
`trajectory.py`; tool dan kontrak khusus eval di `src/sellerops/evaluation/`
(`seller_tools.py`, `terminal_tools.py`) — `harness` tidak mengimpornya; workflow di
`src/sellerops/workflows/`. Jangan membuat semua file kosong sekaligus.

Urutan track (keputusan 17 Nov 2026, mengikuti spec seller agent):
**0 → A → C (seller) → B (buyer) → D → E**. Nama track tidak diubah agar
rujukan lama (mis. C3, B1) tetap berlaku.

Kode Track 0 (`baseline/`, simulator offer/diskon lama, dataset
`sellerops-sft-v0.2`, runner LangSmith) **dihapus 17 Nov 2026**; tersedia di
git history (commit `1d3b5e0` dan sebelumnya) dan artefak di
`../../archive/sellerops-sft-0.5b/`.

---

## Track 0 — Fondasi dan baseline (selesai, diarsipkan)

Ringkasan Session 1–10 lama; detail di versi arsip dan `LEARNING_LOG.md`.
Kodenya sudah dihapus dari repo (lihat atas); daftar ini catatan sejarah.

- [x] **S1 Data contracts**: `SellerTask`, `ToolCall`, `AskUser`,
  `FinalAnswer`, tagged action format, test valid/invalid.
- [x] **S2 Environment + tools**: catalog, inventory, discount policy,
  `search_products`, `get_inventory`, `get_discount_policy`,
  `calculate_offer`, structured success/error.
- [x] **S3 Oracle + frozen split**: `sellerops-sft-v0.2` (180 train, 30 dev,
  48 frozen test, 440 SFT records), manifest + checksum.
- [~] **S4 Episode service**: `create_episode`, `reset`, snapshot, replay,
  outcome verifier selesai. Sisa: integrasi `scenario_id` + seed dan test
  isolasi dua episode.
- [x] **S5–S7 Baseline + SFT**: full SFT Qwen2.5-0.5B-Instruct (Colab);
  schema validity 17.9% → 100%, exact action 0% → 70.5% (dev). Hasil di
  `results/baseline`, `results/sft`.
- [x] **S8 (versi router)**: LangGraph router → action → tools, message
  conversion, model aliases. Sejak 17 Nov 2026 ada di
  `src/sellerops/baseline/`.
- [x] **Eval harness**: LangSmith runner, answer keys, bounded schema retry,
  pass@1/pass@k, judge Gemini opsional.
- [x] **Kala Studio** (PR #2), **production-readiness** (PR #3, #4),
  **config provider LLM** (PR #5, #6).

Temuan kunci: e2e dev eval **0/30** (6 Oktober 2026). Router base-model
mengeluarkan teks pada `act`, retry pindah ke `ask`; 60 router call, 0 SFT
call, 0 tool call. Hasil di
`results/evaluation/20261006T005916Z-ebe22193-sellerops-sft/` (kini di arsip).
Tetap jadi bahan cerita debugging, tetapi bukan pembanding eval baru (domain
task-nya berbeda).

---

## Track A — Agent LangGraph + model API

### Session A1 — Chat model, tool, dan agent graph minimal

Status: [x] (17 Nov 2026). Uji API sungguhan (`claude-opus-5-5`, effort
medium, `ScriptedUser`): `omzet-30-hari` dan `toko-tanpa-periode` (multi-turn,
agent bertanya periode dulu) lolos verifier; prompt cache terbaca
(`cache_read` ~3,0–3,3 rb token per call setelah call pertama).

Tujuan: satu task simulator seller berjalan end-to-end dengan Claude API lewat
`StateGraph` agent yang disusun sendiri (model ⇄ tools), termasuk satu
percakapan multi-turn (`ask_user` → balasan user → `final`).

Kerjakan:

- [x] Dependency `langchain-anthropic` di extra `agent` (`langchain-openai`
  untuk provider kedua belum; `uv.lock` belum diputuskan).
- [x] `harness/models.py`: `init_chat_model` dari `get_settings()` (provider,
  model, effort). Refusal fallback server-side belum dipasang (A2).
- [x] `evaluation/seller_tools.py`: enam tool seller (`StructuredTool`) yang
  memanggil `SellerEpisode.call`; episode terikat lewat closure, bukan
  argumen model.
- [x] Tool `final` / `ask_user` di `evaluation/terminal_tools.py` (Pydantic,
  dioper ke graph sebagai `terminal_tools`) sehingga hasil akhir langsung
  masuk `verify_conversation`.
- [x] `harness/state.py`: state agent (messages dengan `add_messages`, jumlah
  step, `final_action`, alasan berhenti). Diimplementasi assistant atas
  permintaan eksplisit user (17 Nov 2026).
- [x] `harness/graph.py`: node `start_turn` (reset per giliran), `model`,
  `tools`, conditional edge setelah model dan tools, batas langkah per giliran
  (beda dengan `recursion_limit`).
- [x] Prompt caching lewat `cache_control` di `ChatAnthropic`; terverifikasi
  lewat `usage_metadata["input_token_details"]["cache_read"]`.
- [x] Test tanpa network (`tests/harness/test_graph.py`, scripted chat model):
  happy path, ask_user, teks tanpa tool, batas langkah, final invalid, error
  tool, mode chat, dan percakapan multi-turn dengan `ScriptedUser`. Di-skip
  sampai `build_agent_graph` diimplementasi.
- [x] Adapter Anthropic tulis tangan sudah dihapus.

Bagian inti untuk user:

- Desain state, node `model` dan `tools`, serta conditional edge.
- Memilih stop condition dan perilaku saat batas langkah habis atau model
  membalas tanpa tool call.

Definition of done:

- Semua test di `tests/harness/test_graph.py` lolos (tidak ada skip).
- Satu dev task lolos `verify_conversation` dengan Claude API sungguhan.
- Agent berhenti aman pada output invalid atau batas langkah (ada test-nya).

### Session A2 — Robustness: error, retry, timeout

Status: [ ]

- [ ] Tool error dikembalikan sebagai `ToolMessage(status="error")`
  terstruktur dari node `tools`, bukan exception.
- [ ] Retry API (rate limit, overload) lewat `RetryPolicy` pada node `model`
  dan retry bawaan client, terpisah dari retry perilaku model;
  `with_fallbacks` untuk model cadangan.
- [ ] Timeout per call dan per episode; budget token.
- [ ] Test adversarial: tool tidak ada, argumen invalid, produk tidak ada,
  proposal caption melanggar batas.

Bagian inti untuk user: memetakan jenis error ke retry / lanjut / berhenti dan
menaruhnya di node, edge, atau `RetryPolicy` yang tepat.

DoD: setiap jenis error punya perilaku eksplisit dan test.

### Session A3 — Eval multi-turn seller agent

Status: [ ]

- [ ] CLI eval (`scripts/evaluate_seller.py`) yang menjalankan agent A1 per
  task dev lewat `run_conversation` (episode + `thread_id` per percakapan,
  `LLMUser` atau `ScriptedUser`), k ≥ 3 repetisi; laporkan pass@1, pass@k,
  dan pass^k (`metrics.py`) per family.
- [ ] Catat token, biaya, latency p50/p95, rata-rata steps (dari LangSmith /
  `usage_metadata`).
- [ ] Error analysis minimal 5 kegagalan (ala Hamel Husain) dari trace
  LangSmith; failure penting jadi regression test.
- [ ] Opsional: model NVIDIA via `ChatOpenAI` + `base_url` sebagai pembanding
  ketiga.

Bagian inti untuk user: membaca trajectory gagal dan menulis RCA.

DoD: tabel hasil per family (ScriptedUser vs LLMUser) tersimpan di
`results/evaluation/` dengan config, model id, data/tasks version; frozen test
belum disentuh.

### Session A4 — Context, memory, trajectory replay

Status: [ ]

- [ ] Context budget dengan node `summarize` sendiri (dipicu edge saat token
  melewati budget; pesan lama diganti lewat `RemoveMessage`).
- [ ] Checkpointer `SqliteSaver`; trajectory diambil dari `get_state_history`
  + trace LangSmith, diekspor ke JSONL (task, model, messages, tool calls,
  observations, hasil evaluator, latency, usage).
- [ ] Replay / time travel dari `checkpoint_id` pada environment version yang
  sama.
- [ ] Opsional: memory preferensi lewat LangGraph `Store`, di-load hanya saat
  relevan.

DoD: satu failed trajectory bisa direproduksi; satu task multi-turn memakai
ringkasan dengan benar.

---

## Track C — Seller agent (dikerjakan sebelum Track B)

Spec: https://claude.ai/artifact/Wond21GKwf7ui7xdgcSHSa

Catatan (17 Nov 2026) — simulator seller dan eval multi-turn (selesai):

- Simulator lama (offer/diskon, satu task = satu pesan) sudah dihapus.
  Dunia eval Track A dan C sekarang **simulator seller** (`simulator/`):
  katalog Kala Studio + order, stok, review sintetis 90 hari dengan skenario
  tertanam. Detail: `src/sellerops/evaluation/README.md`.
- Eval **multi-turn** (pola τ-bench) tersedia: task dengan `hidden_info`,
  persona, tujuan; `ScriptedUser` / `LLMUser`; runner giliran
  (`conversation.py`, checkpointer + `thread_id`); `ask_user` mengakhiri
  giliran dan balasan user masuk sebagai `ToolMessage` untuk call itu;
  verifier menilai hasil akhir (+ jumlah giliran, pass^k).
- Graph agent (Track A) tidak perlu tahu soal multi-turn: runner mereset
  `final_action` / `stop_reason` di input setiap giliran.

### Session C1 — Data order sintetis

Status: [x] (17 Nov 2026)

- [x] Generator order, stok, dan review sintetis yang deterministik (seed)
  untuk katalog Kala Studio, dengan pola yang bisa diuji (lonjakan, stok
  habis → penurunan kategori, stok menipis, produk tidak laku, review
  negatif berulang): `simulator/generator.py`, `make sim`.
- [x] Read model penjualan, performa produk, stok + restock, review, produk,
  proposal caption: `simulator/store.py` (endpoint HTTP menyusul di C2/C3
  bersama `StoreClient`).
- [x] 19 task eval (10 dev, 9 test frozen) dengan kunci jawaban dihitung dari
  data; verifier + reference solution lolos semua task.

DoD: dataset tersimpan dengan seed dan manifest (checksum); test pola di
`tests/simulator/`.

Keputusan terbuka di spec yang diambil default di C1: target restock N = 14
hari (`target_days`, bisa diubah), "perlu restock segera" = habis ≤ 7 hari.

### Session C2 — Analisis penjualan (agent)

Status: [ ]

- [ ] Tool baca: ringkasan penjualan, top/lambat, stok + saran restock,
  insight review; seller agent = graph agent Track A.
- [ ] Task eval dengan jawaban yang bisa diverifikasi dari data sintetis.

DoD: seller agent menjawab pertanyaan analisis dan lolos verifier.

### Session C3 — Perbaikan caption dengan approval (workflow)

Status: [ ]

- [ ] `StateGraph` custom: node LLM mengusulkan caption → `interrupt()` untuk
  approve/reject → `Command(resume=...)` → node backend PATCH → audit log →
  revert. Checkpointer wajib agar approval bisa ditunda.
- [ ] LLM hanya mengisi node usulan; alur, approval, dan penulisan data
  dikendalikan graph.

Bagian inti untuk user: desain state, node/edge approval, dan audit log.

DoD: caption tidak pernah berubah tanpa approval; revert terbukti lewat test.

### Session C4 — KPI dan laporan mingguan (workflow)

Status: [ ]

- [ ] `StateGraph` deterministik: node hitung KPI → node narasi LLM → node
  cek angka narasi terhadap KPI.

DoD: laporan mingguan deterministik dari data sintetis + narasi LLM yang
diperiksa terhadap angka.

---

## Track B — Buyer agent di atas Kala Studio (setelah Track C)

### Session B1 — Tool Kala Studio

Status: [ ]

- [ ] Tool `@tool` `search_products`, `recommend_products` (BM25
  `POST /api/recommend`), `get_product`, `add_to_cart` yang memanggil store
  API (atau service layer-nya langsung untuk test).
- [ ] Deskripsi tool dan error message dirancang untuk model (lihat
  *Writing effective tools for agents*).
- [ ] Hanya data publik; tidak ada aksi harga atau negosiasi.

DoD: tiap tool punya test sukses dan gagal.

### Session B2 — Buyer agent + task eval

Status: [ ]

- [ ] Buyer agent = graph agent Track A dengan tool B1.
- [ ] 20–30 task belanja (constraint ukuran, budget IDR, kategori, produk
  tidak ada, permintaan ambigu → harus bertanya) dengan verifier
  programmatic.
- [ ] Eval pass@1 / pass^k + biaya.

DoD: demo satu percakapan belanja sampai item masuk cart; hasil eval
tersimpan.

### Session B3 — Chat UI di toko

Status: [ ]

- [ ] Endpoint chat dengan streaming `graph.stream(stream_mode="messages")`
  dan checkpointer per percakapan (`thread_id`); widget chat di frontend
  Kala Studio.
- [ ] Tampilkan tool call yang terjadi (transparansi untuk demo).

DoD: `make store` lalu bisa belanja lewat chat di browser.

---

## Track D — Lanjutan (setelah A, C, dan B)

- [ ] **Analyst agent** dengan `run_code` (python/sql/bash) ter-sandbox di
  Docker, plus dashboard.
- [ ] **RAG katalog produk** / agent umum dengan retriever/vector store
  LangChain. Plan: https://claude.ai/artifact/77zTgEAD9uVorNcST3zVDj
- [ ] **Jalur post-training opsional**: perbaiki router 0.5B (constrained
  decoding / structured output) atau cari use case sendiri untuk 0.5B;
  DPO hanya jika failure audit menghasilkan preference pairs yang layak
  (lihat Session 11–13 di versi arsip).

## Track E — Report dan paket aplikasi

- [ ] README: diagram arsitektur, setup, satu perintah demo, satu perintah
  eval, tabel hasil.
- [ ] Analisis minimal lima failure case dan limitations.
- [ ] Penjelasan proyek versi 3 menit dan 15 menit.
- [ ] Resume bullet hanya dengan angka terverifikasi.

DoD: repo bisa dijalankan ulang dari config yang tercatat; README memisahkan
fitur selesai dan future work.

## Current next session

**A2** (robustness: `RetryPolicy`, fallback, timeout) dan **A3** (CLI eval
multi-turn). Catatan dari uji API A1: model menghitung sendiri angka turunan
di `message` (mis. "turun sekitar 34%") yang tidak ada di output tool;
verifier hanya mengecek `final.data`. Perlu diputuskan: tambah field
pembanding di tool, perketat prompt, atau nilai `message` lewat judge (A3).
