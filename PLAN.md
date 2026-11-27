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
agent), `tools/` (tool production: `seller.py` di atas `SellerDataClient`, `store.py` di B1),
`trajectory.py`; tool dan kontrak khusus eval di `src/sellerops/evaluation/`
(`seller_tools.py`, `terminal_tools.py`) — `harness` tidak mengimpornya; workflow di
`src/sellerops/workflows/`. Jangan membuat semua file kosong sekaligus.

Urutan track (keputusan 17 Nov 2026, mengikuti spec seller agent):
**0 → A → C (seller) → B (buyer) → D → E**; A5 (context & memory) disisipkan
sesudah C3 (keputusan 18 Nov). Keputusan 27 Nov: target dekat berhenti di
**A5** (C3 tuntas → C3b → A5); C4 dan buyer agent (Track B) di sesi lain. Nama track tidak diubah agar
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

Status: [x] (17 Nov 2026; ditulis assistant atas permintaan eksplisit Nigka).
Peta error lengkap di `harness/errors.py`.

- [x] Error tool → `ToolMessage(status="error")` berisi `{"ok": false,
  "error": {"code", "message"}}` (`unknown_tool`, `invalid_arguments`,
  `tool_error`, `terminal_not_alone`), bentuk sama dengan error bisnis.
- [x] Retry API: SDK `max_retries=2` + `RetryPolicy` (2 percobaan, backoff)
  pada node `model`/`summarize` dengan `is_transient_error` (429/408/409/5xx/
  529/timeout/jaringan); 4xx lain gagal cepat. Node `tools` tidak di-retry
  (bisa menulis). `with_fallbacks` ke `SELLEROPS_LLM_FALLBACK_MODELS`.
- [x] Timeout per call (client), budget per giliran: `max_turn_tokens`,
  `max_turn_seconds` (`token_budget` / `time_budget`), dicek setelah semua
  tool call terjawab; prompt caching otomatis hanya untuk ChatAnthropic;
  `repair_tool_pairs` untuk thread yang lanjut setelah `final`.
- [x] Test adversarial (`tests/harness/test_robustness.py`): tool tidak ada,
  argumen invalid, produk tidak ada, caption melewati batas lalu pulih,
  retry, gagal cepat, fallback, budget.
- [x] Runtime context: `AgentContext` (`user_id`, `store_id`) lewat
  `context_schema`; tool dengan `runtime: ToolRuntime` mendapat context.

### Session A3 — Eval multi-turn seller agent

Status: [x] (17 Nov 2026)

- [x] CLI eval (`scripts/evaluate_seller.py`, `make eval-smoke` / `make eval`;
  logika di `evaluation/seller_eval.py`): episode + graph + `thread_id` per
  percakapan, `ScriptedUser` atau `LLMUser`, k repetisi paralel; pass@1, pass@k,
  pass^k per family; judge NVIDIA opsional (`--judge`) untuk `message`.
  Smoke run pertama (17 Nov, Haiku 5.5, scripted, k=1): 8/10.
- [x] Token (input/output/cache), latency p50/p95, rata-rata tool call dan
  giliran per run (`UsageMetadataCallbackHandler`); biaya dalam $ belum.
- [x] Run penuh `--user llm --k 3 --judge` (17 Nov, Haiku 5.5, judge
  `nvidia/nemotron-3-ultra-550b-a55b`): pass@1 0.83, pass@3 0.90, pass^3 0.80;
  gagal di caption (3/3, `caption_flow`) dan slow_movers (2/3, `answer`);
  judge 24/25, 0 error. `results/evaluation/20261117-155835-*`.
  Lihat kegagalan: `python scripts/show_failures.py [--judge]`.
- [x] Error analysis 5 akar masalah (`docs/error-analysis-2026-11-17.md`):
  caption mengubah field tak diminta, klarifikasi tidak lengkap, tren dikarang
  (→ `compare_to_previous` di tool), bug `run_id` runner, kontrak `final.data`
  (→ validator bentuk). Dev: pass@1 0.83 → **1.00**, pass^3 0.80 → **1.00**
  (`results/evaluation/20261117-163243-*`). Regression:
  `tests/evaluation/test_eval_regressions.py`.
- [x] Pembanding NVIDIA `nemotron-3-ultra-550b-a55b` sebagai agent
  (`--agent-provider openai_compatible`): pass@1 0.83, pass^3 0.80, p50 19 s.
- [ ] Validasi judge dengan label manusia (`scripts/judge_labels.py`):
  butuh label Nigka.
- [ ] Angka final split test (`--allow-test`), sekali, untuk laporan.

Bagian inti untuk user: membaca trajectory gagal dan menulis RCA.

DoD: tabel hasil per family (ScriptedUser vs LLMUser) tersimpan di
`results/evaluation/` dengan config, model id, data/tasks version; frozen test
belum disentuh.

### Session A4 — Context, memory, trajectory replay

Status: [x] (17 Nov 2026), kecuali memory `Store` (opsional).

- [x] Node `summarize`: dipicu di awal giliran saat input token panggilan
  terakhir ≥ `max_context_tokens`; pesan lama diringkas lalu dihapus dengan
  `RemoveMessage` (potongan tidak pernah memisahkan tool call dari hasilnya);
  ringkasan dikirim setelah system prompt.
- [x] `SqliteSaver` (`harness/trajectory.py`), ekspor trajectory (pesan, tool
  call, usage, timeline checkpoint dari `get_state_history`); eval menyimpan
  pesan per percakapan dan, dengan `--save-threads`, seluruh thread.
- [x] Replay dari `checkpoint_id` (`scripts/replay_eval_thread.py`): episode
  simulator dibangun ulang dari tool call sebelum checkpoint; kegagalan
  caption baseline direproduksi 2/2.
- [ ] ~~Opsional: memory preferensi lewat LangGraph `Store`~~ → dipindah ke A5.

DoD: satu failed trajectory bisa direproduksi (caption); ringkasan multi-turn
teruji (`tests/harness/test_context.py`).

### Session A5 — Context engineering & memory (episodic, semantic, procedural)

Status: [ ] — dikerjakan **sesudah C3 dan C3b** (alasan reject caption =
sumber memori pertama; compaction perlu placeholder gambar dari C3b).
Keputusan 18 Nov 2026, urutan C3b diperbarui 27 Nov. Target dekat berhenti di
sini.

Tujuan: satu komponen yang menentukan *apa saja* yang masuk ke prompt, dari
mana, seberapa banyak, dan dalam urutan apa, alih-alih menempel seluruh
thread. Pembagian memori mengikuti CoALA:

| Jenis | Isi | Contoh SellerOps | Tempat |
|---|---|---|---|
| Working | percakapan berjalan | pesan + hasil tool thread ini | checkpointer (sudah ada) |
| Episodic | kejadian | "CP-0003 ditolak: terlalu panjang" | `Store` + data backend |
| Semantic | fakta | "seller selalu pakai 30 hari", RAG katalog | `Store`, indeks RAG |
| Procedural | cara bertindak | system prompt, deskripsi tool, "caption tanpa emoji" | kode ber-versi + `Store` per seller |

Susunan prompt (urut dari paling stabil, supaya prompt cache tidak rusak):
procedural umum → procedural seller → semantic seller → episodic relevan →
ringkasan thread → working memory. Budget token per bagian; yang dibuang
lebih dulu saat penuh: episodic paling tidak relevan, lalu hasil tool lama;
procedural tidak pernah dibuang.

Rencana folder (dibuat bertahap, jangan semua file kosong sekaligus):

```
src/sellerops/harness/context/
├── builder.py      # susun prompt: urutan bagian, budget token, breakpoint cache
├── compaction.py   # dari graph.py: summarize, repair_tool_pairs, + buang hasil tool lama
├── memory.py       # tipe memori, namespace Store per store_id, baca/tulis
└── extract.py      # jalur tulis: ekstraksi memori setelah thread selesai
```

`harness/context` tetap generik (tidak tahu seller); sumber memori spesifik
domain (mis. feedback reject C3) ditulis di `seller/` / `workflows/` dan
dioper sebagai *provider*, seperti tool dioper ke `build_agent_graph`.

Langkah:
- [ ] Desain (Nigka): format memori (isi, sumber/provenance, waktu), budget
  per bagian, aturan prioritas saat penuh.
- [ ] `builder.py` (Nigka): context builder dipakai node `model`.
- [ ] Pindahkan compaction dari `graph.py` ke `context/compaction.py` +
  *tool result clearing*: hasil tool dari giliran lama diganti ringkasan
  pendek di prompt (state tetap utuh). (assistant, mekanis)
- [ ] Sumber pertama, satu per jenis: procedural = prompt yang ada;
  semantic = preferensi periode seller; episodic = alasan reject caption
  dari backend C3 (deterministik, tanpa LLM).
- [ ] Jalur tulis bertahap: deterministik dari backend dulu → ekstraksi
  setelah thread selesai → (opsional) tool `remember`.
- [ ] Penjagaan: namespace per `store_id` (tidak bocor antar seller),
  provenance per memori, seller bisa melihat/menghapus, memori = data bukan
  instruksi (tidak boleh mengalahkan system prompt).
- [ ] Eval: task "seller kembali di thread baru" (memori harus dipakai, mis.
  tidak menanyakan ulang periode; caption revisi menghindari alasan reject
  lama). Eval biasa menjalankan agent **tanpa** memori agar task tetap
  terisolasi. Cek juga celah: konteks membengkak di tengah giliran (banyak
  tool besar) dan kualitas ringkasan.

DoD: prompt disusun oleh builder dengan budget yang teruji; satu memori per
jenis terbukti dipakai di task eval thread baru; memori satu seller tidak
pernah muncul di prompt seller lain (test).

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

Status: [x] (18 Nov 2026)

- [x] Tool seller di `harness/tools/seller.py` (`make_seller_tools(client)`,
  Protocol `SellerDataClient`); eval memakai tool yang sama dengan
  `SellerEpisode` sebagai client (17 Nov).
- [x] Service layer + in-process client untuk production (18 Nov, assistant):
  `seller/analytics.py` (analitik bersama; simulator kini memakainya juga,
  test paritas), `seller/service.py` (`SellerService` di SQLite,
  `InProcessSellerClient`), endpoint `/api/seller/*` (`seller/api.py`).
  Seller agent = graph agent Track A + `make_seller_tools(InProcessSellerClient(service))`.
- [x] Task eval dengan jawaban yang bisa diverifikasi dari data sintetis
  (19 task C1; eval dev pass@1 1.00, lihat `docs/error-analysis-2026-11-17.md`).

DoD: seller agent menjawab pertanyaan analisis dan lolos verifier.

### Session C3 — Perbaikan caption dengan approval (workflow)

Status: [~] backend selesai (18 Nov); graph approval selesai (21 Nov:
`load` + `await_decision` ditulis bersama Nigka, sisanya assistant atas
permintaan eksplisit Nigka) di `workflows/approval/` (`approval_graph.py`,
`approval_prompts.py`, `approval_state.py`; 18 test di
`tests/workflows/`). Wiring ke API selesai 27 Nov
(`workflows/approval/runner.py` + `api.py`, route `…/{id}/approval`).
Halaman seller `/seller` selesai 27 Nov (`web/seller.*`; toko menampilkan
caption dari DB seller). Merge ke `main` lewat PR #7. Tinggal: coba reject →
revisi dengan model asli (`make test-live` / `make try-approval` /
`make store` + `/seller`; berbayar).

- [ ] `StateGraph` custom: node LLM mengusulkan caption → `interrupt()` untuk
  approve/reject → `Command(resume=...)` → node backend PATCH → audit log →
  revert. Checkpointer wajib agar approval bisa ditunda.
- [ ] LLM hanya mengisi node usulan; alur, approval, dan penulisan data
  dikendalikan graph.

Keputusan (17 Nov 2026, Nigka):
- Approval memakai graph dengan `interrupt()` (bukan endpoint saja), supaya
  reject + feedback bisa lanjut ke node revisi LLM.
- Maksimal 5 kali reject; setelah reject ke-5 graph berhenti dengan
  `stop_reason="max_rejections"` (seller menolak terus-menerus).
- Backend (DB + service) tetap sumber kebenaran: status proposal, versi caption,
  audit log, revert. Node graph hanya memanggil service; revert di luar graph.
- Usulan caption **reaktif saja** (18 Nov): dibuat hanya saat seller
  memintanya di chat. Tidak ada workflow terjadwal yang mengusulkan caption
  sendiri dari analitik.

Backend C3 **selesai** (18 Nov, assistant): `seller/db.py` (tabel
`caption_proposals`, `caption_versions`, `audit_log`; trigger append-only dan
"caption hanya berubah bersama versi baru"), state machine di
`seller/service.py` (pending → applied / rejected / stale, applied →
reverted; approve = apply dalam satu transaksi; revisi lewat `revision_of`,
maks `MAX_REJECTIONS = 5`), endpoint approve/reject/revert, 40+ test
termasuk approve bersamaan dan rollback saat gagal di tengah.

**Sisa untuk Nigka: graph approval** (`src/sellerops/workflows/caption_approval.py`).
Usulan desain (state, node, alur, keputusan, test): `docs/c3-approval-graph-design.md`.
Tutorial langkah demi langkah (11 test, sudah dicek bisa dipenuhi): `docs/tutorial-c3-approval-graph.md`.
Yang dipanggil graph dari service:

| Node | Panggilan service |
|---|---|
| propose (LLM mengisi field) | `service.propose_caption(pid, reason, actor="agent", name=…, description=…, tags=…, revision_of=None / id ditolak)` |
| tunggu seller | `interrupt({"proposal": …})` → `Command(resume={"decision": "approve" \| "reject", "feedback": …, "user": …})` |
| approve | `service.approve(proposal_id, actor=f"seller:{user}")` → `SellerError("stale_proposal")` bila caption sudah berubah |
| reject | `service.reject(proposal_id, actor=…, feedback=…)`; `rejections_in_chain` di hasilnya |
| batal | `service.cancel(proposal_id, actor=…)` → `cancelled` (keputusan 20 Nov) |
| stop | `rejections_in_chain >= MAX_REJECTIONS` → `stop_reason="max_rejections"` (service juga menolak revisi ke-6: `max_rejections`) |

Revert tetap di luar graph (endpoint / `service.revert`). Pertanyaan desain
untuk Nigka: apa isi state (proposal_id, feedback, jumlah reject)? Node mana
yang boleh memanggil LLM? Apa yang terjadi bila resume datang untuk proposal
yang sudah `stale`?

Bagian inti untuk user: desain state, node/edge approval, dan audit log.

DoD: caption tidak pernah berubah tanpa approval; revert terbukti lewat test.

### Session C3b — Gambar produk sebagai input caption (multimodal)

Status: [x] — selesai 27 Nov 2026. Pipeline gambar dan backend oleh
assistant; graph dan prompt juga ditulis assistant atas permintaan eksplisit
Nigka (27 Nov: "segera selesaikan, mau di-deploy sebagai portofolio"). Sisa:
satu run live berbayar (`tests/integration/test_caption_c3b_live.py`).

Tujuan: model melihat foto produk saat menulis atau merevisi caption, supaya
caption bisa menyebut hal yang terlihat (warna, motif, potongan) tanpa
mengarang. Sekarang `draft_revision` hanya menerima teks (caption + data
produk).

Rancangan (dari diskusi 21 Nov):

- **Format pesan**: *standard content blocks* LangChain (`{"type": "image",
  "base64": …, "mime_type": …}` di `HumanMessage`), bukan format SDK
  Anthropic langsung, agar tetap jalan lewat `init_chat_model` dan provider
  lain.
- **Gambar diambil backend, bukan model**: backend mengunduh dari URL katalog
  (hanya host katalog yang dikenal, mis. `cdn.dummyjson.com`), mengecilkan
  (sisi terpanjang ≈1024 px, cukup untuk caption dan hemat token: ≈1.000–1.500
  token per gambar), menyimpan di cache berdasarkan hash SHA-256
  (`var/images/`), lalu mengirim base64. Model tidak pernah menerima URL
  mentah (tidak ada fetch dari URL yang dipilih model, hasil deterministik
  untuk test/replay).
- **Jejak di proposal**: proposal menyimpan hash gambar yang dipakai
  (`inputs`, kolom baru atau di `changes` meta) sehingga audit bisa menjawab
  "caption ini ditulis setelah melihat gambar apa".
- **Grounding**: prompt membedakan fakta data (bahan, garansi, harga) dan hal
  yang terlihat di foto. Yang hanya terlihat (mis. "biru tua") boleh ditulis,
  klaim yang tidak terlihat dan tidak ada di data tetap dilarang. Ditambah cek
  di eval/judge.
- **Kemampuan model**: flag `supports_vision(model)` di `harness/models.py`
  (seperti `supports_prompt_caching`); kalau model tidak mendukung gambar,
  graph jatuh ke mode teks saja, bukan gagal.
- **Konteks (kaitan A5)**: gambar mahal di prompt. Di compaction A5, gambar
  dari giliran lama diganti placeholder teks (`[gambar FS180 #a1b2c3, sudah
  dilihat]`); state tetap utuh.

Langkah:
- [x] Pipeline gambar backend (`seller/images.py`): `ImageStore` Protocol +
  `LocalImageStore` (`var/images/stores/<store_id>/images/<sha256>.jpg`,
  `SELLEROPS_IMAGES_DIR`), hanya https ke host katalog (redirect juga dicek),
  batas 10 MB / 40 MP, flatten ke putih, sisi terpanjang 1024 px, JPEG,
  cache per URL. `ProductImages.load(urls, limit)` melewati gambar gagal
  (log) supaya pemanggil lanjut dengan teks; `to_content_block` membuat
  block standar LangChain. `service.product_images(pid)` = URL katalog
  (thumbnail 300 px dulu, lalu `images` 1000 px). Pillow di extra `[agent]`.
- [x] `supports_vision(model)` di `harness/models.py` (Claude: ya;
  openai_compatible: hanya bila `SELLEROPS_LLM_VISION=1`; fallback chain:
  semua model harus bisa).
- [x] Desain: satu foto penuh 1000 px (`IMAGE_LIMIT = 1`, thumbnail 300 px
  hanya kalau tidak ada foto lain), di setiap draft (pertama dan revisi)
  oleh graph approval; chat agent tidak mengirim gambar (tetap teks).
- [x] Node `draft_caption` (pengganti `draft_revision`): gambar dimuat di dalam
  node setiap kali jalan, state hanya menyimpan ref (`draft_inputs`).
- [x] Jejak gambar di proposal: `propose_caption(..., inputs={"images":
  [asdict(ref), ...]})` (kolom `inputs`, migrasi untuk DB lama), tampil di
  `/seller` lewat `GET /api/seller/images/{sha256}` (assistant).
- [x] Test: fake model menerima content block gambar; host di luar daftar
  ditolak; cache hit tidak mengunduh ulang; model tanpa vision tetap jalan
  (`tests/workflows/test_caption_c3b.py`, `tests/seller/test_images.py`).
- [x] Eval workflow caption (27 Nov, `scripts/evaluate_caption.py`, 11
  skenario, cek kode + cross-judge): Haiku k=3 pass@1 0.91 / pass^3 0.73
  (kode 1.00, bertanya benar 33/33); Nemotron k=1 0.64. Laporan:
  `docs/eval-caption-2026-11-27.md`. Terbuka: bahasa caption (27/33 ditulis
  dalam Bahasa Indonesia walaupun caption sekarang Inggris), kalibrasi judge.
- [ ] Test live (`make test-live`, opsional; eval di atas sudah mencakupnya).

DoD: revisi caption memakai foto produk lewat backend (tanpa URL ke model),
hash gambar tercatat di proposal, dan alur tetap jalan untuk model tanpa
vision.

**Bagian 2: agent boleh bertanya ke seller** (keputusan Nigka, 27 Nov).
Foto menjawab yang terlihat (warna, motif, potongan); yang tidak terlihat dan
tidak ada di data (bahan, ukuran, perawatan) ditanyakan ke seller, bukan
dikarang. Tetap workflow (flowchart bisa digambar dulu), bukan agent loop.

Keputusan:
1. **Dari awal**: agent boleh bertanya sebelum caption pertama, bukan hanya
   saat revisi. Caption pertama ditulis graph lewat *caption request*
   (`CR-0001`: produk, field, instruksi seller).
2. **Apa adanya**: kalau seller tidak tahu (jawaban kosong → `None`),
   caption tetap ditulis dari yang ada, dan `reason` menegaskan bahwa data
   tambahan dibutuhkan. Tidak ada putaran tanya ulang tanpa batas.
3. **Memory**: jawaban disimpan sebagai fakta produk (`product_facts`,
   append-only, jawaban terbaru per key menang), jadi lain kali tidak perlu
   bertanya lagi. Ini bahan pertama A5 (semantic memory per produk).
4. **Audit**: fakta yang dipakai dicatat di `inputs.seller_facts` proposal.

Langkah:
- [x] Backend (assistant): tabel `caption_requests` + `product_facts`,
  `service.request_caption / get_request / list_requests / cancel_request`,
  `remember_facts / product_facts`, `propose_caption(request_id=...)`
  (request → `proposed` dalam transaksi yang sama, hanya field yang diminta);
  `GET/POST /api/seller/products/{id}/facts`, `GET /caption-requests`,
  `POST /caption-requests/{id}/cancel`.
- [x] Kontrak interrupt (`workflows/approval/questions.py`): `SellerQuestion`,
  `question_request(...)` (`kind="question"`), `AnswerIn` / `Answer`
  (`{"answers": {key: str | None}, "user"}`), maks 3 pertanyaan.
- [x] Runner + API (assistant): thread `caption-request-CR-…` (proposal dengan
  `request_id` ikut thread itu), view `waiting_for` = `decision` | `answer`,
  `answer` menyimpan fakta dulu lalu resume, `POST /caption-requests`,
  `.../approval/answer` untuk request dan proposal. Halaman `/seller`:
  form "Minta agent menulis", form pertanyaan, panel fakta.
  Test dengan graph stub (`tests/workflows/test_caption_questions.py`).
- [x] Graph (assistant, atas permintaan): `load` menerima `{"request_id"}`;
  `draft_caption` (draft pertama atau revisi) memanggil model dengan tool
  `CaptionDraft | AskSeller` (`tool_choice="any"`); `ask_seller` =
  `interrupt(question_request(...))`; `submit_draft` meneruskan `request_id`
  (draft pertama) atau `revision_of`, plus `inputs`. Aturan: bertanya maks
  satu putaran per versi (`asked_this_version`, di-reset saat reject), hanya
  key yang belum ada di `product_facts` (jawaban "tidak tahu" juga dihitung
  ada), maks 3 pertanyaan.
- [x] Prompt: tiga sumber (data produk, foto, fakta seller), foto hanya untuk
  yang terlihat, `null` → caption menyebut data tambahan dibutuhkan di
  `reason`; `reason` dalam Bahasa Indonesia.

### Deploy demo publik (27 Nov 2026)

- [x] Image `store` memasang `[store,agent]` lewat `uv.lock`; `fly.toml`
  (satu mesin, tanpa volume, region `sin`); `docs/deploy.md`.
- [x] Mode demo (`SELLEROPS_DEMO_MODE=1`): budget LLM per pengunjung per jam
  dan global per hari (`workflows/approval/budget.py`, HTTP 429), dipotong
  hanya sesaat sebelum LLM jalan; dua usulan contoh (`seller/demo.py`);
  banner sisa kuota di `/seller`.
- [ ] Nigka: `fly launch` + `fly secrets set ANTHROPIC_API_KEY` + `fly deploy
  --ha=false`; isi link demo di README; batas belanja bulanan di Anthropic
  Console.

### Session C4 — KPI dan laporan mingguan (workflow)

Status: [ ]

- [ ] `StateGraph` deterministik: node hitung KPI → node narasi LLM → node
  cek angka narasi terhadap KPI.

DoD: laporan mingguan deterministik dari data sintetis + narasi LLM yang
diperiksa terhadap angka.

---

## Track B — Buyer agent di atas Kala Studio (setelah Track C)

Ditunda ke sesi berikutnya (keputusan 27 Nov: target dekat sampai A5).

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

(27 Nov 2026) Target dekat: **C3 tuntas → C3b → A5**, lalu berhenti dulu.

1. **C3, tuntaskan**: satu putaran reject → revisi dengan model asli
   (berbayar, atas persetujuan Nigka).
2. **C3b**: gambar produk sebagai input caption (pipeline backend oleh
   assistant; desain prompt/node oleh Nigka).
3. **A5**: context engineering & memory (desain + `builder.py` oleh Nigka).

C4 dan Track B (buyer agent) di sesi lain. Validasi judge tetap di akhir.
