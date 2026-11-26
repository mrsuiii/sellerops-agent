# CLAUDE.md — SellerOps Agent

Konteks untuk sesi Claude Code (termasuk lewat T3 Code). Ringkasan ini disusun
dari riwayat diskusi project di aplikasi Claude (6–17 November 2026). Bahasa
kerja: **Bahasa Indonesia**.

Baca dulu **`HANDOFF.md`** (status terakhir + langkah berikutnya), lalu
`../../TARGET.md` (sasaran, diperbarui
16 Nov 2026), `PLAN.md` (session berikutnya: **A1 — chat model, tool, dan
agent graph minimal**), `../../AGENTS.md`, bagian terbaru `../../LEARNING_LOG.md`,
dan `README.md`. Versi lama TARGET/PLAN ada di `../../archive/`.

## Tujuan

- Repo ini adalah **portofolio all-in-one** Nigka: agent seller/buyer di atas
  LangGraph + Claude API, simulator e-commerce multi-turn, evaluation harness,
  dan demo toko fullstack (Kala Studio). Jalur SFT Qwen 0.5B sudah diarsipkan.
- Nigka ingin **belajar membangun agent yang robust** dengan menyusunnya sendiri
  di atas LangGraph/LangChain.

## Stack agent (keputusan 17 Nov 2026)

- **Semua agent dan workflow memakai LangGraph / LangChain v1, tanpa
  `create_agent`.** Isi `create_agent` disusun sendiri sebagai `StateGraph`:
  node `model` → conditional edge → node `tools` → `model` …, dengan state
  `add_messages`, counter langkah, dan `final_action`. Middleware LangChain
  tidak dipakai (hanya berlaku di `create_agent`); padanannya ditulis sebagai
  node / edge / field state. `ToolNode` dan source `create_agent` boleh dibaca
  sebagai referensi.
- Komponen framework lain dipakai bila ada: `RetryPolicy` pada node,
  `with_fallbacks`, `interrupt()` + `Command(resume=...)`, `Store`, streaming.
- Jawaban akhir `final` / `ask_user` berupa tool dengan skema Pydantic; edge
  mengakhiri graph saat tool itu dipanggil.
- Model default: `claude-haiku-5-5`, effort `medium` (keputusan 17 Nov;
  ganti lewat `SELLEROPS_LLM_MODEL` / `SELLEROPS_LLM_EFFORT`).
- Model: `init_chat_model` → `ChatAnthropic` (`langchain-anthropic`) atau
  `ChatOpenAI` dengan `base_url` untuk NVIDIA (`langchain-openai`). Jangan
  memanggil SDK `anthropic` / `openai` langsung dari kode agent.
- Tool: `@tool` dari `langchain_core` + `bind_tools`, dieksekusi node
  `tools` sendiri.
- Workflow pasti (approval caption, KPI): `StateGraph` custom + `interrupt()`.
- State, resume, replay: checkpointer (`InMemorySaver` di test, `SqliteSaver`
  lokal) dan `get_state_history`; tracing dan eval lewat LangSmith.
- Sebelum memakai parameter `ChatAnthropic` (effort, caching, fallback) atau
  API LangGraph, cek dokumentasi versi yang terpasang.
- Lokasi kode: `src/sellerops/harness/` (agent: `models.py`, `state.py`,
  `graph.py`; `harness/tools/` untuk tool production: `seller.py` = enam
  tool seller di atas Protocol `SellerDataClient`, `store.py` Kala Studio di
  B1). Khusus eval ada di `evaluation/`: `seller_tools.py` (tool yang sama,
  client = `SellerEpisode`),
  `terminal_tools.py` (kontrak `final`/`ask_user`), `prompts.py`
  (`SELLER_SYSTEM_PROMPT`), `seller_tasks.py`, `user_sim.py`
  (`ScriptedUser`/`LLMUser`), `conversation.py` (loop giliran),
  `seller_verifier.py`, `metrics.py`. Harness tidak punya prompt default:
  `system_prompt` wajib dioper ke `build_agent_graph`. Simulator seller
  (`generator.py`, `store.py`, `episode.py`) ada di `src/sellerops/simulator/`,
  khusus eval; datanya di `data/seller_sim/` (`make sim`).
  Backend seller production ada di `src/sellerops/seller/` (`analytics.py`
  dipakai bersama simulator, `db.py` SQLite, `service.py` `SellerService` +
  `InProcessSellerClient`, `registry.py` `SellerServices.for_store`, `api.py` `/api/seller`); `seller` tidak mengimpor
  `harness`, `evaluation`, atau `simulator`.
  `harness` tidak pernah mengimpor `evaluation` atau `simulator`; tool dan
  `terminal_tools` diterima graph sebagai argumen. Workflow di
  `src/sellerops/workflows/`.

## Kontrak belajar (penting)

- Bagian agentic (konfigurasi model, state, graph/node/edge, routing,
  stop condition, retry, approval, orkestrasi tool) **disusun Nigka sendiri
  dengan primitif LangGraph** (`StateGraph`, node, edge), bukan
  `create_agent`. Jangan menulis
  implementasinya kecuali diminta eksplisit. Jelaskan konsep, review, dan beri
  hint bertahap (lihat `PLAN.md`).
- Saat memakai fitur framework, jelaskan juga apa yang dikerjakannya di balik
  layar (mis. reducer `add_messages`, `recursion_limit`, `RetryPolicy`), karena ini bahan
  belajar.
- Jelaskan alasan setiap keputusan desain.

## Arah produk

Dua agent di satu marketplace, **tanpa negosiasi harga**:

- **Buyer agent**: asisten belanja (search, recommend, get_product,
  add_to_cart), hanya memakai data publik.
- **Seller agent**: analisis penjualan dan saran untuk seller.
  - MVP: ringkasan penjualan, produk top/lambat, stok + restock, insight
    review, dan **perbaikan caption dengan approval manusia** (fitur utama:
    propose → approve → backend PATCH → audit log → bisa revert).
  - Lanjutan: dashboard + analyst agent dengan `run_code` (python/sql/bash)
    yang di-sandbox di Docker.
  - Butuh generator data order sintetis lebih dulu.
- **Hybrid**: workflow predefined untuk alur pasti yang perlu diaudit
  (approval caption, KPI, laporan mingguan); agent loop untuk yang open-ended
  (buyer agent, analisis bebas). Aturan praktis: kalau flowchart-nya bisa
  digambar sebelum user bertanya, jadikan workflow.
- Rencana lanjutan: SellerOps jadi agent lebih umum dengan RAG atas katalog
  produk.
- Rename repo ditunda sampai buyer agent ada.

Dokumen terkait (artifact di aplikasi Claude):
- Spec seller agent: https://claude.ai/artifact/Wond21GKwf7ui7xdgcSHSa
- Plan agent umum + RAG produk: https://claude.ai/artifact/77zTgEAD9uVorNcST3zVDj

## Status saat ini (17 Nov 2026)

Detail lengkap dan langkah berikutnya: **`HANDOFF.md`**.

- Pipeline lama Qwen 0.5B (router → SFT, e2e dev 0/30) dan simulator
  offer/diskon lama **sudah dihapus** dari repo (17 Nov); artefak di
  `../../archive/sellerops-sft-0.5b/`, kode di git history.
- **Simulator seller multi-turn** (Session C1) selesai: data sintetis 90 hari
  dengan skenario tertanam, read model, 19 task (10 dev, 9 test frozen),
  `ScriptedUser`/`LLMUser`, runner giliran, verifier. Lihat
  `src/sellerops/evaluation/README.md`.
- **Track A selesai** (A1–A4, diimplementasi assistant atas permintaan
  eksplisit Nigka, 17 Nov): `harness/graph.py` (retry, fallback, budget,
  error terstruktur, runtime context `AgentContext`, node `summarize`),
  `harness/errors.py`, `harness/trajectory.py` (SQLite, ekspor, replay).
  Eval dev Haiku 5.5: pass@1 = pass^3 = 1.00 setelah error analysis
  (`docs/error-analysis-2026-11-17.md`). Kontrak belajar tetap berlaku untuk
  graph Track C (approval caption C3, KPI C4).
- **Backend seller C2 + C3** selesai (18 Nov, assistant): SQLite, state
  machine caption (approve = apply atomik, stale, reject + revisi maks 5,
  revert sebagai versi baru), audit log append-only, `/api/seller`. Graph
  approval C3 (`workflows/approval/`) selesai 21 Nov, ditulis assistant atas
  permintaan eksplisit Nigka; di API lewat `workflows/approval/api.py`
  (`/caption-proposals/{id}/approval`, 27 Nov).
- **Kala Studio** (PR #2): FastAPI + HTML/CSS/JS, katalog DummyJSON dalam IDR,
  recommender BM25 di `POST /api/recommend`; keranjang masih di browser
  (`localStorage`). Belum ada tool agent.
- **Production-readiness** (PR #3, #4): `pyproject.toml`, Dockerfile, settings
  env (`sellerops.settings.get_settings()`), structured logging.
- **Provider LLM** (PR #5, #6): `SELLEROPS_LLM_PROVIDER` = `anthropic`
  (default) atau `openai_compatible` (NVIDIA; `langchain-openai` terpasang, dipakai judge eval;
  adapter agent belum).
- Kalau memakai Claude: aktifkan prompt caching untuk system prompt dan tools
  (`cache_control` lewat `ChatAnthropic`), dan cek `cache_read_input_tokens` di
  `usage_metadata`.

## Konvensi kerja

- **Tanpa GitHub Actions CI** (akun billing-locked). Cek lokal dengan
  `make lint test` atau `pre-commit`.
- Nama branch harus bermakna (mis. `feat/llm-provider-switch`).
- Nigka lebih suka perubahan langsung di folder lokal daripada lewat PR,
  kecuali diminta membuat PR.
- Jangan pernah membaca, mencetak, atau meng-commit isi `.env` atau API key.

## Perintah umum

```bash
make test    # pytest
make lint    # ruff
make store   # Kala Studio di http://127.0.0.1:8000 (docs di /docs)
uv sync --extra agent   # atau: pip install -e ".[agent]"
```

## Bacaan agent harness (referensi belajar)

Anthropic *Building effective agents*, *Effective harnesses for long-running
agents*, *Effective context engineering*, *Writing effective tools for agents*;
HumanLayer *12-Factor Agents*; Thorsten Ball *How to Build an Agent*;
mini-swe-agent; paper SWE-agent, τ-bench/τ²-bench (metrik pass^k), ReAct;
tulisan Hamel Husain tentang evals; dokumentasi LangGraph (graph API,
persistence, interrupts, time travel) dan source `create_agent` /
`ToolNode` sebagai referensi pembanding.
