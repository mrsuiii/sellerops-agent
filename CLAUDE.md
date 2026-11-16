# CLAUDE.md — SellerOps Agent

Konteks untuk sesi Claude Code (termasuk lewat T3 Code). Ringkasan ini disusun
dari riwayat diskusi project di aplikasi Claude (6–17 November 2026). Bahasa
kerja: **Bahasa Indonesia**.

Baca juga sebelum mengubah kode: `../../TARGET.md` (sasaran, diperbarui
16 Nov 2026), `PLAN.md` (session berikutnya: **A1 — chat model, tool, dan
agent graph minimal**), `../../AGENTS.md`, bagian terbaru `../../LEARNING_LOG.md`,
dan `README.md`. Versi lama TARGET/PLAN ada di `../../archive/`.

## Tujuan

- Repo ini adalah **portofolio all-in-one** Nigka: simulator e-commerce
  deterministik, evaluation harness, SFT Qwen2.5-0.5B, dan demo toko fullstack.
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
  `graph.py`, `prompts.py`; `harness/tools/` hanya untuk tool production:
  seller di Track C, `store.py` Kala Studio di B1). Khusus eval ada di
  `evaluation/`: `sim_tools.py` (tool simulator via EpisodeService) dan
  `terminal_tools.py` (kontrak `final`/`ask_user`). Simulator
  (`SellerEnvironment`, `EpisodeService`) ada di `src/sellerops/simulator/`,
  khusus eval dan pembuatan dataset.
  `harness` tidak pernah mengimpor `evaluation`; tool dan `terminal_tools`
  diterima graph sebagai argumen, `src/sellerops/workflows/`
  (workflow), `src/sellerops/baseline/` (pipeline LangGraph lama router Qwen →
  SFT, hanya untuk reproduksi baseline 0/30; jangan dikembangkan).

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

## Status saat ini (16 Nov 2026)

- Pipeline lama: router (Qwen 0.5B) memilih satu rute/tool lalu kode jalan
  sekali, tanpa loop. E2E dev eval **0/30** karena router base-model
  mengeluarkan teks pada `act`, sehingga model aksi SFT tidak pernah dipanggil.
  SFT sendiri: schema validity 17.9% → 100%, exact action 0% → 70.5% (dev).
- Rencana: ganti agent 0.5B dengan model kuat via API; 0.5B tetap sebagai
  baseline (mungkin dapat use case sendiri nanti).
- **Kala Studio** (PR #2): demo toko fashion, FastAPI + HTML/CSS/JS polos,
  snapshot katalog DummyJSON dalam IDR, recommender BM25 di
  `POST /api/recommend`. Belum disambungkan sebagai tool agent
  `recommend_products`.
- **Production-readiness** (PR #3, #4): packaging `pyproject.toml`, Dockerfile,
  settings berbasis env (`sellerops.settings.get_settings()`), structured
  logging, hardening store API.
- **Provider LLM** (PR #5, #6): pilih lewat `SELLEROPS_LLM_PROVIDER`:
  `anthropic` (default, `ANTHROPIC_API_KEY`, `SELLEROPS_LLM_MODEL`,
  `SELLEROPS_LLM_EFFORT`) atau `openai_compatible` (NVIDIA build.nvidia.com via
  `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`). Baru konfigurasi; agent
  graph belum disusun (bagian Nigka).
- 16 Nov 2026: adapter Anthropic tulis tangan dibuat di
  `src/sellerops/harness/adapters/` (dengan test). Sejak keputusan stack
  17 Nov akan digantikan `ChatAnthropic` dan dihapus di Session A1.
- 17 Nov 2026: `src/sellerops/agent/` dipindah ke `src/sellerops/baseline/`
  (`ModelAdapter` lama → `LocalHFModel`).
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
