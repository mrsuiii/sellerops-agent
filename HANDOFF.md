# HANDOFF — SellerOps Agent (17 November 2026)

Catatan serah-terima antar sesi / antar ADE. Baca ini dulu, lalu `CLAUDE.md`
(konteks + kontrak belajar) dan `PLAN.md` (rencana per session). Bahasa kerja:
Bahasa Indonesia.

## 1. Ringkasan sesi 16–17 Nov 2026

**Keputusan arsitektur**

- Agent dibangun di **LangGraph `StateGraph` yang disusun sendiri, tanpa
  `create_agent`** dan tanpa middleware LangChain. Komponen framework lain
  dipakai (`ChatAnthropic`, `@tool`, `add_messages`, `RetryPolicy`,
  checkpointer, `interrupt`).
- Model: Claude API lewat `ChatAnthropic` (`claude-opus-5-5`, effort dari
  `SELLEROPS_LLM_EFFORT`). Adapter Anthropic tulis tangan (16 Nov) sudah dihapus.
- **Pemisahan lapisan** (arah dependensi satu arah):
  - `harness/` = agent, dipakai eval + production. Tidak boleh mengimpor
    `evaluation/` atau `simulator/`. Tool, `terminal_tools`, dan
    `system_prompt` dioper sebagai argumen ke `build_agent_graph`.
  - `evaluation/` = khusus eval: `sim_tools.py`, `terminal_tools.py`
    (`final`/`ask_user` + protokol `to_action()`), `prompts.py`
    (`SIM_SYSTEM_PROMPT`), verifier, runner.
  - `simulator/` = dunia simulator lama (`SellerEnvironment`, `EpisodeService`).
- Tool production nanti lewat **service layer / API** toko (bukan menulis
  data langsung); rencana antarmuka `StoreClient` (in-process untuk test, HTTP
  untuk production). `store/api.py` tetap dipakai frontend.
- **Urutan track: 0 → A → C (seller) → B (buyer) → D → E.** Spec seller agent
  (https://claude.ai/artifact/Wond21GKwf7ui7xdgcSHSa): dua pertanyaan terbuka
  (provider, harness) sudah ditandai selesai di dokumen.

**Perubahan kode / repo**

- Commit `1d09892` (main): `agent/` → `baseline/` (`ModelAdapter` →
  `LocalHFModel`), scaffold `harness/`, `evaluation/sim_tools.py` +
  `terminal_tools.py`, paket `simulator/`, `langchain-anthropic` di extra
  `agent`, test baru.
- **Belum di-commit** (cek `git status`):
  - `harness/prompts.py` → `evaluation/prompts.py` (`SIM_SYSTEM_PROMPT`);
    `system_prompt` kini wajib di `build_agent_graph`.
  - Catatan multi-turn di `PLAN.md` (awal Track C), update `CLAUDE.md`, README.
  - Artefak fine-tune 0.5B (55 file tracked + bobot 1.9 GB yang gitignored)
    dipindah ke `../../archive/sellerops-sft-0.5b/` (ada README di sana):
    `results/`, `artifacts/`, `notebooks/`, `data/sft/`, `configs/`,
    `docs/sft-training.md`.
  - File ini, `AGENTS.md`.
  - `uv.lock` sudah untracked sejak sebelum sesi; belum diputuskan.

**Environment** (conda `llm-post-training`)

- Terpasang: `langchain-anthropic` 1.7.10 (ikut: `anthropic` 2.9.0,
  `langchain-core` 1.8.1, `langsmith` 0.14.17), `langchain` 1.4.2,
  `langgraph` 1.2.12. Belum terpasang: `langchain-openai`,
  `langgraph-checkpoint-sqlite`, `fastapi` (karena itu 2 test store gagal).
- `ruff` tidak ada di env conda; dijalankan lewat
  `uv run --isolated --no-project --with ruff ruff check src scripts tests`.

**Status test** (`~/miniforge3/envs/llm-post-training/bin/python -m pytest -q`):
192 lolos, 8 skip (test graph A1, menunggu implementasi), 2 gagal di
`tests/store` (fastapi tidak terpasang; sudah gagal sebelum sesi).

## 2. Arahan terbaru user yang BELUM dikerjakan

User (17 Nov): **"lupakan baseline 0.5B, hapus simulator lama, buat simulator
baru jadi multi-turn."** Artefak fine-tune sudah dipindah (lihat atas); sisa
pekerjaan di bawah belum dimulai. Tiga keputusan diajukan ke user tapi belum
dijawab eksplisit; usulan default assistant:

1. **Hapus semua kode 0.5B + simulator lama**: `src/sellerops/baseline/`,
   `simulator/`, `dataset.py`, `schemas.py` (bagian lama), `data/tasks/`,
   `data/environment/`, `data/evaluation/`, `data/manifest.json`,
   `data/DATASET_CARD.md`, `scripts/generate_sellerops_dataset.py`,
   `scripts/evaluate_agent.py`, `evaluation/{verifier,answer_keys,dataset,
   target,evaluators,sim_tools,terminal_tools,prompts,telemetry}.py`,
   `AGENT_WALKTHROUGH.md`, `docs/episode-replay-and-verification.md`, extra
   `local-model` (torch), test terkait. Pertahankan yang generik:
   `evaluation/metrics.py` (pass@k), `judge.py` (perlu rubrik baru), pola
   runner LangSmith.
2. **Domain simulator baru = seller agent** (sesuai spec, sekaligus Session
   C1): katalog Kala Studio (`data/store/fashion_catalog.json`, 54 produk,
   review asli) + order sintetis 90 hari dengan seed tetap dan skenario
   tertanam (lonjakan, penurunan, stok habis, review negatif berulang). Tool
   eval/production: `get_sales_summary`, `get_product_performance`,
   `get_inventory_status`, `get_reviews`, `get_product`,
   `propose_caption_update`.
3. **Eval multi-turn** (pola τ-bench): task berisi info tersembunyi user,
   persona, tujuan; `UserSimulator` dengan dua implementasi — `ScriptedUser`
   (deterministik, untuk test tanpa network) dan `LLMUser` (Claude, untuk
   eval); runner giliran agent → user → agent dengan checkpointer +
   `thread_id`; `ask_user` mengakhiri giliran, bukan episode; verifier
   menilai hasil akhir percakapan (+ jumlah giliran, pass^k).

Konsekuensi: `tests/harness/test_graph.py` saat ini memakai simulator lama;
harus ditulis ulang di atas simulator baru (perilaku yang dites tetap sama:
happy path, ask_user, teks tanpa tool, batas langkah, final invalid, exception
tool, mode chat tanpa terminal tools). Session A3 "vs baseline" diganti
menjadi eval agent di simulator seller.

## 3. Langkah berikutnya (urut)

1. Konfirmasi 3 keputusan di atas ke user (atau pakai default bila user
   bilang lanjut).
2. Commit perubahan yang belum di-commit (user mengizinkan commit langsung di
   `main`; jangan sertakan `uv.lock` tanpa ditanya).
3. Hapus kode 0.5B + simulator lama; perbaiki import/test/README/PLAN.
4. Bangun simulator seller (C1): generator order sintetis (seed, manifest,
   test pola), read model penjualan/stok/review, tool eval.
5. Task eval multi-turn + `UserSimulator` + runner giliran + verifier seller.
6. Tulis ulang test graph A1 di atas simulator baru; lalu **Nigka**
   mengerjakan TODO A1 di `harness/state.py` dan `harness/graph.py`
   (A1-1 … A1-5). Assistant hanya review + hint (kontrak belajar).

## 4. Hal yang perlu diingat

- Kontrak belajar: bagian agentic (state, node, edge, routing, stop condition,
  retry, approval) ditulis Nigka sendiri; assistant boleh scaffold, tool,
  simulator, test, dan plumbing.
- `harness/` tidak boleh mengimpor `evaluation/` atau `simulator/`.
- Opus 5.5 menolak forced `tool_choice`; graph harus menangani balasan tanpa
  tool call. Server-side refusal fallback belum dipasang (rencana
  `with_fallbacks` di A2).
- Simpan `task_id`, seed, model id, token, latency di setiap eval run.
- Jangan baca / cetak / commit `.env`.
