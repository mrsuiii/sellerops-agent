# HANDOFF — SellerOps Agent (18 November 2026, sesi 4)

Catatan serah-terima antar sesi / antar ADE. Baca ini dulu, lalu `CLAUDE.md`
(konteks + kontrak belajar) dan `PLAN.md` (rencana per session). Bahasa kerja:
Bahasa Indonesia.

## 0. Sesi 4 (18 Nov): backend seller C2 + C3 (assistant, atas permintaan Nigka)

Branch `feat/c2-c3-backend` (dari `feat/seller-agents`; belum di-push).

- `src/sellerops/seller/analytics.py`: analitik dipindah dari
  `simulator/store.py`; simulator kini subclass `SellerAnalytics`, jadi kunci
  jawaban eval dan angka production berasal dari kode yang sama (test
  paritas `tests/seller/test_analytics_parity.py`).
- `seller/db.py`: SQLite (`SELLEROPS_SELLER_DB`, default `var/seller.sqlite`,
  di-seed otomatis dari `data/seller_sim`). Trigger: `audit_log` dan
  `caption_versions` append-only; caption produk hanya bisa berubah bersama
  baris versi yang cocok.
- `seller/service.py`: `SellerService` (analitik + state machine caption +
  audit log) dan `InProcessSellerClient` (client production untuk
  `make_seller_tools`; tanpa approve).
- `seller/api.py`: `/api/seller/*`, dipasang di `store/api.py`. Header
  `X-Seller-User` = actor; **belum ada auth**.
- Docker store: menyalin `data/seller_sim`, DB di volume `seller-db`.
- Seam multi-toko (21 Nov, tanpa membuat multi-tenant): `SellerService`
  terikat ke `store_id` (DB mencatat pemiliknya di `meta`),
  `seller/registry.py` (`SellerServices.for_store`, kini
  `SingleStoreServices`), API menentukan toko dan actor hanya di
  `current_store_id` dan `actor`. `AgentContext.store_id` sudah ada di harness.
  Kalau multi-toko dibuat: implementasi `SellerServices` baru, auth mengisi
  dua dependency itu (`seller_user` / `actor`). `thread_id` approval API sudah
  diberi prefix toko (`kala-studio:caption-approval-CP-0001`).
- Graph approval C3 (21 Nov): `workflows/approval/approval_graph.py`
  (load → await_decision/interrupt → apply | cancel_flow | record_rejection
  → draft_revision (LLM, satu-satunya dengan RetryPolicy) → submit_revision),
  prompt di `approval_prompts.py`. Node berefek samping idempoten; jawaban
  resume tidak valid → tanya ulang. 18 test di `tests/workflows/`.
- Coba sendiri: `python scripts/try_caption_approval.py [--fake]` (jadi
  seller di terminal; `q` lalu `--resume CP-xxxx`), dan integration test
  model asli `make test-live` (`tests/integration/`, marker `live`,
  dilewati tanpa `SELLEROPS_LIVE_TESTS=1`).
- LangGraph Studio: `uv sync --extra dev --extra studio`, lalu `make studio`
  (atau `make studio-fake`). Graph di `langgraph.json`:
  `caption_approval_demo` (node `agent_proposes` + graph approval sebagai
  subgraph; input deskripsi) dan `caption_approval` (input `proposal_id`).
  Entry point: `workflows/approval/studio.py`; data di `var/studio/`.
- Graph approval di API (27 Nov): `workflows/approval/runner.py`
  (`ApprovalRunner`: proposal id → thread rantai, thread dimulai saat pertama
  dibaca/diputuskan, lock per thread, keputusan hanya untuk versi yang sedang
  ditunggu → `stale_decision`, node gagal → `state="error"` + retry) dan
  `workflows/approval/api.py`: `GET …/{id}/approval`,
  `POST …/{id}/approval/decision`, `POST …/{id}/approval/retry`. Checkpoint di
  `var/approvals.sqlite` (`SELLEROPS_APPROVALS_DB`). Dipasang di app store
  hanya jika extra `[agent]` ada (image Docker store belum). Route
  approve/reject/cancel lama tetap ada sebagai panggilan service tanpa graph.
  6 test di `tests/workflows/test_approval_api.py`.
- Halaman seller `/seller` (27 Nov, `web/seller.*`): antrean usulan, diff
  lama/baru, approve / reject (+ add_fields) / cancel lewat route approval,
  revert, audit log. Toko kini menampilkan caption dari DB seller
  (`shop_catalog` di `store/api.py`, indeks BM25 dibangun ulang hanya saat
  caption berubah). `make store-fake` = tanpa biaya LLM.
- C3b bagian assistant (27 Nov, branch `feat/c3b-images`): pipeline gambar
  `seller/images.py`, `supports_vision`, `inputs` di proposal, gambar yang
  dilihat model di `/seller`. Bagian 2 (agent bertanya ke seller): caption
  request `CR-…`, memori `product_facts`, kontrak `workflows/approval/questions.py`,
  runner/API `waiting_for` + `.../approval/answer`, UI pertanyaan di `/seller`.
  Sisa (Nigka): graph (input `request_id`, draft pertama, `AskSeller`,
  `ask_seller` interrupt, gambar di draft) dan prompt tiga sumber. Sampai
  graph mendukung `request_id`, `POST /caption-requests` berakhir
  `state="error"` di `load` (lihat `PLAN.md` C3b bagian 2).
- 285 test lolos (+4 live di-skip), ruff bersih.

Berikutnya (27 Nov; PR #7 sudah di-merge ke `main`): target dekat sampai
A5. (1) coba reject → revisi dengan model asli (`make test-live` /
`make try-approval`; berbayar), (2) **C3b** gambar produk sebagai input
caption (`PLAN.md` C3b), (3) **A5 context & memory** (episodic/semantic/
procedural, folder `harness/context/`; rencana di `PLAN.md` A5). C4 dan
buyer agent (Track B) di sesi lain. Validasi judge ditunda ke akhir (keputusan Nigka 18 Nov).

## 1. Yang selesai di sesi 3

Arahan user 17 Nov ("lupakan baseline 0.5B, hapus simulator lama, buat
simulator baru jadi multi-turn") dikerjakan dengan tiga default yang diusulkan
sesi sebelumnya (user: "lanjutkan"):

1. **Kode 0.5B + simulator lama dihapus**: `baseline/`, simulator
   offer/diskon (`SellerEnvironment`, `EpisodeService`), `dataset.py`, bagian
   lama `schemas.py`, `data/{tasks,environment,evaluation}`, manifest/dataset
   card lama, `scripts/{generate_sellerops_dataset,evaluate_agent}.py`, modul
   eval lama (`verifier`, `answer_keys`, `dataset`, `target`, `evaluators`,
   `sim_tools`, `telemetry`, `runner`), `AGENT_WALKTHROUGH.md`,
   `docs/episode-replay-and-verification.md`, extra `local-model`, test
   terkait. Dipertahankan: `metrics.py` (+ `pass_hat_metrics` untuk pass^k),
   `judge.py` (sekarang: judge `message` via NVIDIA, lihat bawah).
2. **Simulator seller (Session C1)** di `src/sellerops/simulator/`:
   `generator.py` (seed `20261117`, 90 hari s.d. `AS_OF=2026-11-15`, skenario
   tertanam: spike FS163, stockout FS186, low stock FS180, slow movers
   FS174/FS157, keluhan "ukuran kekecilan" FS188), `store.py` (read model +
   proposal caption yang tidak mengubah produk), `episode.py`
   (`SellerEpisode`). Data + checksum di `data/seller_sim/` (`make sim`).
3. **Eval multi-turn** di `src/sellerops/evaluation/`: `seller_tools.py`
   (enam tool dari `harness/tools/seller.py`, client = `SellerEpisode`), `terminal_tools.py` (`final(message, data)`,
   `ask_user(question)`), `prompts.py` (`SELLER_SYSTEM_PROMPT`),
   `seller_tasks.py` (10 dev + 9 test frozen, kunci jawaban dihitung dari
   data), `user_sim.py` (`ScriptedUser`, `LLMUser`), `conversation.py`
   (`run_conversation`: `ask_user` → balasan sebagai `ToolMessage` untuk call
   itu → invoke lagi di `thread_id` yang sama), `seller_verifier.py`
   (`finished`, `answer`, `grounded`, `caption_flow`).
4. **Test graph A1 ditulis ulang** di atas simulator seller
   (`tests/harness/test_graph.py`, 9 test, + tes multi-turn dengan
   `ScriptedUser` dan `InMemorySaver`). Sudah dicek bisa dipenuhi dengan
   implementasi graph sementara di scratchpad (tidak di-commit, supaya TODO
   A1 tetap untuk Nigka).

Keputusan kecil yang diambil (bisa diubah): target restock 14 hari
(`target_days`), "restock segera" = habis ≤ 7 hari; review negatif = rating
≤ 2; `final.data` berisi fakta kunci terstruktur per family (lihat docstring
`FinalAnswer`).

**Status test** (`~/miniforge3/envs/llm-post-training/bin/python -m pytest -q`):
82 lolos, 9 skip (test graph A1) sebelum graph diimplementasi; sesudahnya
94 lolos, 0 skip; 2 gagal di `tests/store` (fastapi belum
terpasang di env conda; sudah gagal sebelum sesi). Ruff bersih
(`uv run --isolated --no-project --with ruff ruff check src scripts tests`).

**Update (akhir sesi):** atas permintaan eksplisit user ("implementasi yang
graph.py dong"), assistant mengimplementasikan `harness/state.py` dan
`harness/graph.py`. Desain: node `start_turn` mereset `steps` /
`final_action` / `stop_reason` setiap invoke (budget step per giliran); step =
eksekusi tool (termasuk yang gagal); terminal tool hanya berlaku bila
dipanggil sendirian dengan argumen valid, selain itu jadi
`ToolMessage(status="error")` supaya model bisa memperbaiki; teks tanpa tool
call → `no_tool_call` (eval) atau `answer` (chat). 94 test lolos, 0 skip.

## 2. Status dan langkah berikutnya

**Track A selesai** (A1–A4; Nigka meminta assistant mengerjakan semuanya):

- A2 robustness: `harness/errors.py` (peta error), `RetryPolicy` + SDK retry,
  `with_fallbacks` (`SELLEROPS_LLM_FALLBACK_MODELS`), budget token/waktu per
  giliran, error tool berkode, `repair_tool_pairs`, provider NVIDIA untuk
  agent, prompt caching otomatis hanya untuk Claude.
- Runtime context: `AgentContext` (`user_id`, `store_id`) → `ToolRuntime`.
- A3 eval: error analysis `docs/error-analysis-2026-11-17.md` (5 akar
  masalah). Dev Haiku 5.5 (`LLMUser`, k=3): pass@1 0.83 → **1.00**, pass^3
  0.80 → **1.00**; judge 29/30. Pembanding Nemotron 3 Ultra: 0.83 / 0.80.
  Perbaikan: aturan prompt (field caption, klarifikasi lengkap, tren dari
  tool), `compare_to_previous` di tool, validator bentuk `final.data`.
- A4: node `summarize`, `SqliteSaver`, ekspor trajectory, replay
  (`scripts/replay_eval_thread.py`).

Sisa / berikutnya:

1. **Butuh Nigka:** labeli 25–50 percakapan untuk validasi judge
   (`python scripts/judge_labels.py export` → isi `human_*` → `score`).
2. Angka split test sekali untuk laporan (`--allow-test`), jangan untuk tuning.
3. Temuan terbuka: prosa `message` kadang melebih-lebihkan agregat (3 kasus
   judge); `find_product(name)` untuk model yang lebih lemah; memory `Store`
   (opsional A4).
4. Track C: backend C3 (DB proposal/versi/audit + service, desain di diskusi
   17 Nov, keputusan di PLAN C3) lalu graph approval (bagian Nigka); service
   layer + client production C2.

Bug penting yang ditemukan eval: key `run_id` di `metadata` config membuat
giliran kedua dst. no-op di LangGraph. Runner memakai `eval_run_id`.

## 3. Hal yang perlu diingat

- Kontrak belajar: bagian agentic (state, node, edge, routing, stop condition,
  retry, approval) ditulis Nigka sendiri; assistant boleh scaffold, tool,
  simulator, test, dan plumbing.
- `harness/` tidak boleh mengimpor `evaluation/` atau `simulator/`.
- Split `test` di `data/seller_sim/tasks.test.jsonl` **frozen**: jangan dipakai
  untuk tuning prompt atau graph.
- Model default sejak 17 Nov: **`claude-haiku-5-5`, effort `medium`**
  (`DEFAULT_LLM_MODEL` di `settings.py`; ganti lewat `SELLEROPS_LLM_MODEL`).
  Dua dev task lolos di Haiku dan Opus 5.5. Opus 5.5 menolak forced
  `tool_choice`; graph tidak memakainya dan menangani balasan tanpa tool call.
- Jangan baca / cetak / commit `.env`.
