# HANDOFF — SellerOps Agent (17 November 2026, sesi 2)

Catatan serah-terima antar sesi / antar ADE. Baca ini dulu, lalu `CLAUDE.md`
(konteks + kontrak belajar) dan `PLAN.md` (rencana per session). Bahasa kerja:
Bahasa Indonesia.

## 1. Yang selesai di sesi ini

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
   `judge.py` (Gemini; rubrik masih lama, `with_judge` dihapus).
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

## 2. Langkah berikutnya

1. A1 terverifikasi dengan API sungguhan: 2 dev task lolos (termasuk
   multi-turn), prompt cache terbaca. Worktree ini tidak punya `.env`; uji
   memuat `.env` dari checkout utama lewat `load_dotenv` di script scratchpad.
   Temuan: angka turunan di `message` (persentase) dihitung model sendiri.
2. A2 (robustness: `RetryPolicy`, `with_fallbacks`, timeout) dan A3 (CLI eval
   multi-turn, `LLMUser`, pass^k).
3. Opsional: rubrik judge baru untuk `message` (analis seller), endpoint HTTP
   seller + `StoreClient` (C2/C3), keputusan `uv.lock` (masih untracked).

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
