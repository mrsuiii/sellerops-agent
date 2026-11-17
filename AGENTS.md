# AGENTS.md — SellerOps Agent

Instruksi untuk AI coding agent apa pun (Claude Code, Codex, Cursor, dsb.).
Bahasa kerja: Bahasa Indonesia.

Sebelum mengubah kode, baca berurutan:

1. `HANDOFF.md` — status terakhir, pekerjaan yang belum selesai, langkah berikutnya.
2. `CLAUDE.md` — konteks proyek, keputusan stack, **kontrak belajar** (bagian
   agentic ditulis user sendiri; assistant menjelaskan, mereview, memberi hint).
3. `PLAN.md` — rencana per session dan definition of done.
4. `../../AGENTS.md` — aturan environment lab (conda `llm-post-training`,
   jelaskan sebelum install dependency, catat di `../../LEARNING_LOG.md`).

Aturan singkat: jangan baca/cetak/commit `.env`; `src/sellerops/harness/`
tidak boleh mengimpor `evaluation/` atau `simulator/`; jalankan test dengan
`~/miniforge3/envs/llm-post-training/bin/python -m pytest -q`.
