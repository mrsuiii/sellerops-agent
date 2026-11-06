# Evaluasi penuh task dev — 6 Oktober 2026

Konfigurasi: 30 task dev, satu episode/task, greedy, router qwen-base, action sellerops-sft; offline tanpa judge. Max model calls 8, max retries 2, max new tokens 256.

| Kelompok task | Lolos | Total |
|---|---:|---:|
| inventory_offer | 0 | 10 |
| policy_qa | 0 | 10 |
| product_search | 0 | 10 |

- pass@1: 0/30 (0%). pass@k untuk k>1 tidak diukur karena hanya satu repetition.
- Error evaluator: 0. Runtime failures: 0.
- Seluruh episode berakhir waiting_user / missing_structured_outcome.
- Total: 60 panggilan router, 30 retry, 0 panggilan SFT, 0 tool calls, 22.215 token model.
- Router schema validity: 50%; latency p50 6,43 detik, p95 7,53 detik.

## Pola kegagalan

Pada semua 30 task, attempt pertama memilih act tetapi text tidak kosong, sehingga parsing gagal. Retry menghasilkan ask yang valid, lalu graph berhenti waiting_user. Karena tidak ada action SFT final/ask_user, verifier memberi missing_structured_outcome.

Hasil ini mengukur sistem dengan routing saat ini; belum mengukur kemampuan action model SFT karena tidak pernah dipanggil. Prioritas perbaikan adalah kepatuhan output router terhadap schema sambil mempertahankan keputusan act ketika memperbaiki format.

Gemini judge tidak diaktifkan karena key tidak tersedia di environment sesi. Frozen test tidak dievaluasi.

Artefak: [summary](summary.json), [episode logs](episodes.jsonl), [metadata](metadata.json), [router prompt saat evaluasi](router_prompt.txt).
