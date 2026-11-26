# Seller eval 20261117-155835-dev-llm-claude-haiku-5-5-0437

model `claude-haiku-5-5` (effort medium), user `llm`, split `dev`, k=3, data `seller-sim-v1`

| family | tasks | pass@1 | pass@3 | pass^1 | pass^3 |
|---|---|---|---|---|---|
| **all** | 10 | 0.83 | 0.90 | 0.83 | 0.80 |
| caption | 1 | 0.00 | 0.00 | 0.00 | 0.00 |
| restock | 1 | 1.00 | 1.00 | 1.00 | 1.00 |
| review_insight | 1 | 1.00 | 1.00 | 1.00 | 1.00 |
| sales_drop | 1 | 1.00 | 1.00 | 1.00 | 1.00 |
| sales_spike | 1 | 1.00 | 1.00 | 1.00 | 1.00 |
| sales_summary | 2 | 1.00 | 1.00 | 1.00 | 1.00 |
| slow_movers | 1 | 0.33 | 1.00 | 0.33 | 0.00 |
| top_category | 1 | 1.00 | 1.00 | 1.00 | 1.00 |
| top_products | 1 | 1.00 | 1.00 | 1.00 | 1.00 |

- trials 30, crashed 0
- failed checks: {'caption_flow': 3, 'answer': 2}
- stop reasons: {'final': 30}
- latency per conversation p50 5.97s, p95 8.67s
- avg tool calls 1.60, avg agent turns 1.30
- asked clarification when needed: 9/12
- agent tokens per trial: input 10731.2, output 960.1, cache read 8307.6, cache write 0.0
- judge `nvidia/nemotron-3-ultra-550b-a55b` on verifier passes: 24/25 pass, 0 judge errors
