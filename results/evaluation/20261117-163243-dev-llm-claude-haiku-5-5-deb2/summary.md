# Seller eval 20261117-163243-dev-llm-claude-haiku-5-5-deb2

model `claude-haiku-5-5` (effort medium), user `llm`, split `dev`, k=3, data `seller-sim-v1`

| family | tasks | pass@1 | pass@3 | pass^1 | pass^3 |
|---|---|---|---|---|---|
| **all** | 10 | 1.00 | 1.00 | 1.00 | 1.00 |
| caption | 1 | 1.00 | 1.00 | 1.00 | 1.00 |
| restock | 1 | 1.00 | 1.00 | 1.00 | 1.00 |
| review_insight | 1 | 1.00 | 1.00 | 1.00 | 1.00 |
| sales_drop | 1 | 1.00 | 1.00 | 1.00 | 1.00 |
| sales_spike | 1 | 1.00 | 1.00 | 1.00 | 1.00 |
| sales_summary | 2 | 1.00 | 1.00 | 1.00 | 1.00 |
| slow_movers | 1 | 1.00 | 1.00 | 1.00 | 1.00 |
| top_category | 1 | 1.00 | 1.00 | 1.00 | 1.00 |
| top_products | 1 | 1.00 | 1.00 | 1.00 | 1.00 |

- trials 30, crashed 0
- failed checks: none
- stop reasons: {'final': 30}
- latency per conversation p50 5.89s, p95 10.11s
- avg tool calls 1.40, avg agent turns 1.40
- asked clarification when needed: 12/12
- agent tokens per trial: input 11497.1, output 867.6, cache read 10059.9, cache write 0.0
- judge `nvidia/nemotron-3-ultra-550b-a55b` on verifier passes: 29/30 pass, 0 judge errors
