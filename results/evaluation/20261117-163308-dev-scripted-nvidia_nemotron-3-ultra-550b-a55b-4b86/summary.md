# Seller eval 20261117-163308-dev-scripted-nvidia_nemotron-3-ultra-550b-a55b-4b86

model `nvidia/nemotron-3-ultra-550b-a55b` (effort None), user `scripted`, split `dev`, k=1, data `seller-sim-v1`

| family | tasks | pass@1 | pass^1 |
|---|---|---|---|
| **all** | 10 | 0.90 | 0.90 |
| caption | 1 | 1.00 | 1.00 |
| restock | 1 | 1.00 | 1.00 |
| review_insight | 1 | 0.00 | 0.00 |
| sales_drop | 1 | 1.00 | 1.00 |
| sales_spike | 1 | 1.00 | 1.00 |
| sales_summary | 2 | 1.00 | 1.00 |
| slow_movers | 1 | 1.00 | 1.00 |
| top_category | 1 | 1.00 | 1.00 |
| top_products | 1 | 1.00 | 1.00 |

- trials 10, crashed 0
- failed checks: {'answer': 1}
- stop reasons: {'final': 10}
- latency per conversation p50 18.38s, p95 29.77s
- avg tool calls 1.30, avg agent turns 1.40
- asked clarification when needed: 4/4
- agent tokens per trial: input 8913.3, output 816.9, cache read 0.0, cache write 0.0
