# Seller eval 20261117-154812-dev-scripted-claude-haiku-5-5-4121

model `claude-haiku-5-5` (effort medium), user `scripted`, split `dev`, k=1, data `seller-sim-v1`

| family | tasks | pass@1 | pass^1 |
|---|---|---|---|
| **all** | 10 | 0.80 | 0.80 |
| caption | 1 | 0.00 | 0.00 |
| restock | 1 | 1.00 | 1.00 |
| review_insight | 1 | 1.00 | 1.00 |
| sales_drop | 1 | 1.00 | 1.00 |
| sales_spike | 1 | 1.00 | 1.00 |
| sales_summary | 2 | 1.00 | 1.00 |
| slow_movers | 1 | 0.00 | 0.00 |
| top_category | 1 | 1.00 | 1.00 |
| top_products | 1 | 1.00 | 1.00 |

- trials 10, crashed 0
- failed checks: {'caption_flow': 1, 'answer': 1}
- stop reasons: {'final': 10}
- latency per conversation p50 5.28s, p95 8.69s
- avg tool calls 1.80, avg agent turns 1.30
- asked clarification when needed: 3/4
- agent tokens per trial: input 11113.9, output 939.7, cache read 8978.1, cache write 0.0
