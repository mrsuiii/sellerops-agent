# Seller eval 20261117-163424-dev-llm-nvidia_nemotron-3-ultra-550b-a55b-2e25

model `nvidia/nemotron-3-ultra-550b-a55b` (effort None), user `llm`, split `dev`, k=3, data `seller-sim-v1`

| family | tasks | pass@1 | pass@3 | pass^1 | pass^3 |
|---|---|---|---|---|---|
| **all** | 10 | 0.83 | 0.90 | 0.83 | 0.80 |
| caption | 1 | 0.33 | 1.00 | 0.33 | 0.00 |
| restock | 1 | 1.00 | 1.00 | 1.00 | 1.00 |
| review_insight | 1 | 0.00 | 0.00 | 0.00 | 0.00 |
| sales_drop | 1 | 1.00 | 1.00 | 1.00 | 1.00 |
| sales_spike | 1 | 1.00 | 1.00 | 1.00 | 1.00 |
| sales_summary | 2 | 1.00 | 1.00 | 1.00 | 1.00 |
| slow_movers | 1 | 1.00 | 1.00 | 1.00 | 1.00 |
| top_category | 1 | 1.00 | 1.00 | 1.00 | 1.00 |
| top_products | 1 | 1.00 | 1.00 | 1.00 | 1.00 |

- trials 30, crashed 0
- failed checks: {'answer': 3, 'finished': 2}
- stop reasons: {'final': 28, 'no_tool_call': 2}
- latency per conversation p50 18.96s, p95 48.76s
- avg tool calls 1.17, avg agent turns 1.57
- asked clarification when needed: 12/12
- agent tokens per trial: input 9150.6, output 839.3, cache read 146.1, cache write 0.0
- judge `nvidia/nemotron-3-ultra-550b-a55b` on verifier passes: 24/25 pass, 0 judge errors
