# Seller eval 20261117-163140-dev-llm-claude-haiku-5-5-1dd7

model `claude-haiku-5-5` (effort medium), user `llm`, split `dev`, k=3, data `seller-sim-v1`

| family | tasks | pass@1 | pass@3 | pass^1 | pass^3 |
|---|---|---|---|---|---|
| **all** | 1 | 1.00 | 1.00 | 1.00 | 1.00 |
| sales_drop | 1 | 1.00 | 1.00 | 1.00 | 1.00 |

- trials 3, crashed 0
- failed checks: none
- stop reasons: {'final': 3}
- latency per conversation p50 8.58s, p95 9.51s
- avg tool calls 2.33, avg agent turns 1
- asked clarification when needed: 0/0
- agent tokens per trial: input 14019.3, output 1535.0, cache read 11709.3, cache write 0.0
- judge `nvidia/nemotron-3-ultra-550b-a55b` on verifier passes: 3/3 pass, 0 judge errors
