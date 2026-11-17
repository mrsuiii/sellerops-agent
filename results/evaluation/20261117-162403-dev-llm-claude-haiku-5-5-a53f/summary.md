# Seller eval 20261117-162403-dev-llm-claude-haiku-5-5-a53f

model `claude-haiku-5-5` (effort medium), user `llm`, split `dev`, k=1, data `seller-sim-v1`

| family | tasks | pass@1 | pass^1 |
|---|---|---|---|
| **all** | 1 | 0.00 | 0.00 |
| caption | 1 | 0.00 | 0.00 |

- trials 1, crashed 0
- failed checks: {'caption_flow': 1}
- stop reasons: {'final': 1}
- latency per conversation p50 8.46s, p95 8.46s
- avg tool calls 3, avg agent turns 1
- asked clarification when needed: 0/1
- agent tokens per trial: input 16123.0, output 1338.0, cache read 10913.0, cache write 0.0
