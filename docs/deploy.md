# Deploy: public demo on Fly.io

The demo is one container (`Dockerfile`, target `store`): the Kala Studio shop,
the seller page (`/seller`) and the caption approval agent (C3/C3b) with the
real Claude model. Config: `fly.toml`.

## What the demo protects against

| Risk | Guard |
| --- | --- |
| Someone runs up the LLM bill | `SELLEROPS_DEMO_MODE=1`: at most `SELLEROPS_DEMO_LLM_PER_CLIENT_HOUR` agent steps per visitor per hour and `SELLEROPS_DEMO_LLM_PER_DAY` for everyone per UTC day (`workflows/approval/budget.py`). Only steps that run the LLM count (new caption request, answer, retry, reject); a refused request is free. Over the limit: HTTP 429 with a readable message, shown on the seller page. |
| Visitors deface the demo | No volume: the machine stops after idling and starts from the image again, with the seeded data and two demo proposals (`seller/demo.py`). |
| Key leaks | `ANTHROPIC_API_KEY` is a Fly secret; the image has no `.env` (`.dockerignore`). |
| No login | The seller name is a trusted header (`X-Seller-User`). Fine for a demo with nothing private in it; put real auth in front before using it with real store data. |

Also set a monthly spend limit in the Anthropic Console as the last line of defence.

## First deploy

```bash
brew install flyctl && fly auth login
fly launch --no-deploy --copy-config --name <your-app-name>   # creates the app, keeps fly.toml
fly secrets set ANTHROPIC_API_KEY=sk-ant-...                  # you type this; never commit it
fly deploy --ha=false                                         # one machine
fly open /seller
```

`--ha=false` matters: Fly starts two machines by default, and two processes
would each have their own SQLite file, locks and budget. Check with
`fly scale show` (count 1).

Optional, for showing traces of the agent: `fly secrets set
LANGSMITH_API_KEY=... LANGSMITH_TRACING=true LANGSMITH_PROJECT=sellerops-demo`.

## Updating

```bash
make lint test && fly deploy --ha=false
fly logs                                  # JSON logs, one line per request
```

## Try it locally first

```bash
docker build --target store -t sellerops-store .
docker run --rm -p 8000:8000 -e SELLEROPS_DEMO_MODE=1 -e SELLEROPS_APPROVAL_FAKE=1 sellerops-store
# real model: replace SELLEROPS_APPROVAL_FAKE=1 with --env-file .env (your key stays on your machine)
```

## Railway instead

The same Dockerfile works: build target `store`, the variables from
`fly.toml` `[env]`, one replica, no volume. Set `SELLEROPS_CLIENT_IP_HEADER`
only to a header the platform's proxy itself sets to the client IP (a header
the client can send, such as the first entry of `X-Forwarded-For`, lets a
visitor dodge the per-visitor limit). Left unset, the budget counts the proxy
as one client, so the hourly limit is shared by everyone; the daily limit
works either way.
