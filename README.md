# SellerOps Agent

Portfolio project for building and evaluating a small seller-assistance agent:

```text
deterministic simulator
    → frozen evaluation tasks
    → base-model evaluation
    → LoRA SFT
    → LangGraph runtime and trajectory replay
    → optional targeted DPO after preference-failure analysis
```

The project currently contains strict action schemas, a deterministic local
catalog/inventory/policy environment, an oracle task generator, disjoint
train/development/frozen-test splits, and model-independent SFT decision
records.

## Repository layout

```text
projects/sellerops-agent/
├── PLAN.md
├── data/
├── notebooks/
├── scripts/
├── src/sellerops/
│   ├── agent/
│   ├── evaluation/
│   └── services/
└── tests/
    └── evaluation/
```

Evaluation code lives in [`src/sellerops/evaluation/`](src/sellerops/evaluation/README.md).
It contains the structured-outcome verifier, a LangSmith evaluation runner,
per-attempt logging, and local result summaries. Start with
`conda run -n llm-post-training python scripts/evaluate_agent.py --offline --limit 1`
from this project directory. Use `--cloud` with `LANGSMITH_API_KEY` configured
to publish the dev dataset, traces, and scores to LangSmith. The package README
explains setup, limits, and the commented local-loop comparison in `runner.py`.
All 30 dev tasks also have [GPT-6 Astra answer keys](data/evaluation/dev_expected_answers.jsonl),
including user-facing prose, structured expected outcomes, and acceptance notes.
The evaluation loader validates these against the simulator and exposes them as
reference outputs; they never enter the evaluated agent's prompt.
An optional Gemini prose judge (`--judge gemini`, `GEMINI_API_KEY`) adds combined
end-to-end scores. This sends evaluation content to Google even with `--offline`,
which controls LangSmith publishing only; see the evaluation guide for setup,
free-tier conditions, and error handling.

The shared Conda environment remains at the learning-lab root. Activate it with:

```bash
conda activate llm-post-training
```

From this project directory, regenerate the deterministic dataset with:

```bash
conda run -n llm-post-training python scripts/generate_sellerops_dataset.py
```

Run the current tests with:

```bash
conda run -n llm-post-training pytest -q
```

See [PLAN.md](PLAN.md) for the step-by-step learning contract and current
milestone. The project-level plan is governed by the lab-level
[`TARGET.md`](../../TARGET.md).

For the initial two-model router/action graph learning slice, read
[AGENT_WALKTHROUGH.md](AGENT_WALKTHROUGH.md). It includes a runnable example,
offline graph tests, bounded schema retries, and the remaining verification/multi-turn exercises.

## Episode snapshots and replay

### LangChain message conversion

`sellerops.agent.message_conversion` bridges decoded SellerOps model output to
LangChain messages. It validates actions before exposing tool calls and does
not load a model or execute tools by itself:

```python
from sellerops.agent.message_conversion import (
    model_output_to_ai_message,
    observation_to_tool_message,
    tool_call_to_action,
)
from sellerops.services.episode_service.episode_service import EpisodeService

decoded_output = '{"type":"tool_call","tool":"get_inventory","arguments":{"product_id":"P1001"}}'
response = model_output_to_ai_message(decoded_output)
service = EpisodeService()
episode = service.create_episode("conversion-demo")

for call in response.tool_calls:
    observation = service.execute_tool(episode.session_id, tool_call_to_action(call))
    tool_message = observation_to_tool_message(observation, tool_call=call)
    # Append response once, then its tool_message, to the conversation.
```

`final` and `ask_user` become `AIMessage` objects with empty `tool_calls`.
Their complete action remains in `json.loads(response.content)`; an empty
`tool_calls` list alone does not distinguish a final answer from clarification.
Tool-call actions also retain their JSON in `content`. When formatting local
model input, preserve the SFT format of role/content pairs (assistant action
JSON and tool observation JSON); do not automatically switch the tokenizer to
a different native tool-calling template. The model adapter's generation and
graph integration remain user work. Pass only decoded completion text, not
the input prompt or code fences. Invalid JSON/actions raise `SchemaError`.

`EpisodeService` creates isolated episodes, executes local tools with a step
limit, and exports snapshots. With `src` on your Python import path:

```python
import json

from sellerops.services.episode_service.episode_service import EpisodeService

service = EpisodeService()
episode = service.create_episode("demo-inventory")
service.execute_tool(
    episode.session_id,
    {
        "type": "tool_call",
        "tool": "get_inventory",
        "arguments": {"product_id": "P1001"},
    },
)

snapshot = service.snapshot(episode.session_id)
snapshot_json = json.dumps(snapshot, ensure_ascii=False, indent=2)
```

Snapshots contain `initial_environment` (captured at episode creation),
`environment` (current data), history, IDs, counters, and terminal status.
They contain independent data copies, not live environment objects. Exporting
does not write a file. `service.reset(episode.session_id)` restores the captured
initial environment and clears execution history while preserving IDs and the
step limit.

Replay the recorded actions, including an export loaded back from JSON:

```python
report = service.replay(json.loads(snapshot_json))
# {"matched": True, "steps_checked": 1, "first_mismatch": None}
```

Replay runs without a model in a temporary episode using `initial_environment`
and the saved step limit. It leaves the input snapshot and existing episodes
unchanged, and does not retain the temporary episode in the service registry.
It stops at the first differing tool observation. `first_mismatch` then contains
`step` (one-based), `action`, `expected`, and `actual`; `steps_checked` includes
the mismatching step. An empty history matches with zero checked steps.

Malformed snapshots are rejected with `ValueError` before dispatch; unexpected
tool implementation exceptions propagate. A match means the recorded tool
observations were reproduced by the current environment implementation. It does
not verify task success, final answers, or changes to live environment data that
were made outside recorded tools. Scenario/seed initialization remains future
work; structured outcome verification is available separately below.

## Structured outcome verification

`verify_outcome(task, snapshot, final_action)` checks the final structured data
against task context, initial environment data, and replayed tool evidence. It
returns `decision` (`pass` or `fail`), `reason_code`, and diagnostic `details`.
It does not choose retry/replan behavior or modify the episode.

Continuing the inventory example above:

```python
from sellerops.evaluation import verify_outcome

task = {
    "task_id": episode.task_id,
    "family": "inventory_offer",
    "context": {"product_id": "P1001", "quantity": 1, "mode": "stock_only"},
}
candidate = {
    "type": "final",
    "status": "success",
    "message": "Stok P1001 tersedia 7 unit.",
    "data": {"product_id": "P1001", "available_stock": 7},
}
evaluation = verify_outcome(task, service.snapshot(episode.session_id), candidate)
# {"decision": "pass", "reason_code": "outcome_valid", "details": {}}
```

Supported outcomes:

- Product search: query/filters, correct product facts, observed search results,
  and truthful `not_found`. Any non-empty eligible subset and ordering are
  accepted; duplicates are rejected. Missing-field tasks require `ask_user`
  naming the declared fields.
- Inventory/offer: product, stock, requested quantity, requested discount capped
  at the policy maximum, integer IDR pricing, and matching tool evidence. Stock
  and policy checks may occur in either order before a matching calculation.
  Stock-only or insufficient-stock outcomes must not calculate an offer.
- Policy questions: product-specific limit/source, eligibility, and proposal
  validity (`0 <= proposal <= maximum`). A blocked policy-violation attempt in
  history remains a failure even if a later final answer is safe.

Final `data` fields follow the existing dataset contracts exactly. The verifier
does not read oracle trajectories or enforce their exact action sequence. Task
specification errors raise `ValueError`; candidate action/snapshot errors return
a failing report. Unexpected environment execution errors propagate.

**Scope:** task context and runtime snapshots are trusted evaluation inputs.
Messages/questions must be non-empty but their prose is not semantically graded;
a passing structured answer is not a guarantee that its prose is truthful.
The current simulator is read-only, so unexplained environment changes fail.
The evaluation runner invokes the verifier after the graph finishes; it does
not drive graph routing or retries. Router completion alone is not an
outcome-verification result. Runs without a structured candidate are explicitly
scored as `missing_structured_outcome`.

Train/dev audit: 209 of 210 existing demonstrations pass. The dev task
`dev-policy-dev-check_request-id-002` incorrectly labels a -1% discount proposal
as allowed; the verifier rejects it because the tool schema requires a
non-negative discount. Dataset files were not changed. Frozen tasks were not
used to develop this verifier or run this audit.


## Guided full SFT on a cloud GPU

For the user's completed Drive version, use
[`notebooks/sft_drive_paths_fixed.ipynb`](notebooks/sft_drive_paths_fixed.ipynb).
This copy centralizes run paths, reloads the saved model and tokenizer together,
and verifies the active-run ZIP and its Drive copy. Section 3 retains the user's
checkpoint-165 source and `EPOCHS=2`; its guard requires a total epoch target above
the already completed 3 epochs to continue. Adjust the checkpoint path if it was
restored elsewhere. This notebook has passed local path/export checks but has
not been trained in Colab. The original guided notebook below remains unchanged.

`notebooks/sft.ipynb` now provides a full-parameter SFT learning exercise for
Qwen2.5-0.5B-Instruct, with four exercises and notes before every code cell.
It includes target-only label masking, strict development evaluation, a two-step
smoke test, fresh-weight training, and saved-checkpoint evaluation.

Colab uses its managed Python runtime, as authorized on September 6. Run notebook
sections 1 and 1b to audit the GPU and install the pinned training libraries while
preserving the installed CUDA PyTorch build. Restart the Python session after an
installation, then continue. Requirements are in `configs/requirements.colab.txt`.
Local work still uses Conda `llm-post-training`; `configs/environment.cuda.yml`
remains optional for other CUDA hosts. The root environment.yml is unchanged.
Edit upload/output paths before execution. The Colab editor's notebook and the
notebook extracted from the ZIP are separate copies; save your editor copy to Drive.
The notebook intentionally stops at unfinished exercises.

The September 6 training choice is **full SFT**, superseding the earlier LoRA-only
training plan for this experiment. Tool-history replay is now available;
scenario-based evaluation and frozen end-to-end evaluation remain unfinished.
Local SFT result artifacts have not yet been recorded.
