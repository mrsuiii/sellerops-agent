# SellerOps Synthetic Dataset v0.2

This dataset teaches a small language model to select structured SellerOps
actions for three task families:

1. product search and recommendation;
2. inventory-aware, policy-compliant offer creation;
3. discount-policy retrieval and questions.

## Source and provenance

All examples are synthetic. They are generated from the versioned local product
catalog, inventory, and discount-policy fixtures by a deterministic oracle. No
production conversations or hidden chain-of-thought are included.

Generate the files with:

```bash
# Run from projects/sellerops-agent/
conda run -n llm-post-training python scripts/generate_sellerops_dataset.py
```

The seed, versions, counts, template assignments, file hashes, and frozen test
checksum are recorded in `manifest.json`.

## Splits

| Split | Tasks | Intended use |
|---|---:|---|
| `train` | 180 | SFT training |
| `dev` | 30 | preprocessing checks and model selection |
| `test` | 48 | frozen end-to-end evaluation only |

Scenario template IDs are disjoint across splits. Frozen test tasks must not be
used for SFT, prompt tuning, preference-pair generation, or failure-specific
patching.

## Development evaluation answer keys

The evaluation harness joins the 30 dev tasks with
[`evaluation/dev_expected_answers.jsonl`](evaluation/dev_expected_answers.jsonl)
by task ID. GPT-6 Astra authored these references from task requirements and
simulator facts. Each key includes a user-facing answer, a structured terminal
action, acceptance notes, and tool calls that support validation. These are
evaluation references and are not added to the agent's input or SFT records.

The companion manifest records author model, version, date, and source hashes.
The evaluation loader checks complete coverage and validates every structured
answer using a fresh simulator episode and the outcome verifier. It rejects
stale task/fixture hashes or invalid keys. The negative discount proposal in
the dev set is correctly rejected, independently of the old oracle label.

The prose is a reference answer, not an exact-match grading target. Existing
structured-outcome grading remains authoritative for structured scores. The optional
Gemini judge uses these references to assess final prose and combined end-to-end
scores. Regenerating the original dataset does not regenerate
these model-authored keys; source changes require review and versioning of the
keys and their manifest. The training and frozen test files are unchanged.

## SFT record contract

`sft/train.jsonl` and `sft/dev.jsonl` contain one decision target per record.
Each record ends in the assistant message to predict and declares
`target_message_index`. During preprocessing, all earlier context must receive
label `-100`; only the final assistant message contributes to loss.

Assistant messages contain exactly one tagged JSON action:

- `tool_call`
- `ask_user`
- `final`

Every record includes the same explicit system contract: the exact tagged-JSON
action schemas, all four allowed tool names and their argument schemas, and the
rule to emit one JSON action without Markdown or hidden reasoning. Native model
tool-call wrappers such as `<tool_call>` are not part of this dataset contract.

Tool observations are structured JSON and are context, never prediction
targets in this dataset version.

## Important limitation

The oracle is deliberately small and rule-based. SFT results show adaptation to
this environment and action language; they do not establish general e-commerce
or real-world policy competence.
