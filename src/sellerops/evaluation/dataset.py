"""Prepare dev examples without leaking oracle trajectories into agent inputs."""

import json
from copy import deepcopy
from hashlib import sha256
from uuid import NAMESPACE_URL, uuid5

from langsmith.schemas import Example
from langsmith.utils import LangSmithNotFoundError

from ..dataset import DATASET_VERSION, ENVIRONMENT_VERSION
from ..schemas import canonical_json, validate_task
from ..settings import get_settings
from .answer_keys import load_answer_keys

PROJECT_ROOT = get_settings().project_root

def load_dev_records() -> list[dict]:
    """This first integration intentionally loads only the development split."""
    content = (PROJECT_ROOT / "data/tasks/dev.jsonl").read_bytes()
    task_checksum = sha256(content).hexdigest()
    tasks, seen = [], set()
    for line in content.decode("utf-8").splitlines():
        if not line.strip():
            continue
        task = json.loads(line)
        validate_task(task)
        if task["split"] != "dev" or task["environment_version"] != ENVIRONMENT_VERSION:
            raise ValueError("Expected dev tasks for the current simulator version")
        if task["task_id"] in seen:
            raise ValueError(f"Duplicate task_id: {task['task_id']}")
        seen.add(task["task_id"])
        tasks.append(task)
    if not tasks:
        raise ValueError("Development dataset is empty")
    answers, answer_metadata = load_answer_keys(PROJECT_ROOT, tasks, task_checksum)
    # A changed answer key must create a new LangSmith dataset identity even
    # when prompts are unchanged. The original task-file hash stays explicit.
    checksum = sha256(canonical_json(answer_metadata).encode()).hexdigest()
    records = []
    for task in tasks:
        answer = answers[task["task_id"]]
        records.append({
            "inputs": {"task_id": task["task_id"], "prompt": task["prompt"]},
            "outputs": {"task": {key: task[key] for key in ("task_id", "family", "context")},
                        **{key: deepcopy(answer[key]) for key in (
                            "expected_answer", "expected_final_action", "acceptance_notes"
                        )}},
            "metadata": {key: task[key] for key in (
                "family", "locale", "seed", "split", "environment_version", "provenance"
            )} | {"dataset_version": DATASET_VERSION, "source_sha256": checksum, **answer_metadata},
        })
    return records


def as_examples(records: list[dict]) -> list[Example]:
    fingerprint = records[0]["metadata"]["source_sha256"]
    dataset_id = uuid5(NAMESPACE_URL, f"sellerops/dev/{fingerprint}")
    return [Example(id=uuid5(dataset_id, r["inputs"]["task_id"]), dataset_id=dataset_id,
                    **deepcopy(r)) for r in records]


def prepare_cloud_dataset(client, examples: list[Example]) -> list[Example]:
    """Reuse a versioned dataset; validate existing examples instead of overwriting."""
    fingerprint = examples[0].metadata["source_sha256"]
    name = f"sellerops-dev-{fingerprint[:12]}"
    try:
        dataset = client.read_dataset(dataset_name=name)
    except LangSmithNotFoundError:
        dataset = client.create_dataset(
            dataset_name=name,
            description="SellerOps dev prompts, verifier context and reviewed expected answers; no oracle trajectories.",
            metadata={"source_sha256": fingerprint, "dataset_version": DATASET_VERSION},
        )
    if (dataset.metadata or {}).get("source_sha256") != fingerprint:
        raise ValueError("Cloud dataset fingerprint differs from the local evaluation sources")
    existing = {example.id: example for example in client.list_examples(dataset_id=dataset.id)}
    expected = {example.id: example for example in examples}

    def check_content(remote_examples):
        if remote_examples.keys() - expected.keys():
            raise ValueError("Cloud dataset has unexpected examples; use a clean dataset version")
        for key, remote in remote_examples.items():
            local = expected[key]
            if remote.inputs != local.inputs or remote.outputs != local.outputs:
                raise ValueError(f"Cloud example differs from local source: {key}")
            # LangSmith may add system metadata; source fields must still match.
            if any((remote.metadata or {}).get(k) != v for k, v in local.metadata.items()):
                raise ValueError(f"Cloud example metadata differs: {key}")

    check_content(existing)
    missing = [e for e in examples if e.id not in existing]
    if missing:
        client.create_examples(dataset_id=dataset.id, examples=[
            {"id": e.id, "inputs": e.inputs, "outputs": e.outputs, "metadata": e.metadata}
            for e in missing
        ])
        existing = {e.id: e for e in client.list_examples(dataset_id=dataset.id)}
        check_content(existing)
    if existing.keys() != expected.keys():
        raise ValueError("Cloud dataset upload is incomplete; retry to populate missing examples")
    # Server timestamps identify the actual dataset version. Inputs/references
    # were checked against local records, so scoring still uses the same facts.
    return [existing[e.id] for e in examples]
