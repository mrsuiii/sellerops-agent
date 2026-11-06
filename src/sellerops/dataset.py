"""Generate deterministic SellerOps tasks, oracle trajectories, and SFT records."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .environment import CATALOG, INVENTORY, SellerEnvironment, environment_fixtures
from .schemas import canonical_json, validate_sft_record, validate_task


DATASET_VERSION = "sellerops-sft-v0.2"
ENVIRONMENT_VERSION = "sellerops-sim-v0.1"
GENERATOR_VERSION = "sellerops-generator-v0.2"
DEFAULT_SEED = 20260819

SYSTEM_PROMPT = """
You are the SellerOps decision policy.

Return exactly one valid JSON object and nothing else. Do not return Markdown,
code fences, explanations, reasoning, or multiple JSON values.

The available actions and their exact schemas are:

1. Call one tool:
{"type":"tool_call","tool":"<tool_name>","arguments":{...}}

Available tools:
- search_products: Find products by query and optional filters.
  Required arguments: query (string) and filters (object). Filters may contain
  only brand (string), category (string), color (string), in_stock (boolean),
  and max_price_idr (integer).
- get_inventory: Retrieve current inventory for one known product.
  Required arguments: product_id (string).
- get_discount_policy: Retrieve the discount policy for one known product.
  Required arguments: product_id (string).
- calculate_offer: Calculate a policy-compliant offer for one known product.
  Required arguments: product_id (string), quantity (integer of at least 1),
  and discount_percent (integer from 0 through 100).

2. Ask for required information that is missing:
{"type":"ask_user","question":"<question>","missing_fields":["<field>"]}

3. Return the final response after enough information is available:
{"type":"final","status":"<status>","message":"<message>","data":{}}
The status must be exactly "success", "unavailable", or "not_found".

Rules:
- Use only the action fields and tool names listed above.
- The tool field must be a tool-name string, not a tool-definition object.
- Put tool inputs under arguments; never use an input field.
- Do not use <tool_call> tags or any native model-specific tool-call wrapper.
- Use search_products when a product_id is not yet known.
- Use product_id for inventory, discount-policy, and offer calls.
- Never invent tool observations, product information, or policy limits.
- Never exceed a discount limit.
- Return one action at a time.
""".strip()


@dataclass(frozen=True)
class TemplateSpec:
    template_id: str
    split: str
    family: str
    locale: str
    variant: str
    count: int


TEMPLATES: tuple[TemplateSpec, ...] = (
    # Train: five scenario templates per family, 12 instances each = 180 tasks.
    TemplateSpec("search.train.budget.id", "train", "product_search", "id", "budget", 12),
    TemplateSpec("search.train.color.id", "train", "product_search", "id", "color", 12),
    TemplateSpec("search.train.keyword.id", "train", "product_search", "id", "keyword", 12),
    TemplateSpec("search.train.budget.en", "train", "product_search", "en", "budget", 12),
    TemplateSpec("search.train.missing_budget.id", "train", "product_search", "id", "missing_budget", 12),
    TemplateSpec("offer.train.maximum.id", "train", "inventory_offer", "id", "maximum", 12),
    TemplateSpec("offer.train.requested.id", "train", "inventory_offer", "id", "requested", 12),
    TemplateSpec("offer.train.quantity.id", "train", "inventory_offer", "id", "quantity", 12),
    TemplateSpec("offer.train.safe.en", "train", "inventory_offer", "en", "safe", 12),
    TemplateSpec("offer.train.out_of_stock.id", "train", "inventory_offer", "id", "out_of_stock", 12),
    TemplateSpec("policy.train.limit.id", "train", "policy_qa", "id", "limit", 12),
    TemplateSpec("policy.train.limit.en", "train", "policy_qa", "en", "limit", 12),
    TemplateSpec("policy.train.proposal.id", "train", "policy_qa", "id", "proposal", 12),
    TemplateSpec("policy.train.source.id", "train", "policy_qa", "id", "source", 12),
    TemplateSpec("policy.train.eligibility.id", "train", "policy_qa", "id", "eligibility", 12),
    # Dev: one unseen scenario template per family, 10 instances each = 30 tasks.
    TemplateSpec("search.dev.brand.id", "dev", "product_search", "id", "brand", 10),
    TemplateSpec("offer.dev.customer_request.en", "dev", "inventory_offer", "en", "customer_request", 10),
    TemplateSpec("policy.dev.check_request.id", "dev", "policy_qa", "id", "check_request", 10),
    # Frozen test: two unseen scenario templates per family, 8 instances each = 48 tasks.
    TemplateSpec("search.test.combined.id", "test", "product_search", "id", "combined", 8),
    TemplateSpec("search.test.combined.en", "test", "product_search", "en", "combined", 8),
    TemplateSpec("offer.test.cap_request.id", "test", "inventory_offer", "id", "cap_request", 8),
    TemplateSpec("offer.test.cap_request.en", "test", "inventory_offer", "en", "cap_request", 8),
    TemplateSpec("policy.test.audit.id", "test", "policy_qa", "id", "audit", 8),
    TemplateSpec("policy.test.audit.en", "test", "policy_qa", "en", "audit", 8),
)


def _stable_seed(seed: int, label: str) -> int:
    """Derive a deterministic integer seed from a seed and label pair.

    Args:
        seed: Experiment-level integer seed.
        label: Namespace that separates template or task permutations.

    Returns:
        The first eight bytes of a SHA-256 digest interpreted as an integer.
    """
    digest = hashlib.sha256(f"{seed}:{label}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _ordered_products(seed: int, template_id: str, *, in_stock: bool | None = None) -> list[dict[str, Any]]:
    """Return a deterministically shuffled product list with optional stock filtering.

    Args:
        seed: Experiment-level integer seed.
        template_id: Template namespace used to make ordering reproducible.
        in_stock: ``True`` keeps stocked products, ``False`` keeps zero-stock
            products, and ``None`` keeps the full catalog.

    Returns:
        New product dictionaries in a stable pseudo-random order.
    """
    stock_by_id = {row["product_id"]: row["stock"] for row in INVENTORY}
    products = [dict(product) for product in CATALOG]
    if in_stock is True:
        products = [product for product in products if stock_by_id[product["product_id"]] > 0]
    elif in_stock is False:
        products = [product for product in products if stock_by_id[product["product_id"]] == 0]
    random.Random(_stable_seed(seed, template_id)).shuffle(products)
    return products


def _action(tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Construct one tagged tool-call action.

    Args:
        tool: Name of the local tool to call.
        arguments: Tool-specific JSON-compatible arguments.

    Returns:
        A dictionary in the SellerOps ``tool_call`` action format.
    """
    return {"type": "tool_call", "tool": tool, "arguments": arguments}


def _assistant(content: dict[str, Any]) -> dict[str, Any]:
    """Wrap action content as an assistant trajectory event.

    Args:
        content: Tagged action object emitted by the oracle.

    Returns:
        A two-field assistant event containing the action.
    """
    return {"role": "assistant", "content": content}


def _tool(content: dict[str, Any]) -> dict[str, Any]:
    """Wrap an observation as a tool trajectory event.

    Args:
        content: Structured observation returned by the environment.

    Returns:
        A two-field tool event containing the observation.
    """
    return {"role": "tool", "content": content}


def _run_tool(
    environment: SellerEnvironment,
    trajectory: list[dict[str, Any]],
    action: dict[str, Any],
) -> dict[str, Any]:
    """Execute an action and append both sides of the interaction to a trajectory.

    Args:
        environment: Deterministic environment used to execute the action.
        trajectory: Mutable event list receiving assistant and tool events.
        action: Validated tool-call action to execute.

    Returns:
        The structured tool observation produced by ``environment``.
    """
    trajectory.append(_assistant(action))
    observation = environment.execute(action)
    trajectory.append(_tool(observation))
    return observation


def _build_search_context(
    spec: TemplateSpec,
    product: dict[str, Any],
    index: int,
) -> tuple[str, dict[str, Any]]:
    """Create a search prompt and oracle context for one template instance.

    Args:
        spec: Scenario template describing language and search variant.
        product: Product selected as the deterministic target.
        index: Instance index used for varied budgets and use cases.

    Returns:
        A ``(prompt, context)`` pair consumed by the search oracle.
    """
    tag = product["tags"][index % len(product["tags"])]
    budget = product["price_idr"] + (index % 3) * 50_000
    base_filters: dict[str, Any] = {"in_stock": True}

    if spec.variant == "missing_budget":
        use_cases = [
            "bekerja dari rumah", "perjalanan singkat", "meja belajar", "olahraga pagi",
            "mendengarkan musik", "mengisi daya ponsel", "membuat kopi", "melindungi ponsel",
            "membawa laptop", "menata meja", "mengetik", "aktivitas harian",
        ]
        prompt = f"Saya butuh produk untuk {use_cases[index]}. Tolong bantu pilihkan."
        return prompt, {"missing_fields": ["budget_idr"], "use_case": use_cases[index]}

    if spec.variant == "budget":
        filters = {**base_filters, "category": product["category"], "max_price_idr": budget}
        if spec.locale == "id":
            prompt = (
                f"Cari produk kategori {product['category']} dengan anggaran maksimal "
                f"Rp{budget:,} dan pastikan stok tersedia."
            )
        else:
            prompt = (
                f"Find an in-stock {product['category']} product with a maximum budget of "
                f"IDR {budget:,}."
            )
        return prompt, {"query": product["category"], "filters": filters}

    if spec.variant == "color":
        filters = {**base_filters, "color": product["color"]}
        prompt = f"Carikan produk {tag} warna {product['color']} yang masih tersedia."
        return prompt, {"query": tag, "filters": filters}

    if spec.variant == "keyword":
        prompt = f"Rekomendasikan produk terkait '{tag}' yang stoknya tersedia."
        return prompt, {"query": tag, "filters": base_filters}

    if spec.variant == "brand":
        filters = {**base_filters, "brand": product["brand"]}
        prompt = f"Cari produk merek {product['brand']} untuk kebutuhan {tag} yang tersedia."
        return prompt, {"query": tag, "filters": filters}

    filters = {
        **base_filters,
        "color": product["color"],
        "max_price_idr": budget,
    }
    if spec.locale == "id":
        prompt = (
            f"Temukan produk {tag} warna {product['color']}, stok tersedia, dengan harga "
            f"tidak lebih dari Rp{budget:,}."
        )
    else:
        prompt = (
            f"Find an in-stock {product['color']} {tag} product priced at no more than "
            f"IDR {budget:,}."
        )
    return prompt, {"query": tag, "filters": filters}


def _build_offer_context(
    spec: TemplateSpec,
    product: dict[str, Any],
    index: int,
    environment: SellerEnvironment,
) -> tuple[str, dict[str, Any]]:
    """Create an inventory/offer prompt and its structured decision context.

    Args:
        spec: Scenario template describing offer behavior and language.
        product: Product selected as the deterministic target.
        index: Instance index used to vary quantity and discount requests.
        environment: Source of the product's maximum discount policy.

    Returns:
        A ``(prompt, context)`` pair containing product, quantity, and discount
        information for the offer oracle.
    """
    product_id = product["product_id"]
    maximum = environment.policies[product_id]["max_discount_percent"]
    quantity = 1 + index % 3
    safe_request = max(0, maximum - index % 3)

    if spec.variant == "out_of_stock":
        quantity = 1 + index // 2
        prompt = (
            f"Periksa apakah {product['name']} ({product_id}) tersedia sebanyak {quantity} unit. "
            "Jika tidak, jangan buat penawaran."
        )
        return prompt, {"product_id": product_id, "quantity": quantity, "mode": "stock_only"}

    if spec.variant == "maximum":
        # There are ten in-stock products and twelve instances. Repeated
        # products use a different, still-feasible quantity to avoid duplicate
        # prompts while preserving deterministic coverage.
        quantity = 1 + index // 10
        prompt = (
            f"Cek stok {product['name']} ({product_id}) lalu buat penawaran {quantity} unit dengan "
            "diskon maksimum yang diizinkan."
        )
        return prompt, {"product_id": product_id, "quantity": quantity, "requested_discount_percent": maximum}

    if spec.variant == "requested":
        prompt = (
            f"Pelanggan meminta {quantity} unit {product['name']} ({product_id}) dengan diskon "
            f"{safe_request}%. Verifikasi stok dan kebijakan sebelum membuat penawaran."
        )
        return prompt, {"product_id": product_id, "quantity": quantity, "requested_discount_percent": safe_request}

    if spec.variant == "quantity":
        prompt = (
            f"Siapkan penawaran untuk {quantity} unit {product['name']} ({product_id}) dengan "
            f"diskon {safe_request}% setelah memeriksa stok dan batas diskon."
        )
        return prompt, {"product_id": product_id, "quantity": quantity, "requested_discount_percent": safe_request}

    if spec.variant in {"safe", "customer_request"}:
        prompt = (
            f"Check stock and prepare a policy-compliant offer for {quantity} unit(s) of "
            f"{product['name']} ({product_id}) with a {safe_request}% discount."
        )
        return prompt, {"product_id": product_id, "quantity": quantity, "requested_discount_percent": safe_request}

    # Frozen cap-request scenarios deliberately request more than the limit; the
    # oracle must cap the offer at the policy maximum instead of violating it.
    unsafe_request = maximum + 5 + index % 3
    if spec.locale == "id":
        prompt = (
            f"Pelanggan meminta diskon {unsafe_request}% untuk {quantity} unit {product['name']} "
            f"({product_id}). Cek stok dan buat penawaran terbaik yang tetap patuh kebijakan."
        )
    else:
        prompt = (
            f"A customer requests {unsafe_request}% off {quantity} unit(s) of {product['name']} "
            f"({product_id}). Check stock and make the best policy-compliant offer."
        )
    return prompt, {
        "product_id": product_id,
        "quantity": quantity,
        "requested_discount_percent": unsafe_request,
    }


def _build_policy_context(
    spec: TemplateSpec,
    product: dict[str, Any],
    index: int,
    environment: SellerEnvironment,
) -> tuple[str, dict[str, Any]]:
    """Create a policy-question prompt and structured policy context.

    Args:
        spec: Scenario template describing the policy question variant.
        product: Product selected as the deterministic target.
        index: Instance index used to vary proposed discounts.
        environment: Source of the product's policy maximum.

    Returns:
        A ``(prompt, context)`` pair consumed by the policy oracle.
    """
    product_id = product["product_id"]
    maximum = environment.policies[product_id]["max_discount_percent"]
    proposal = maximum + (-1 if index % 2 == 0 else 3)
    context: dict[str, Any] = {"product_id": product_id, "question_type": spec.variant}

    if spec.variant == "limit":
        if spec.locale == "id":
            prompt = f"Berapa batas diskon untuk {product['name']} ({product_id})? Sertakan sumber kebijakannya."
        else:
            prompt = f"What is the discount limit for {product['name']} ({product_id})? Include the policy source."
    elif spec.variant == "proposal":
        context["proposed_discount_percent"] = proposal
        prompt = (
            f"Apakah diskon {proposal}% boleh diberikan untuk {product['name']} ({product_id})? "
            "Periksa kebijakan terlebih dahulu."
        )
    elif spec.variant == "source":
        prompt = f"Ambil kebijakan diskon dan sumber resminya untuk produk {product_id}."
    elif spec.variant == "eligibility":
        prompt = f"Cek apakah {product['name']} ({product_id}) dapat diberi diskon dan sebutkan batasnya."
    elif spec.variant == "check_request":
        context["proposed_discount_percent"] = proposal
        prompt = f"Verifikasi apakah permintaan diskon {proposal}% untuk produk {product_id} sesuai kebijakan."
    elif spec.locale == "id":
        context["proposed_discount_percent"] = proposal
        prompt = (
            f"Audit usulan diskon {proposal}% untuk {product['name']} ({product_id}) berdasarkan "
            "sumber kebijakan yang berlaku."
        )
    else:
        context["proposed_discount_percent"] = proposal
        prompt = (
            f"Audit a proposed {proposal}% discount for {product['name']} ({product_id}) against "
            "the applicable policy source."
        )
    return prompt, context


def _oracle_search(
    environment: SellerEnvironment,
    context: dict[str, Any],
    locale: str,
) -> list[dict[str, Any]]:
    """Produce the canonical search trajectory for a structured context.

    Args:
        environment: Deterministic environment used for product lookup.
        context: Search constraints or missing-field metadata.
        locale: ``id`` or ``en`` language used for final messages.

    Returns:
        An assistant/tool event sequence ending in ``ask_user`` or ``final``.
    """
    if "missing_fields" in context:
        question = (
            "Berapa anggaran maksimum Anda?"
            if locale == "id"
            else "What is your maximum budget?"
        )
        return [_assistant({"type": "ask_user", "question": question, "missing_fields": context["missing_fields"]})]

    trajectory: list[dict[str, Any]] = []
    observation = _run_tool(
        environment,
        trajectory,
        _action("search_products", {"query": context["query"], "filters": context["filters"]}),
    )
    products = observation["result"]["products"]
    if not products:
        message = "Tidak ada produk yang cocok." if locale == "id" else "No matching products were found."
        trajectory.append(_assistant({"type": "final", "status": "not_found", "message": message, "data": {"products": []}}))
    else:
        names = ", ".join(product["name"] for product in products)
        message = (
            f"Produk yang sesuai: {names}."
            if locale == "id"
            else f"Matching products: {names}."
        )
        trajectory.append(_assistant({"type": "final", "status": "success", "message": message, "data": {"products": products}}))
    return trajectory


def _oracle_offer(
    environment: SellerEnvironment,
    context: dict[str, Any],
    locale: str,
) -> list[dict[str, Any]]:
    """Produce a stock-aware, policy-compliant offer trajectory.

    Args:
        environment: Deterministic environment used for stock and pricing tools.
        context: Product, quantity, and requested discount details.
        locale: ``id`` or ``en`` language used for final messages.

    Returns:
        A trajectory that checks stock, optionally checks policy, and ends with
        an unavailable or successful final action.
    """
    trajectory: list[dict[str, Any]] = []
    product_id = context["product_id"]
    quantity = context["quantity"]
    inventory = _run_tool(environment, trajectory, _action("get_inventory", {"product_id": product_id}))
    available = inventory["result"]["stock"]
    if available < quantity:
        message = (
            f"Stok tidak mencukupi: diminta {quantity}, tersedia {available}."
            if locale == "id"
            else f"Insufficient stock: requested {quantity}, available {available}."
        )
        trajectory.append(
            _assistant(
                {
                    "type": "final",
                    "status": "unavailable",
                    "message": message,
                    "data": {"available_stock": available, "product_id": product_id, "requested_quantity": quantity},
                }
            )
        )
        return trajectory

    if context.get("mode") == "stock_only":
        message = (
            f"Stok tersedia sebanyak {available} unit."
            if locale == "id"
            else f"There are {available} units in stock."
        )
        trajectory.append(_assistant({"type": "final", "status": "success", "message": message, "data": {"available_stock": available, "product_id": product_id}}))
        return trajectory

    policy = _run_tool(
        environment,
        trajectory,
        _action("get_discount_policy", {"product_id": product_id}),
    )
    maximum = policy["result"]["max_discount_percent"]
    requested = context["requested_discount_percent"]
    applied = min(requested, maximum)
    offer = _run_tool(
        environment,
        trajectory,
        _action(
            "calculate_offer",
            {"product_id": product_id, "discount_percent": applied, "quantity": quantity},
        ),
    )
    result = offer["result"]
    if locale == "id":
        message = (
            f"Penawaran patuh kebijakan: diskon {applied}%, total Rp{result['total_idr']:,} "
            f"untuk {quantity} unit."
        )
    else:
        message = (
            f"Policy-compliant offer: {applied}% discount, IDR {result['total_idr']:,} "
            f"total for {quantity} unit(s)."
        )
    trajectory.append(_assistant({"type": "final", "status": "success", "message": message, "data": result}))
    return trajectory


def _oracle_policy(
    environment: SellerEnvironment,
    context: dict[str, Any],
    locale: str,
) -> list[dict[str, Any]]:
    """Produce a trajectory answering a product discount-policy question.

    Args:
        environment: Deterministic environment used to retrieve policy data.
        context: Product ID, question type, and optional proposed discount.
        locale: ``id`` or ``en`` language used for the final message.

    Returns:
        A policy lookup followed by a structured final answer.
    """
    trajectory: list[dict[str, Any]] = []
    product_id = context["product_id"]
    policy = _run_tool(
        environment,
        trajectory,
        _action("get_discount_policy", {"product_id": product_id}),
    )
    result = policy["result"]
    maximum = result["max_discount_percent"]
    question_type = context["question_type"]
    final_data: dict[str, Any] = dict(result)
    if "proposed_discount_percent" in context:
        proposal = context["proposed_discount_percent"]
        allowed = proposal <= maximum
        final_data.update(
            {"proposal_allowed": allowed, "proposed_discount_percent": proposal}
        )
        if locale == "id":
            message = (
                f"Batas diskon produk {product_id} adalah {maximum}% menurut {result['source']}. "
                f"Usulan {proposal}% {'diizinkan' if allowed else 'tidak diizinkan'}."
            )
        else:
            message = (
                f"The discount limit for {product_id} is {maximum}% under {result['source']}. "
                f"The proposed {proposal}% discount is {'allowed' if allowed else 'not allowed'}."
            )
    elif question_type == "eligibility":
        eligible = maximum > 0
        final_data["discount_eligible"] = eligible
        message = (
            f"Produk {product_id} {'dapat' if eligible else 'tidak dapat'} diberi diskon; "
            f"batasnya {maximum}% menurut {result['source']}."
        )
    elif locale == "id":
        message = (
            f"Batas diskon produk {product_id} adalah {maximum}% menurut {result['source']}."
        )
    else:
        message = (
            f"The discount limit for {product_id} is {maximum}% under {result['source']}."
        )
    trajectory.append(
        _assistant(
            {
                "type": "final",
                "status": "success",
                "message": message,
                "data": final_data,
            }
        )
    )
    return trajectory


def _build_task(
    spec: TemplateSpec,
    index: int,
    seed: int,
    environment: SellerEnvironment,
) -> dict[str, Any]:
    """Build, validate, and return one deterministic task instance.

    Args:
        spec: Template specification controlling family, split, and variant.
        index: Zero-based instance number within the template.
        seed: Experiment seed used for product selection and task metadata.
        environment: Shared read-only environment used by the oracle.

    Returns:
        A validated task containing prompt, context, oracle trajectory, verifier,
        and provenance metadata.
    """
    if spec.variant == "out_of_stock":
        products = _ordered_products(seed, spec.template_id, in_stock=False)
    elif spec.family == "inventory_offer":
        products = _ordered_products(seed, spec.template_id, in_stock=True)
    else:
        products = _ordered_products(seed, spec.template_id)
    product = products[index % len(products)]

    if spec.family == "product_search":
        prompt, context = _build_search_context(spec, product, index)
        trajectory = _oracle_search(environment, context, spec.locale)
    elif spec.family == "inventory_offer":
        prompt, context = _build_offer_context(spec, product, index, environment)
        trajectory = _oracle_offer(environment, context, spec.locale)
    else:
        prompt, context = _build_policy_context(spec, product, index, environment)
        trajectory = _oracle_policy(environment, context, spec.locale)

    task_seed = _stable_seed(seed, f"{spec.template_id}:{index}") % (2**31)
    task = {
        "task_id": f"{spec.split}-{spec.template_id.replace('.', '-')}-{index:03d}",
        "split": spec.split,
        "template_id": spec.template_id,
        "family": spec.family,
        "locale": spec.locale,
        "seed": task_seed,
        "environment_version": ENVIRONMENT_VERSION,
        "prompt": prompt,
        "context": context,
        "trajectory": trajectory,
        "verifier": {
            "expected_action_types": [event["content"]["type"] for event in trajectory if event["role"] == "assistant"],
            "expected_final_status": trajectory[-1]["content"].get("status"),
            "forbid_policy_violation": True,
            "required_tools": [event["content"]["tool"] for event in trajectory if event["role"] == "assistant" and event["content"]["type"] == "tool_call"],
        },
        "provenance": {
            "generator": GENERATOR_VERSION,
            "source": "synthetic_deterministic_oracle",
            "template_id": spec.template_id,
        },
    }
    validate_task(task)
    return task


def generate_tasks(seed: int = DEFAULT_SEED) -> dict[str, list[dict[str, Any]]]:
    """Generate all train, development, and frozen-test task splits.

    Args:
        seed: Seed controlling reproducible product ordering and task IDs.

    Returns:
        A dictionary mapping ``train``, ``dev``, and ``test`` to task lists.
    """
    environment = SellerEnvironment()
    output = {"train": [], "dev": [], "test": []}
    for spec in TEMPLATES:
        for index in range(spec.count):
            output[spec.split].append(_build_task(spec, index, seed, environment))
    return output


def task_to_sft_records(task: dict[str, Any]) -> list[dict[str, Any]]:
    """Create one final-assistant-target record per oracle decision point.

    Args:
        task: Validated train or development task with an oracle trajectory.

    Returns:
        SFT records whose final message is the assistant action to predict.

    Raises:
        ValueError: If a frozen-test task is supplied.
    """

    if task["split"] == "test":
        raise ValueError("frozen test tasks must never be serialized as SFT records")
    context_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task["prompt"]},
    ]
    records: list[dict[str, Any]] = []
    assistant_index = 0
    for event in task["trajectory"]:
        serialized = {"role": event["role"], "content": canonical_json(event["content"])}
        if event["role"] == "assistant":
            messages = [dict(message) for message in context_messages] + [serialized]
            record = {
                "record_id": f"{task['task_id']}-a{assistant_index:02d}",
                "task_id": task["task_id"],
                "split": task["split"],
                "template_id": task["template_id"],
                "family": task["family"],
                "locale": task["locale"],
                "dataset_version": DATASET_VERSION,
                "environment_version": task["environment_version"],
                "messages": messages,
                "target_message_index": len(messages) - 1,
                "provenance": {
                    "source_task_id": task["task_id"],
                    "generator": GENERATOR_VERSION,
                    "masking_contract": "final_assistant_message_only",
                },
            }
            validate_sft_record(record)
            records.append(record)
            assistant_index += 1
        context_messages.append(serialized)
    return records


def generate_sft_records(
    tasks: dict[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    """Convert train and development tasks into model-independent SFT records.

    Args:
        tasks: Mapping of split names to generated task lists.

    Returns:
        A mapping containing only ``train`` and ``dev`` record lists.
    """
    return {
        split: [record for task in tasks[split] for record in task_to_sft_records(task)]
        for split in ("train", "dev")
    }


def _json_bytes(value: Any) -> bytes:
    """Serialize a value as pretty, deterministic UTF-8 JSON bytes.

    Args:
        value: JSON-serializable value to encode.

    Returns:
        Indented JSON bytes ending in one newline.
    """
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def _jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    """Serialize dictionaries as compact canonical JSON Lines bytes.

    Args:
        rows: Records to encode, one object per line.

    Returns:
        UTF-8 bytes containing one canonical JSON object per row.
    """
    return b"".join((canonical_json(row) + "\n").encode() for row in rows)


def _sha256(content: bytes) -> str:
    """Compute the hexadecimal SHA-256 digest of byte content.

    Args:
        content: Bytes whose integrity fingerprint is needed.

    Returns:
        Lowercase hexadecimal SHA-256 digest.
    """
    return hashlib.sha256(content).hexdigest()


def build_dataset_files(seed: int = DEFAULT_SEED) -> dict[str, bytes]:
    """Build every generated dataset artifact in memory.

    Args:
        seed: Seed passed to task and SFT-record generation.

    Returns:
        Mapping from relative file paths to deterministic file bytes, including
        fixtures, task splits, SFT records, and their manifest.
    """

    tasks = generate_tasks(seed)
    sft_records = generate_sft_records(tasks)
    fixtures = environment_fixtures()
    files: dict[str, bytes] = {
        "environment/catalog.json": _json_bytes(fixtures["catalog"]),
        "environment/inventory.json": _json_bytes(fixtures["inventory"]),
        "environment/policies.json": _json_bytes(fixtures["policies"]),
        "tasks/train.jsonl": _jsonl_bytes(tasks["train"]),
        "tasks/dev.jsonl": _jsonl_bytes(tasks["dev"]),
        "tasks/test.jsonl": _jsonl_bytes(tasks["test"]),
        "sft/train.jsonl": _jsonl_bytes(sft_records["train"]),
        "sft/dev.jsonl": _jsonl_bytes(sft_records["dev"]),
    }
    template_ids = {
        split: sorted({task["template_id"] for task in tasks[split]})
        for split in ("train", "dev", "test")
    }
    manifest = {
        "created_on": "2026-08-20",
        "dataset_version": DATASET_VERSION,
        "environment_version": ENVIRONMENT_VERSION,
        "generator_version": GENERATOR_VERSION,
        "seed": seed,
        "task_counts": {split: len(rows) for split, rows in tasks.items()},
        "sft_record_counts": {split: len(rows) for split, rows in sft_records.items()},
        "template_ids": template_ids,
        "frozen_test_sha256": _sha256(files["tasks/test.jsonl"]),
        "file_sha256": {path: _sha256(content) for path, content in sorted(files.items())},
        "notes": [
            "Templates are disjoint across train, dev, and frozen test splits.",
            "Only train and dev tasks are serialized into SFT records.",
            "Each SFT record trains only its final assistant message.",
            "Every model input includes the explicit custom action and tool contract.",
            "No hidden chain-of-thought is stored.",
        ],
    }
    files["manifest.json"] = _json_bytes(manifest)
    return files


def write_dataset(output_dir: Path, seed: int = DEFAULT_SEED) -> dict[str, bytes]:
    """Generate dataset files and write them beneath an output directory.

    Args:
        output_dir: Root directory receiving the relative dataset paths.
        seed: Seed passed to deterministic dataset generation.

    Returns:
        The same relative-path-to-bytes mapping written to disk.
    """
    files = build_dataset_files(seed)
    for relative_path, content in files.items():
        destination = output_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
    return files
