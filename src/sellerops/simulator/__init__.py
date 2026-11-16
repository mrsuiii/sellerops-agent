"""Deterministic SellerOps simulator (evaluation and dataset generation only).

- ``environment``: ``SellerEnvironment`` with the synthetic catalog, inventory,
  discount policies and the four read-only tools.
- ``episode_state`` / ``episode_service``: one isolated environment per task,
  tool history, step limit, snapshot and replay (input of ``verify_outcome``).

Not used by production agents; ``sellerops.harness`` must not import it.
"""
