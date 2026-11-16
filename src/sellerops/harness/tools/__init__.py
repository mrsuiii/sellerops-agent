"""Production tools bound to the agent graph (none yet).

- Seller agent tools (Track C), e.g. ``get_reviews``, ``get_product``.
- ``store``: Kala Studio tools for the buyer agent (Track B), behind a
  ``StoreClient`` interface (in-process for tests, HTTP for production).

Eval-only tools live in ``sellerops.evaluation`` (``sim_tools``,
``terminal_tools``); this package must not import them.
"""
