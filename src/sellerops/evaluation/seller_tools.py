"""Seller tools for evaluation: the production tools bound to one ``SellerEpisode``.

The definitions (names, descriptions, argument schemas) live in
``sellerops.harness.tools.seller`` so eval measures exactly what production
binds. Here the client is a ``SellerEpisode``: every call lands in the episode
history that the verifier reads (grounding, caption flow), and the tools are
closures over that one episode, so the model never sees a session id.
"""

from __future__ import annotations

from langchain_core.tools import BaseTool

from ..harness.tools.seller import SELLER_TOOL_NAMES
from ..harness.tools.seller import make_seller_tools as _make_tools
from ..simulator.episode import SellerEpisode

__all__ = ["SELLER_TOOL_NAMES", "make_seller_tools"]


def make_seller_tools(episode: SellerEpisode) -> list[BaseTool]:
    """The six seller tools bound to one episode."""
    return _make_tools(episode)
