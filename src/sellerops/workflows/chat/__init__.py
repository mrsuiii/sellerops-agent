"""Seller chat: a conversation with the seller agent over the production seller tools.

The graph is ``harness.build_agent_graph`` in chat mode (no terminal tools: a
plain-text reply ends the turn). The agent can read sales, stock, reviews and
products, and propose caption changes; it cannot approve them.
"""
