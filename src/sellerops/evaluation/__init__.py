"""Evaluation for the seller agent on the seller simulator (multi-turn).

- ``seller_tools`` / ``terminal_tools`` / ``prompts``: what the agent gets;
- ``seller_tasks``: dev/test tasks with computed answer keys;
- ``user_sim``: ``ScriptedUser`` (tests) and ``LLMUser`` (eval);
- ``conversation``: the agent → user → agent turn loop;
- ``seller_verifier``: outcome checks; ``metrics``: pass@k and pass^k.
"""
