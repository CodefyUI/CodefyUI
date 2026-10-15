"""``python -m app.agent start|stop`` (what ``cdui agent`` runs).

Nothing at module level but this docstring: a ``spawn`` run subprocess
re-imports the parent's ``__main__`` module, and that import must stay
cheap and free of side effects.
"""

if __name__ == "__main__":
    import sys

    from app.agent.cli import main

    sys.exit(main())
