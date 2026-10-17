"""``cdui agent <start|stop|forget> ...``: hands off to ``app.agent`` in the venv.

``scripts/dev.py`` routes the ``agent`` group here after hopping into the
backend venv and applying the dev user-data dir
(``_dispatch_agent_subcommand``), so the agent finds the same
``server.addr`` and ``session.token`` as the server ``cdui start``
launched. Everything else lives in ``app.agent.cli``, which
``python -m app.agent`` also runs.
"""

from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent / "backend"


def main(argv: list[str] | None = None) -> int:
    if str(_BACKEND) not in sys.path:
        sys.path.insert(0, str(_BACKEND))
    from app.agent.cli import main as agent_main  # noqa: PLC0415 — needs venv

    return agent_main(argv)


if __name__ == "__main__":
    sys.exit(main())
