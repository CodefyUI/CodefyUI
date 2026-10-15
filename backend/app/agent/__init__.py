"""The CodefyUI worker agent (``cdui agent``, #771).

A worker agent dials the control plane's ``/ws/worker`` and executes the
runs it is assigned, each in a subprocess of its own, so a crash, an
out-of-memory kill or a forced stop ends that run and nothing else.

- ``cli``: ``cdui agent start|stop`` (``python -m app.agent``).
- ``connection``: the WebSocket client (hello, heartbeat, dispatch).
- ``runner``: one spawned subprocess per run, its pipe, cancel and crash.
- ``child``: the subprocess's entry point; runs ``InProcessExecutor``.

The wire format is ``app.core.worker_protocol``, shared with the control
plane. Nothing in this package but ``child`` (inside the subprocess) may
import torch.
"""
