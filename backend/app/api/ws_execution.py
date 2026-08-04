"""``/ws/execution`` — an attach/subscribe VIEW over the Run Service (#121).

The socket used to *be* the run. It created the ``asyncio.Task``, held the
only reference to it, and cancelled it on ``WebSocketDisconnect``: closing a
browser tab killed a six-hour training job, and a flaky Wi-Fi hop did the
same. #120 moved ownership into ``core.run_service``; this module is what is
left over once that is true — a projection of one run's durable event log
onto one socket.

Protocol v2
-----------
Client -> server (``action``):

``execute``   Legacy shape, unchanged, and still the only thing the canvas
              sends to start work. Internally: ``RunService.submit(...,
              options.lane="interactive", session=<this socket's state>)``
              followed by an automatic attach from cursor 0.
``attach``    ``{run_id, cursor?}`` — replay ``exec_run_events`` after
              *cursor*, then live-tail. This is what a reloaded tab sends.
``detach``    Stop forwarding. **Never cancels.** So does closing the
              socket, which is the entire point of the wave.
``cancel``    ``{run_id?}`` — the ONLY way to stop a run. Defaults to the
              attached one. (``stop`` is kept as a deprecated alias because
              a prebuilt frontend from before this change still sends it.)
``clear_cache`` Drop this socket's ``ExecutionCache``.

Server -> client: every frame of a run's event log, verbatim, as
``{"type": <event type>, **payload, "run_id", "cursor"}`` — the durable log
IS the wire vocabulary (see ``run_service``'s event-vocabulary block), so
replayed history and live progress are indistinguishable to the client,
which is what makes re-attach transparent. Plus three transport frames the
log has no opinion about: ``attached``, ``detached``, ``cancel_ack``, and
the pre-existing ``cache_cleared`` / ``error``.

One flag on one frame is not part of the log vocabulary: an
``execution_error`` carrying ``"rejected": true`` is a submit the service
REFUSED (#123 — the interactive cap, or one-run-per-session). Nothing
started and nothing failed, so a client must not treat it as a run
outcome; the run this socket is already following, if any, is named in
``run_id`` and is still going.

The replay/live boundary
------------------------
The one genuinely hard part, and the reason ``RunService.subscribe`` and
``RunStore.get_events`` are shaped the way they are. See ``_pump``.

One socket, one attachment
--------------------------
Deliberately not a multiplexer (the issue says so): a tab has one socket and
watches one run. Attaching again replaces the previous attachment — it does
not cancel the previous RUN.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from urllib.parse import urlparse

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..config import settings
from ..core.auth import (
    TOKEN_QUERY_PARAM,
    constant_time_equals,
    host_is_allowed,
    session_token,
)
from ..core.cache import ExecutionCache
from ..core.run_service import (
    LANE_INTERACTIVE,
    EVENT_RUN_STOPPED,
    InteractiveSession,
    RunService,
    RunServiceUnavailable,
    RunSubmitError,
)
from ..core.run_store import EventRecord

logger = logging.getLogger(__name__)

router = APIRouter()

# ── protocol constants ──────────────────────────────────────────────────

ACTION_EXECUTE = "execute"
ACTION_ATTACH = "attach"
ACTION_DETACH = "detach"
ACTION_CANCEL = "cancel"
#: Pre-v2 name for ``cancel``. A prebuilt frontend tarball from before this
#: release sends it, and those users update the backend first.
ACTION_STOP = "stop"
ACTION_CLEAR_CACHE = "clear_cache"

#: How many stored events one replay query pulls. Bounded so a run with
#: 50k events streams out instead of materialising in one list; the loop
#: keeps going until the log is exhausted, so this is purely a memory knob.
REPLAY_PAGE_SIZE = 500


def _frame(record: EventRecord) -> dict[str, Any]:
    """One stored event as a wire message.

    The payload is spread FIRST so the transport keys always win: ``run_id``
    is also inside the ``execution_start`` payload (same value), and a
    future event type must never be able to shadow the ``cursor`` a client
    resumes from.

    ``cursor`` rides on every frame so a client can re-attach exactly where
    it left off after a reconnect, instead of replaying history it has
    already rendered. ``run_id`` says which run a frame describes — needed
    by a client that watches several runs, and useful in a log either way.

    A client does NOT have to filter on it: ``attach`` awaits the previous
    pump's cancellation before sending ``attached``, so every frame of the
    old run precedes the acknowledgement of the new one on a single ordered
    socket. Re-attaching cannot interleave two runs.
    """
    payload = record.payload if isinstance(record.payload, dict) else {}
    return {**payload, "type": record.type, "run_id": record.run_id,
            "cursor": record.cursor}


def _error(message: str) -> dict[str, Any]:
    return {"type": "error", "error": message}


class _ExecutionSocket:
    """One WebSocket's state: its cache, its attachment, its send lock."""

    def __init__(self, ws: WebSocket) -> None:
        self._ws = ws
        # Per SOCKET, exactly as before this change — a reload always got a
        # fresh cache, and still does. It is lent to every run this socket
        # starts, which is what makes "edit one node, hit Run, only that
        # subtree re-executes" work across Run clicks.
        #
        # Its byte budget (#135) is therefore PER CONNECTION, not per
        # server: ten open editor tabs are ten budgets. /api/health reports
        # the instance count next to the total for that reason.
        self._cache = ExecutionCache()
        self._run_id: str | None = None
        self._pump: asyncio.Task[None] | None = None
        # The pump and the receive loop both send. Starlette's send is not
        # re-entrant, and interleaved frames would be malformed JSON on the
        # wire rather than merely out of order.
        self._send_lock = asyncio.Lock()

    # ── plumbing ────────────────────────────────────────────────────────

    @property
    def _service(self) -> RunService | None:
        """The process's run service, or None if the lifespan never ran.

        Read per action rather than checked at handshake time: the socket
        accepts, and answers ``clear_cache`` and unknown actions, without a
        run service. That keeps the auth/origin contract testable on its own
        and keeps this module honest about what it actually needs a service
        FOR.
        """
        return getattr(self._ws.app.state, "run_service", None)

    async def _send(self, message: dict[str, Any]) -> None:
        async with self._send_lock:
            await self._ws.send_text(json.dumps(message))

    # ── attachment ──────────────────────────────────────────────────────

    async def _replay(self, run_id: str, after_cursor: int) -> int:
        """Send every stored event after *after_cursor*; return the last one.

        Returns *after_cursor* unchanged when there is nothing to send, so
        the caller can use the result as "everything up to here has been
        delivered" without a special case.
        """
        service = self._service
        if service is None:  # pragma: no cover - shutdown race only
            return after_cursor
        while True:
            events = await service.store.get_events(
                run_id, after_cursor=after_cursor, limit=REPLAY_PAGE_SIZE)
            for record in events:
                await self._send(_frame(record))
                after_cursor = record.cursor
            if len(events) < REPLAY_PAGE_SIZE:
                return after_cursor

    async def _pump_events(self, run_id: str, cursor: int) -> None:
        """Replay from *cursor*, then live-tail. No gap, no duplicate.

        The ordering is the whole design:

        1. **Subscribe first.** From this line on, every event the run emits
           is either queued for us or counted as dropped. Subscribing after
           the replay would leave a window in which an event lands after the
           final ``get_events`` page and before the queue exists — the gap
           this issue exists to make impossible.
        2. **Then replay** the durable log from *cursor* to its end. Events
           emitted DURING that read are already safe in the queue, so the
           read has no deadline and pages freely.
        3. **Then drain the queue, skipping ``cursor <= sent``.** That
           filter is the de-duplicator: the overlap between "what the
           replay read" and "what the queue caught" is exactly the events
           whose cursors we have already sent, and cursors are gapless and
           strictly increasing (``RunStore.append_event``), so the
           comparison is total.
        4. **Re-read the tail whenever the queue reports drops.** The
           subscription is bounded and drops rather than applying
           backpressure — a stalled browser must never slow the run feeding
           it — and the store is the source of truth it is dropping in
           favour of. ``take_dropped`` resets, so each drop episode causes
           exactly one catch-up.
        5. **Re-read once more at the end.** ``close`` sets ``closed`` even
           when the end-of-run sentinel itself is dropped by a full queue,
           and the terminal event may have been dropped with it. This last
           read is what guarantees a client always receives the frame that
           tells it the run is over.

        A subscription to a run that is already finished (or that this
        process never owned) comes back closed; step 2 alone then delivers
        the complete history, which is exactly right for a tab reopened
        after the run ended.
        """
        service = self._service
        if service is None:  # pragma: no cover - shutdown race only
            return
        try:
            async with service.subscribe(run_id) as subscription:
                sent = await self._replay(run_id, cursor)
                while not (subscription.closed and subscription.queue.empty()):
                    record = await subscription.queue.get()
                    if record is None:
                        break
                    # Two independent triggers for the same catch-up. The
                    # drop counter is the expected one; the cursor gap is a
                    # backstop, because ``_emit`` documents one way an event
                    # can become durable without ever being fanned out (an
                    # append cancelled after its COMMIT). Cursors are
                    # gapless, so "skipped a number" is a complete detector
                    # whatever the cause. ``take_dropped`` is FIRST so its
                    # counter is always consumed, never short-circuited.
                    if subscription.take_dropped() or record.cursor > sent + 1:
                        sent = await self._replay(run_id, sent)
                    if record.cursor <= sent:
                        continue
                    await self._send(_frame(record))
                    sent = record.cursor
                await self._replay(run_id, sent)
        except asyncio.CancelledError:
            raise
        except Exception:
            # Two very different failures land here, and the RUN survives
            # both — this coroutine only ever reads. Telling them apart by
            # exception type is guesswork, so we let the SOCKET answer:
            # trying to report the failure either works (the socket is
            # alive, the client is now on "Running" with nothing coming, and
            # needs to hear about it) or fails in turn (the socket is what
            # went away, which is ordinary and not worth a warning).
            try:
                await self._send(_error(
                    f"lost the event stream for run {run_id}; re-attach to "
                    "resume"))
            except Exception:
                logger.debug("run %s: event pump stopped, the socket went "
                             "away", run_id, exc_info=True)
            else:
                logger.warning(
                    "run %s: event pump failed; the client is no longer "
                    "following it", run_id, exc_info=True)

    async def attach(self, run_id: str, cursor: int, status: str) -> None:
        """Point this socket at *run_id*, replacing any previous attachment."""
        await self.detach(notify=False)
        self._run_id = run_id
        await self._send({"type": "attached", "run_id": run_id,
                          "cursor": cursor, "status": status})
        self._pump = asyncio.create_task(self._pump_events(run_id, cursor),
                                         name=f"ws-attach:{run_id}")

    async def detach(self, *, notify: bool = True) -> None:
        """Stop forwarding. NEVER cancels the run — that is the whole point.

        Cancels the pump TASK, which unsubscribes and returns. The run keeps
        running, keeps writing events, and is waiting to be re-attached to.
        """
        task, self._pump = self._pump, None
        run_id, self._run_id = self._run_id, None
        if task is not None and not task.done():
            task.cancel()
            # ``asyncio.wait`` rather than ``await task``: it returns the
            # pump's CancelledError as a result instead of raising it, so
            # there is no ``except CancelledError`` here to also swallow a
            # cancellation aimed at THIS coroutine (uvicorn shutting the
            # connection down). Consuming our own cancellation would leave
            # the handler running past it, back into ``receive_text``.
            await asyncio.wait({task})
        if notify and run_id is not None:
            await self._send({"type": "detached", "run_id": run_id})

    # ── actions ─────────────────────────────────────────────────────────

    async def handle_execute(self, data: dict[str, Any]) -> None:
        """Legacy ``execute``: submit on the interactive lane, then attach.

        Every field of the pre-v2 message keeps its meaning and its default.
        The educational flags become ``options`` (persisted, so the row
        describes the run); the cache, the dirty-node hint and the module
        store become the ``InteractiveSession`` (process-local, and the
        reason canvas semantics survive the move to a server-owned run).

        A run already attached here is DETACHED, not cancelled: this socket
        stops watching it and starts watching the new one. Cancelling would
        put back the exact behaviour #120 removed.

        Since #123 that only arises once the previous run has ENDED. While
        one is still in flight the service refuses the submit outright —
        two runs behind one socket's ``ExecutionCache`` would serve each
        other half-built tensors, the hazard #121 disclosed and left to the
        canvas's disabled Run button. The refusal arrives as an
        ``execution_error`` flagged ``rejected`` and this socket stays
        attached to the run that is actually going, which is the more useful
        of the two.
        """
        service = self._service
        if service is None:
            await self._send(_error("run service not initialised"))
            return

        graph = {
            "nodes": data.get("nodes", []),
            "edges": data.get("edges", []),
            # Graph-embedded presets (#84): portable graphs carry their own
            # presets[]; consulted when the local registry lacks one.
            "presets": data.get("presets", []),
            # Subgraph definitions (core#137) travel with the graph the same
            # way -- they are local to it, so there is no registry to fall
            # back on and the canvas must always send them.
            "subgraphs": data.get("subgraphs", []),
        }
        options = {
            "lane": LANE_INTERACTIVE,
            "device": data.get("device") or "cpu",
            "record_outputs": bool(data.get("record_outputs", False)),
            "error_mode": data.get("error_mode", "fail_fast"),
            "max_retries": data.get("max_retries", 0),
            # Reproducibility (#134). Passed through UNVALIDATED on purpose:
            # ``normalize_options`` owns the seed's type and range, and
            # coercing here would turn a client bug ("seed": "42") into a
            # silently different run instead of the RunSubmitError the
            # canvas already knows how to show.
            "seed": data.get("seed"),
            "deterministic": bool(data.get("deterministic", False)),
            # Educational feature flags. The defaults below are the pre-v2
            # ones verbatim, including weights_persistent defaulting to
            # TRUE here (the canvas keeps its weights) where the service's
            # own default is False (a server-owned run is isolated).
            "verbose": bool(data.get("verbose_mode", False)),
            "graph_id": str(data.get("graph_id", "") or ""),
            "weights_persistent": bool(data.get("weights_persistent", True)),
            "backward_mode": bool(data.get("backward_mode", False)),
            "auto_backward": bool(data.get("auto_backward", False)),
        }
        changed = data.get("changed_nodes")
        session = InteractiveSession(
            cache=self._cache,
            changed_nodes=tuple(changed) if isinstance(changed, list) else (),
            node_state_store=getattr(self._ws.app.state, "node_state_store",
                                     None),
        )

        try:
            result = await service.submit(graph, options=options,
                                          session=session)
        except RunSubmitError as exc:
            # The pre-v2 handler reported a graph the engine refused as
            # ``execution_error``; an envelope it refuses is the same class
            # of thing to a user, so it reads the same on the canvas.
            await self._send({"type": "execution_error", "error": str(exc)})
            return
        except RunServiceUnavailable as exc:
            # A REFUSAL, not a failure — and the difference is load-bearing
            # on the canvas. Nothing started, nothing failed, and whatever
            # this socket was already running is still running: the common
            # case is #123's one-run-per-session rule turning down a second
            # click. A plain ``execution_error`` puts the tab into `error`,
            # which re-enables Run and disables Stop while the first run is
            # mid-epoch — the user loses the ability to stop the very run
            # they are watching.
            #
            # A FLAG rather than a new frame type, so a client that has
            # never heard of it still shows the message (today's behaviour)
            # instead of silently doing nothing with an unknown type. The
            # attached run's id rides along so a client can confirm the
            # refusal is about the run it is already following.
            await self._send({"type": "execution_error", "error": str(exc),
                              "rejected": True, "run_id": self._run_id})
            return
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # ``submit`` writes a row before it starts anything, so a sick
            # database (or provenance capture) fails HERE, on the receive
            # loop — where the pre-v2 handler could not fail because it did
            # its work inside a task. Letting it escape would tear the
            # socket down with a 1011 and leave the canvas spinning; the
            # user gets the same red banner any other failed run gets, and
            # the socket survives to be used again.
            logger.exception("execute failed before the run could start")
            await self._send({"type": "execution_error", "error": str(exc)})
            return

        # From cursor 0: the run was created microseconds ago, so this is
        # both "the whole log" and "nothing yet" — whichever the run task
        # has managed to write by the time the pump reads it.
        await self.attach(result.run_id, 0, result.status)

    async def handle_attach(self, data: dict[str, Any]) -> None:
        service = self._service
        if service is None:
            await self._send(_error("run service not initialised"))
            return
        run_id = data.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            await self._send(_error("attach requires a run_id"))
            return
        cursor = data.get("cursor", 0)
        if isinstance(cursor, bool) or not isinstance(cursor, int) or cursor < 0:
            await self._send(_error("attach cursor must be an integer >= 0"))
            return
        record = await service.store.get_run(run_id)
        if record is None:
            await self._send(_error(f"run '{run_id}' not found"))
            return
        # A cursor past the end of the log is always a client bug — an
        # off-by-one, or a cursor kept from a DIFFERENT run. Accepting it
        # would attach successfully and then deliver nothing, which looks
        # exactly like a healthy attach to a quiet run: the worst possible
        # failure mode for a client whose whole job is following along.
        latest = await service.store.latest_cursor(run_id)
        if cursor > latest:
            await self._send(_error(
                f"attach cursor {cursor} is past the end of run '{run_id}' "
                f"(latest cursor {latest})"))
            return
        await self.attach(run_id, cursor, record.status)

    async def handle_cancel(self, data: dict[str, Any]) -> None:
        """The only path that stops a run.

        Answers with ``cancel_ack`` immediately — cancellation is
        cooperative (a node in a 90-second epoch finishes that epoch first),
        so the acknowledgement and the ``execution_stopped`` frame are
        genuinely different moments and a UI wants both.

        When there is nothing to cancel — no attachment, or a run this
        server has never heard of — it still emits ``execution_stopped`` so
        a canvas stuck on "Running" unsticks. It does NOT emit one for a run
        that merely finished already: that run's real terminal frame is in
        the log, and inventing a second one would overwrite "completed" with
        "cancelled" in the UI.
        """
        service = self._service
        run_id = data.get("run_id") or self._run_id
        if service is None or not isinstance(run_id, str) or not run_id:
            await self._send({"type": EVENT_RUN_STOPPED,
                              "reason": "not_running"})
            return
        outcome = await service.cancel(run_id)
        if outcome is None:
            await self._send({"type": EVENT_RUN_STOPPED,
                              "reason": "not_running", "run_id": run_id})
            return
        await self._send({"type": "cancel_ack", "run_id": outcome.run_id,
                          "status": outcome.status,
                          "cancelled": outcome.cancelled})

    async def handle_clear_cache(self) -> None:
        self._cache.clear()
        await self._send({"type": "cache_cleared"})

    # ── the loop ────────────────────────────────────────────────────────

    async def serve(self) -> None:
        """Read actions until the client goes away.

        Every handler runs INLINE here, unlike the pre-v2 loop which pushed
        the whole run into a task, so one failing action must not be able to
        end the connection. Anything unexpected is answered and the loop
        continues — a socket that dies on a bad message would take a healthy
        attachment (and the user's view of a running job) with it.
        """
        while True:
            raw = await self._ws.receive_text()
            try:
                data = json.loads(raw)
            except ValueError:
                await self._send(_error("malformed message: expected JSON"))
                continue
            if not isinstance(data, dict):
                await self._send(_error("malformed message: expected an object"))
                continue

            action = data.get("action")
            try:
                if action == ACTION_EXECUTE:
                    await self.handle_execute(data)
                elif action == ACTION_ATTACH:
                    await self.handle_attach(data)
                elif action == ACTION_DETACH:
                    await self.detach()
                elif action in (ACTION_CANCEL, ACTION_STOP):
                    await self.handle_cancel(data)
                elif action == ACTION_CLEAR_CACHE:
                    await self.handle_clear_cache()
                else:
                    await self._send(_error(f"Unknown action: {action}"))
            except (asyncio.CancelledError, WebSocketDisconnect):
                raise
            except Exception as exc:
                logger.exception("ws action %r failed", action)
                await self._send(_error(f"{action} failed: {exc}"))


@router.websocket("/ws/execution")
async def websocket_execution(ws: WebSocket):
    # Host header check (DNS rebinding defence). Browsers tricked into
    # resolving attacker.com → 127.0.0.1 still send Host: attacker.com;
    # we reject those before they can even start a graph.
    request_host = ws.headers.get("host", "")
    if not host_is_allowed(request_host):
        await ws.close(code=4003, reason="Host not allowed")
        return

    # Origin policy (browser layer): same as the HTTP middleware.
    origin = ws.headers.get("origin")
    if origin:
        origin_netloc = urlparse(origin).netloc
        if origin_netloc != request_host:
            allowed = {urlparse(o).netloc for o in settings.CORS_ORIGINS}
            if origin_netloc not in allowed:
                await ws.close(code=4003, reason="Origin not allowed")
                return

    # Session-token check. Browsers cannot set custom headers on WebSocket
    # handshakes, so we accept the token via ``?token=`` query parameter.
    # Header is also accepted to keep server-to-server clients (tests, future
    # CLI) consistent with the REST API.
    provided = (
        ws.headers.get("x-codefyui-token")
        or ws.query_params.get(TOKEN_QUERY_PARAM)
    )
    if not constant_time_equals(provided, session_token()):
        await ws.close(code=4401, reason="Missing or invalid session token")
        return

    await ws.accept()

    socket = _ExecutionSocket(ws)
    try:
        await socket.serve()
    except WebSocketDisconnect:
        pass
    finally:
        # UNSUBSCRIBE ONLY. The headline behaviour of the whole wave lives
        # in this one line not calling ``service.cancel``: the tab is gone,
        # the training is not.
        await socket.detach(notify=False)

