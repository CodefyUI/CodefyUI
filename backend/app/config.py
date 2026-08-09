import os
from pathlib import Path
from platformdirs import user_data_dir
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings

# Let the handful of ops not yet implemented for Apple MPS fall back to CPU
# instead of hard-crashing the graph run. Set here (before torch is imported
# anywhere) so it applies to the server, the CLI runner, and tests alike.
# ``setdefault`` keeps an explicit external value (e.g. a test pinning "0" to
# assert native MPS support) authoritative.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def _user_data_root() -> Path:
    """Resolve the per-user data root (lockfile, downloaded plugins, session token).

    Honors ``CODEFYUI_USER_DATA_DIR`` so a dev clone using
    ``scripts/dev.py`` can keep its lockfile in ``.codefyui_dev/`` inside
    the repo, isolated from the global ``cdui`` install at
    ``%LOCALAPPDATA%\\codefyui``. ``plugin_loader.plugins_user_root`` and
    ``auth._token_dir`` honor the same variable so plugin install, the
    server, and the hot-reload token all stay in sync.
    """
    override = os.environ.get("CODEFYUI_USER_DATA_DIR")
    return Path(override) if override else Path(user_data_dir("codefyui", appauthor=False))


def _default_plugins_user_dir() -> Path:
    return _user_data_root() / "plugins"


class Settings(BaseSettings):
    APP_NAME: str = "CodefyUI"
    DEBUG: bool = False
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    CORS_ORIGINS: list[str] = ["http://localhost:5173", "http://localhost:5174", "http://localhost:3000"]
    # Largest FILE accepted by the four upload routes. The multipart body
    # carrying it is bounded a shade higher (this plus an envelope allowance)
    # by core.body_limit; the routes themselves enforce this exact number
    # against the decoded file. See body_limit's module docstring.
    MAX_UPLOAD_SIZE: int = 500 * 1024 * 1024  # 500 MB
    # Default request-body ceiling for EVERY route that is not an upload
    # (-> 413). Enforced by core.body_limit, which counts bytes as they arrive
    # on the ASGI receive channel, so it holds for chunked requests that
    # declare no Content-Length as well. Until core#265 this was three
    # hand-rolled Content-Length comparisons on three routes, each of which a
    # chunked client walked straight past.
    # 64 MB comfortably covers a handful of base64 image inputs.
    # Env-overridable as CODEFYUI_MAX_RUN_BODY_BYTES.
    MAX_RUN_BODY_BYTES: int = 64 * 1024 * 1024  # 64 MB
    # The WebSocket twin of the number above, for `WS /ws/execution` — which
    # carries a whole graph in one text frame and so needs a ceiling of its
    # own. It is NOT enforced here: a WebSocket has no request body, so
    # core.body_limit skips its scope deliberately, and by the time
    # ``receive_text()`` returns the message is already assembled. The only
    # place a frame can be refused before it is buffered is the transport,
    # so this value is handed to uvicorn as ``--ws-max-size`` by
    # ``scripts/dev.py`` (both `cdui start` and `cdui dev`), and the
    # ``websockets`` library enforces it WHILE assembling fragments — the
    # same count-as-it-arrives property core.body_limit gives HTTP.
    #
    # Why it exists at all (core#274): uvicorn's own ``ws_max_size`` default
    # is 16 MB, so before this setting the canvas socket was capped four
    # times STRICTER than the HTTP routes, by a number nobody in this repo
    # chose, that no launch path passed and no user could tune. A graph
    # between 16 and 64 MB was accepted by POST /api/graph/run/{name} and
    # refused by the socket the editor actually uses.
    #
    # It therefore DEFAULTS TO MAX_RUN_BODY_BYTES rather than to a literal
    # of its own (see _ws_cap_follows_body_cap): one graph ceiling, both
    # transports, and raising CODEFYUI_MAX_RUN_BODY_BYTES moves both. Set
    # CODEFYUI_WS_MAX_MESSAGE_BYTES to split them apart deliberately.
    #
    # NOTE this only binds on the launch paths CodefyUI controls. Running
    # `uvicorn app.main:app` by hand gets uvicorn's 16 MB back unless you
    # pass --ws-max-size yourself; the deployment guide says so.
    WS_MAX_MESSAGE_BYTES: int = 64 * 1024 * 1024  # 64 MB

    # ── Stage-2 publish storage + limits (spec Section 8) ──────────────
    # All env-overridable via the CODEFYUI_ prefix like everything else.
    # SQLite file for published apps / API keys / run records; sibling of
    # GRAPHS_DIR so a dev clone and a global install stay isolated.
    DB_PATH: Path = Path(__file__).parent.parent / "data" / "codefyui.db"
    MAX_IMAGE_PIXELS: int = 25_000_000  # per-image decode budget (Decision H2)
    RUN_IO_CAP_BYTES: int = 64 * 1024   # per-field runs.inputs_json/outputs_json cap
    RUNS_RETENTION_DAYS: int = 0        # 0 = keep forever (default); >0 prunes loudly
    # ── Run Service retention + limits (#120) ──────────────────────────
    # Keep-last-N over the exec_runs table (graph runs), applied at startup
    # and after every run reaches a terminal state. Independent of
    # RUNS_RETENTION_DAYS above, which is age-based and covers the
    # unrelated publish `runs` table. Active runs are never deleted and
    # still occupy slots, so this is a straightforward cap on the table.
    #
    # MIND THE INVERTED ZERO, it is deliberate and not a typo: for the
    # age-based knob above, 0 means "no cutoff" -> keep forever; for this
    # count-based one, 0 is a real count -> keep zero finished runs, i.e.
    # delete them all. Disabling THIS one takes a NEGATIVE value. Two
    # different policies over two different tables; neither borrows the
    # other's sentinel.
    RUN_RETENTION_KEEP_LAST: int = 200
    # Per-event ceiling on what exec_run_events stores (and fans out).
    # Retention counts runs, not bytes, so without this one graph emitting
    # base64 PNGs on every node -- or a training payload carrying a
    # per-batch array that grows each epoch -- can put hundreds of MB into
    # the database for a single run, with record_outputs OFF. Over-cap
    # output entries are replaced by an elision marker that keeps
    # output_kind and port so a UI still knows something was there. 128 KB
    # is twice the publish path's RUN_IO_CAP_BYTES because a node_status
    # payload is a whole node's summary (several ports, each embedding up
    # to 256 values) rather than one API field; measured honest payloads
    # sit well under 20 KB, so the headroom means elision only ever fires
    # on genuinely unbounded content. <= 0 disables the cap.
    RUN_EVENT_PAYLOAD_CAP_BYTES: int = 128 * 1024
    # Byte budget for ONE replay page. RUN_EVENT_PAYLOAD_CAP_BYTES bounds a
    # single payload; this bounds the page, so a `limit` cannot multiply the
    # two into a response nobody asked for. Read by GET /api/runs/{id}/events
    # (which stops early and lets the client resume from the returned
    # cursor) — a short page is already normal, so stopping early costs one
    # extra round trip and nothing else. Far above any honest page:
    # ordinary events are a couple of KB.
    RUN_EVENTS_RESPONSE_CAP_BYTES: int = 4 * 1024 * 1024
    # ── Run queue admission (#123) ─────────────────────────────────────
    # How many QUEUED-lane runs may execute at once on one device. The
    # queue key is the RESOLVED device string ("cpu", "cuda:0", "mps"), so
    # every accelerator gets its own FIFO and a six-hour CUDA job never
    # blocks a CPU run.
    #
    # One is the honest GPU default: a second concurrent run on the same
    # card competes for the same VRAM, and the usual outcome of guessing
    # otherwise is a CUDA OOM forty minutes in rather than a slower run.
    # CPU gets two because the failure mode there is contention, not death
    # — and a machine with cores to spare should use them. Note this is
    # RUNS per device, orthogonal to MAX_PARALLEL_NODES below (nodes within
    # one run); the two multiply.
    RUN_QUEUE_MAX_CONCURRENT_GPU: int = 1
    RUN_QUEUE_MAX_CONCURRENT_CPU: int = 2
    # Per-key overrides, e.g. "cuda:0=2,cuda:1=1,cpu=8". A str, not a
    # dict[str, int]: pydantic mapping env vars demand JSON-in-env quote
    # hell, the same reason EXTRA_ALLOWED_HOSTS below is a string. Split on
    # the LAST '=' so a key may contain a colon; malformed entries are
    # ignored with a warning rather than failing startup.
    RUN_QUEUE_MAX_CONCURRENT: str = ""
    # Hard ceiling on concurrent INTERACTIVE-lane runs (the canvas). That
    # lane bypasses the per-device FIFO on purpose — a classroom demo must
    # not wait behind a training job — so this is the only thing standing
    # between "every open tab clicks Run" and a VRAM exhaustion. A submit
    # over the cap is REFUSED (503), never parked: the canvas has a live
    # user in front of it who can retry, and a silently queued click looks
    # exactly like a hang.
    RUN_INTERACTIVE_MAX_CONCURRENT: int = 2
    # Comma-separated extra Host-whitelist entries, e.g.
    # "192.168.1.20:8000,mybox:8000". A str, not list[str]: pydantic list
    # env vars demand JSON-in-env quote hell; split in init_allowed_hosts.
    EXTRA_ALLOWED_HOSTS: str = ""

    # ── Port statistics (#129) ─────────────────────────────────────────
    # GET /api/execution/outputs/{run}/{node}/{port}/stats summarises a
    # captured value instead of shipping it. Above this element count the
    # DISTRIBUTION stats (mean/std/quantiles/histogram) are computed from a
    # deterministic seeded sample and the response is marked
    # `"sampled": true`. Note what this does NOT cover: count, min, max,
    # nan_count, inf_count, zero_frac and integer class balance stay exact
    # at every size, because they are O(n) reductions with no sort — and a
    # sampled NaN count is not an answer anybody can debug with.
    #
    # 4M is a measured number, not a round one: quantiles need a sort, and
    # torch.sort runs ~0.18us/element (0.7s at 4M, 3s at 16M, 9s at 50M). So
    # 4M is the largest tensor this can summarise EXACTLY while still feeling
    # like a click. Raising it buys exactness and pays in seconds, linearly.
    # (torch.quantile additionally refuses inputs over 16.7M, but that is a
    # corroborating observation, not the binding constraint — this module
    # sorts and interpolates itself precisely so it is not bound by it.)
    STATS_SAMPLE_THRESHOLD: int = 4_000_000
    # Elements in that sample. 1M sorts in ~0.17s and pins a quantile to
    # about 0.1% in probability space — far finer than a 64-bar histogram
    # can draw, so the chart is never the thing the sample limits.
    STATS_SAMPLE_SIZE: int = 1_000_000
    # Byte budget for the computed-stats LRU. BYTES, not entries: one payload
    # ranges from ~2 KB (a 64-bin histogram) to ~50 KB (a wide tabular
    # summary), so an entry count would be a limit in name only. 8 MB holds
    # hundreds of ports; a run's worth of inspection never approaches it.
    #
    # Measured as SERIALIZED size, which is what the response costs and what
    # #135 will bound everywhere. Resident size is some multiple of it —
    # Python dicts, str keys and boxed floats run roughly 5-10x — so read this
    # as a cap on payload volume, not as an RSS guarantee.
    STATS_CACHE_MAX_BYTES: int = 8 * 1024 * 1024

    # ── In-memory store budgets (#135) ─────────────────────────────────
    # The three stores that hold tensors between runs used to be bounded by
    # ENTRY COUNT, which on a single 32 GB card is a limit in name only:
    # 256 cached node outputs is either 40 MB of scalars or 200 GB of
    # feature maps, and a count cannot tell those apart. Each store now
    # carries a byte budget as well, evicting LRU by accumulated tensor
    # storage (``core.memory_budget``), and reports its usage in
    # /api/health.
    #
    # MEGABYTES, not bytes, unlike every other cap above. These are numbers
    # a user tunes against the RAM or VRAM in their machine — "1024" reads
    # as a gigabyte, "1073741824" reads as a typo waiting to happen. Both
    # limits apply: whichever binds first evicts. 0 disables the BYTE
    # budget for that store and leaves the count in charge.
    #
    # The defaults total 4 GB, which is deliberately generous for a
    # workstation and deliberately finite: the point is that a long session
    # cannot climb without limit, not that it should feel constrained.
    #
    # Note the execution cache is per WEBSOCKET CONNECTION, so this budget
    # is per connection; the other two are one per server. /api/health
    # reports the instance count alongside the total for exactly that
    # reason.
    EXECUTION_CACHE_MAX_MB: int = 1024
    RUN_OUTPUT_STORE_MAX_MB: int = 2048
    NODE_STATE_STORE_MAX_MB: int = 1024

    NODES_DIR: Path = Path(__file__).parent / "nodes"
    CUSTOM_NODES_DIR: Path = Path(__file__).parent / "custom_nodes"
    GRAPHS_DIR: Path = Path(__file__).parent.parent / "data" / "graphs"
    PRESETS_DIR: Path = Path(__file__).parent / "presets"
    MODELS_DIR: Path = Path(__file__).parent.parent / "data" / "models"
    IMAGES_DIR: Path = Path(__file__).parent.parent / "data" / "images"
    # Uploads for DATA_FILE params (CSVReader et al). Separate from IMAGES_DIR
    # so each kind keeps its own extension whitelist and its own dropdown.
    DATA_FILES_DIR: Path = Path(__file__).parent.parent / "data" / "files"
    EXAMPLES_DIR: Path = Path(__file__).parent.parent.parent / "examples"

    # ── Project directory (spec 7.1) ───────────────────────────────────
    # Set by `cdui start|dev --project <dir>` (CODEFYUI_PROJECT_DIR=<abs>).
    # None = non-project mode (byte-for-byte legacy behavior). When set,
    # the graph/asset roots derive from it UNLESS the specific root was
    # explicitly provided (env/init) — see _derive_project_roots.
    PROJECT_DIR: Path | None = None

    # First-party plugin packs shipped with the repo (<REPO>/plugins/).
    PLUGINS_BUILTIN_DIR: Path = Path(__file__).parent.parent.parent / "plugins"
    # Third-party downloads + lockfile. Resolved per-instance so the
    # CODEFYUI_USER_DATA_DIR env var picked up by dev-mode tooling actually
    # flows through to the server. Without default_factory the snapshot
    # would freeze at module-import time, before scripts/dev.py sets the
    # env var.
    PLUGINS_USER_DIR: Path = Field(default_factory=_default_plugins_user_dir)

    LOG_LEVEL: str = "INFO"
    LOG_DIR: Path | None = None
    LOG_JSON: bool = False

    # Node-level fan-out WITHIN one run: graph_engine builds an
    # asyncio.Semaphore(context.max_workers) over the ready set, and
    # RunService fills that field from here. Live, not vestigial — #123
    # looked at deleting it as dead and found the wiring
    # (config -> RunService._start -> ExecutionContext.max_workers ->
    # graph_engine). Deliberately NOT the same knob as
    # RUN_QUEUE_MAX_CONCURRENT_* above: that one bounds RUNS per device,
    # this one bounds NODES inside a single run.
    MAX_PARALLEL_NODES: int = 4

    @model_validator(mode="after")
    def _ws_cap_follows_body_cap(self) -> "Settings":
        """An unset WS ceiling tracks MAX_RUN_BODY_BYTES (core#274).

        The two caps bound the same thing — one graph — over two transports,
        so the default has to be the same number or the pair drifts the
        moment an operator raises one. ``model_fields_set`` lists ONLY fields
        provided from env/init (never class defaults), so an explicit
        ``CODEFYUI_WS_MAX_MESSAGE_BYTES`` still wins and the two can be
        split apart on purpose. Same precedence idiom as
        ``_derive_project_roots`` below.
        """
        if "WS_MAX_MESSAGE_BYTES" not in self.model_fields_set:
            self.WS_MAX_MESSAGE_BYTES = self.MAX_RUN_BODY_BYTES
        return self

    @model_validator(mode="after")
    def _derive_project_roots(self) -> "Settings":
        """Project mode repoints graphs/assets roots (spec 7.1).

        Precedence: an explicitly-set root (CODEFYUI_GRAPHS_DIR etc., or an
        init kwarg) beats project derivation — detected via model_fields_set,
        which lists ONLY fields provided from env/init, never class defaults.
        DB_PATH / CUSTOM_NODES_DIR are intentionally NOT derived (install-
        global in v1). No module-level settings caching exists, so setting
        these here (before any node reads settings at call time) is safe.
        """
        if self.PROJECT_DIR is None:
            return self
        explicit = set(self.model_fields_set)
        proj = self.PROJECT_DIR.resolve()
        self.PROJECT_DIR = proj
        assets = proj / "assets"
        if "GRAPHS_DIR" not in explicit:
            self.GRAPHS_DIR = proj / "graphs"
        if "IMAGES_DIR" not in explicit:
            self.IMAGES_DIR = assets / "images"
        if "MODELS_DIR" not in explicit:
            self.MODELS_DIR = assets / "models"
        return self

    @property
    def LAYOUT_DIR(self) -> Path | None:
        """Server-side layout dir (<proj>/layout) in project mode, else None.

        Only read by the project-mode save/load split (Task 3); non-project
        mode never touches it. A property (not a field) so it always tracks
        PROJECT_DIR and needs no precedence handling of its own.
        """
        if self.PROJECT_DIR is None:
            return None
        return self.PROJECT_DIR / "layout"

    model_config = {"env_prefix": "CODEFYUI_"}


settings = Settings()
