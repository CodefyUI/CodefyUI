"""Shared pytest fixtures for CodefyUI backend tests."""

import hashlib
import importlib.machinery
import sys
import tempfile
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from httpx import ASGITransport, AsyncClient

# Make scripts/ importable so CLI tests can `import plugins`.
_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from app.config import settings
from app.core.auth import TOKEN_HEADER, init_allowed_hosts, session_token
from app.core.node_base import BaseNode, DataType, PortDefinition
from app.core.node_registry import NodeRegistry, registry
from app.core.plugin_loader import install_plugin_finder, purge_all_plugin_modules
from app.core.preset_registry import preset_registry
from app.main import app

# Captured before the redirect below -- test_config_stage2.py asserts
# against this exact production value (see the fixture near the bottom of
# this file that hands it back for the duration of those tests only).
_DEFAULT_DB_PATH = settings.DB_PATH

# DB isolation: every test run's SQLite DB lives in a temp dir, never in
# backend/data/ (lifespan-driving TestClient tests would otherwise create a
# real codefyui.db there). Module-level on purpose: conftest import runs
# before any hook or fixture, so there is no ordering race.
settings.DB_PATH = Path(tempfile.mkdtemp(prefix="codefyui-test-db-")) / "codefyui-test.db"

# Tests use ``base_url="http://127.0.0.1:8000"`` which the production Host
# whitelist already accepts, but seed it explicitly here so tests don't rely
# on lifespan-time initialisation (lifespan runs once per app instance and
# ASGITransport doesn't always go through it).
init_allowed_hosts(settings.HOST, settings.PORT)

# In-repo plugin packs the test suite loads. One tuple, three consumers (the
# namespace install below and the two registry-discovery paths further down):
# a pack missing from any one of them fails in a different, confusing way —
# an ImportError at collection, or a node type the graph engine cannot find.
_BUILTIN_TEST_PACKS = ("foundations", "deep", "rl", "stats")

# Register those packs in the synthetic `cdui_plugins` namespace AT CONFTEST
# IMPORT TIME, before any test_*.py module is collected. Pack node tests import
# from `cdui_plugins.<pack>.nodes.*` during pytest's collection pass, which runs
# after conftest is imported, so the namespace must exist by then.
_REPO_ROOT = Path(__file__).resolve().parents[2]
purge_all_plugin_modules()
install_plugin_finder(
    builtin_root=_REPO_ROOT / "plugins",
    user_root=_REPO_ROOT / "_phantom_user_root_for_tests",  # never read
    lockfile={
        "schema": 1,
        "plugins": {
            pack: {"source_kind": "builtin", "source": pack}
            for pack in _BUILTIN_TEST_PACKS
        },
    },
)


def _discover_builtin_packs() -> None:
    """Register every in-repo pack's nodes into the global registry.

    ``force_reload`` is left off deliberately: re-registering hands back the
    same class objects, so a test holding an ``is`` comparison against one
    still passes.
    """
    for plugin_id in _BUILTIN_TEST_PACKS:
        plugin_nodes = _REPO_ROOT / "plugins" / plugin_id / "nodes"
        if plugin_nodes.exists():
            # ``plugin_id`` is passed rather than left to be derived from the
            # package name: none of these packs has a hyphen in its id, so
            # both routes agree today, but the suite should exercise the same
            # call production makes.
            registry.discover(
                plugin_nodes,
                f"cdui_plugins.{plugin_id}.nodes",
                plugin_id=plugin_id,
            )


def _packs_missing_from_registry() -> bool:
    """True when a pack this suite needs has no nodes registered.

    ``rediscover_all`` — which ``POST /api/plugins/reload`` runs — rebuilds the
    registry from the machine's REAL lockfile, not from the synthetic one this
    file installs. A pack that is not installed on the developer's machine (or
    on CI, where the lockfile is empty) is therefore silently dropped, and
    every later test that resolves a node BY TYPE gets "Unknown node type".
    """
    registered = {key.split(":", 1)[0] for key in registry._nodes if ":" in key}
    return any(pack not in registered for pack in _BUILTIN_TEST_PACKS)


@pytest.fixture(autouse=True)
def _config_tests_see_default_db_path(request, monkeypatch):
    """test_config_stage2.py asserts the untouched production DB_PATH; hand
    it back for the duration of those tests only.

    A plain fixture, not a ``pytest_runtest_setup``/``teardown`` hook -- it
    only ever runs as part of normal per-item fixture resolution, which
    pytest guarantees happens before that item's test body regardless of
    collection order. ``monkeypatch`` restores the isolated path afterward,
    so no hand-rolled restore bookkeeping is needed.
    """
    if "test_config_stage2" in str(request.node.fspath):
        monkeypatch.setattr(settings, "DB_PATH", _DEFAULT_DB_PATH)
    yield


class _TestSourceNode(BaseNode):
    """Lightweight source node for tests -- no required inputs, no torch."""
    NODE_NAME = "_TestSource"
    CATEGORY = "Test"
    DESCRIPTION = "Emits a constant value"

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return []

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [PortDefinition(name="value", data_type=DataType.ANY)]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        return {"value": params.get("val", "test")}


@pytest.fixture(scope="session", autouse=True)
def registry_with_nodes() -> NodeRegistry:
    """Discover all nodes once per test session, including chapter plugins."""
    if len(registry.nodes) == 0:
        registry.discover(settings.NODES_DIR, "app.nodes")
        registry.discover(settings.CUSTOM_NODES_DIR, "app.custom_nodes")
        _discover_builtin_packs()
        preset_registry.discover(settings.PRESETS_DIR, registry)
    registry._nodes["_TestSource"] = _TestSourceNode
    return registry


@pytest.fixture(autouse=True)
def _ensure_registry_intact(registry_with_nodes):
    """Run before every test: repopulate the registry if a prior test cleared it.

    ``POST /api/plugins/reload`` (and any test that calls ``rediscover_all``)
    clears every registry entry, including the manually-injected
    ``_TestSource`` synthetic node and the built-ins. Without this safety net,
    ws-execution tests that follow such a test see "Unknown node type".

    The plugin packs are checked separately from the built-ins because a
    reload does not lose them the same way: it re-registers the built-ins from
    ``NODES_DIR`` and then the packs from the real lockfile, so ``Start`` comes
    back while an uninstalled pack does not. Testing only for ``Start`` made
    the suite pass or fail on collection order — a pack test sorting before
    ``test_plugin_api`` was fine, one sorting after was not.
    """
    if "_TestSource" not in registry._nodes:
        registry._nodes["_TestSource"] = _TestSourceNode
    if "Start" not in registry._nodes:
        # Wholesale rebuild — registry was nuked by an earlier reload.
        registry.discover(settings.NODES_DIR, "app.nodes")
        registry.discover(settings.CUSTOM_NODES_DIR, "app.custom_nodes")
        registry._nodes["_TestSource"] = _TestSourceNode
    if _packs_missing_from_registry():
        _discover_builtin_packs()
    yield


@pytest.fixture
async def test_client():
    """Async HTTP client connected to the FastAPI app via ASGI transport.

    The base URL is chosen so the ``Host`` header (set automatically by
    httpx) matches the production whitelist seeded above. The session
    token is also pre-attached so tests don't need to know about it.
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport,
        base_url=f"http://127.0.0.1:{settings.PORT}",
        headers={TOKEN_HEADER: session_token()},
    ) as client:
        yield client


@pytest.fixture
def sample_graph():
    """A minimal valid graph: Start -> _TestSource -> Print."""
    return {
        "nodes": [
            {"id": "start", "type": "Start", "position": {"x": -150, "y": 0}, "data": {"params": {}}},
            {"id": "1", "type": "_TestSource", "position": {"x": 0, "y": 0}, "data": {"params": {}}},
            {"id": "2", "type": "Print", "position": {"x": 200, "y": 0}, "data": {"params": {"label": "second"}}},
        ],
        "edges": [
            {"id": "et", "source": "start", "target": "1", "sourceHandle": "trigger", "type": "trigger"},
            {"id": "e1", "source": "1", "target": "2", "sourceHandle": "value", "targetHandle": "value"},
        ],
        "name": "test-graph",
        "description": "A test graph",
    }


@pytest.fixture
async def app_db(tmp_path):
    """Per-test Stage-2 Database on app.state (+ empty app_locks).

    PER TEST, never module/session-scoped: asyncio locks bind to the
    running event loop on first use. The lifespan does not run under
    httpx ASGITransport, so tests set app.state directly (the
    run_output_store precedent in test_api_graph_run.py).
    """
    from app.core.db import Database

    db = Database(tmp_path / "codefyui.db")
    db.connect()
    app.state.db = db
    app.state.app_locks = {}
    try:
        yield db
    finally:
        db.close()
        if hasattr(app.state, "db"):
            delattr(app.state, "db")
        if hasattr(app.state, "app_locks"):
            delattr(app.state, "app_locks")


# ── offline sentence-transformers ────────────────────────────────────────
#
# The Sentence embeddings pack is 90 MB of model on top of a pip install that
# drags in transformers. No test may download either, so the nodes that use
# it are tested against the fake below: same call shape, arithmetic instead
# of a neural network.

#: Width of the fake's embeddings. Narrow enough that a failed assertion
#: prints a readable row. It must stay <= 64: one digest byte becomes one
#: dimension below, and blake2b will not produce more than 64 of them.
_FAKE_EMBED_DIM = 32

#: What a character weighs against a word. Tuned, not guessed: sweeping it
#: over a dozen English and a dozen Chinese sentences, 0.5 is where "shares
#: at least one word" stays above "shares nothing" for English (worst pair
#: 0.64 against a 0.49 ceiling) while "shares most of its characters" stays
#: above it for Chinese (0.32 against 0.27). Lower and Chinese inverts --
#: every CJK sentence is ONE whitespace token, so characters are the only
#: signal there; higher and English inverts, because unrelated English
#: sentences share nearly every letter of the alphabet.
_FAKE_CHAR_WEIGHT = 0.5


def _hashed_direction(piece: str, dim: int) -> np.ndarray:
    """The fixed unit vector that stands for one word or one character."""
    digest = hashlib.blake2b(piece.encode("utf-8"), digest_size=dim).digest()
    # Bytes run 0..255; centring them points the direction anywhere instead
    # of into the all-positive corner, where everything looks alike.
    vector = np.frombuffer(digest, dtype=np.uint8).astype(np.float64) - 127.5
    return vector / np.linalg.norm(vector)


def _hashed_bag(text: str, dim: int) -> np.ndarray:
    """A deterministic bag-of-(words + characters) vector for *text*.

    Each distinct piece gets a pseudo-random direction out of its digest and
    the text is their weighted sum. Two decisions worth the words:

    * blake2b rather than ``hash()``, which is salted per process -- two runs
      of the suite have to agree, or a cached-embedding test passes and fails
      at random;
    * a whole direction per piece rather than one bucket per piece. In 32
      buckets, two unrelated words collide about once in thirty; a single
      collision would make two unrelated texts look nearly identical, which
      is exactly the assertion a node test is about to make.

    Distinct pieces, not a multiset: counting repeats would rank two English
    sentences by how often they say "the".

    Coarse, and deliberately so: 32 dimensions puts unrelated texts around
    0.2-0.5 cosine rather than at 0. Assert on the ORDER of similarities,
    never on an absolute threshold -- and for Chinese, where a whole sentence
    is one token and only the characters separate it from another, pick
    examples whose character overlap is obvious (four or five shared
    characters out of seven) rather than one or two.
    """
    vector = np.zeros(dim, dtype=np.float64)
    # ``sorted``, not set order: float addition is not associative, and set
    # iteration order for strings moves with the per-process hash seed. Two
    # processes would otherwise embed the same text into rows that differ in
    # the last bits -- enough to break an exact-equality assertion about a
    # cached embedding.
    for word in sorted(set(text.split())):
        vector += _hashed_direction(word, dim)
    for char in sorted({char for char in text if not char.isspace()}):
        vector += _FAKE_CHAR_WEIGHT * _hashed_direction(char, dim)
    return vector


class _FakeSentenceTransformer:
    """Stands in for ``sentence_transformers.SentenceTransformer``.

    Records what it was asked for -- ``init_kwargs`` (was the model loaded
    from the local snapshot, on the device the node chose?) and ``calls``
    (which texts, how many times, in which order) -- because that is what a
    node test can assert about a model it never trained.

    Strict on purpose. Both signatures take keywords only after the first
    argument, and name the exact keywords a node may pass, so a call that
    would MISBEHAVE against the real library fails here instead of passing:
    ``SentenceTransformer(path, "cpu")`` binds ``modules`` upstream, not
    ``device``, and an unknown keyword is a typo either way.
    """

    def __init__(self, path, *, device=None, cache_folder=None,
                 local_files_only=False, trust_remote_code=False, token=None):
        self.path = path
        # Every keyword, resolved -- not only the ones passed. A node test
        # asserting ``trust_remote_code is False`` should not have to know
        # whether the node spelled the default out.
        self.init_kwargs = {
            "device": device,
            "cache_folder": cache_folder,
            "local_files_only": local_files_only,
            "trust_remote_code": trust_remote_code,
            "token": token,
        }
        # Mutable, like the real one: nodes clamp it to the token budget the
        # user picked, and a test should see the value they set.
        self.max_seq_length = 128
        self.calls: list[list[str]] = []

    def encode(self, texts, *, batch_size=32, convert_to_numpy=True,
               normalize_embeddings=False, show_progress_bar=False):
        """Deterministic embeddings, shaped like the real return value.

        ``batch_size`` and ``show_progress_bar`` are accepted and ignored --
        there is nothing to batch. ``convert_to_numpy`` is accepted and the
        result is always a numpy array: the torch return path is not
        modelled, because none of these nodes asks for it.
        """
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        self.calls.append(list(items))

        if items:
            matrix = np.stack(
                [_hashed_bag(text, _FAKE_EMBED_DIM) for text in items])
        else:
            matrix = np.zeros((0, _FAKE_EMBED_DIM), dtype=np.float64)

        if normalize_embeddings:
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            # An empty text hashes to the zero vector; dividing it by its own
            # norm would hand the node a row of NaN instead of a row of zeros.
            matrix = matrix / np.where(norms == 0.0, 1.0, norms)

        matrix = matrix.astype(np.float32)
        return matrix[0] if single else matrix


def _invalidate_pack_probe_cache() -> None:
    """Drop the process-wide "which packs are installed" cache.

    ``packs.state`` memoises one answer for the whole process, and the answer
    is partly ``find_spec("sentence_transformers")`` -- which reads
    ``sys.modules`` first, and therefore says YES while the fake is installed.
    Cached on the way in, that answer would outlive the test and tell the next
    one the pack is there; cached on the way out of an EARLIER test, it would
    tell this one the pack is missing however loudly the fake is installed.
    So: dropped at both ends, the same convention ``test_api_packs`` and
    ``test_packs_download`` follow.
    """
    try:
        from app.core.packs import state
    except ImportError:  # an install with no packs package -- nothing to drop
        return
    state.invalidate()


@pytest.fixture
def fake_sentence_transformers(monkeypatch, tmp_path):
    """Install the fake library AND pretend its pack is downloaded.

    Both halves are needed for a node test to reach its own logic: without
    the pack patch the node stops at ``require_pack`` before it ever imports
    anything. Everything is undone by ``monkeypatch`` when the test ends, so
    the rest of the suite still sees whether sentence-transformers is really
    installed on this machine.

    Yields the fake MODULE. ``module.SentenceTransformer`` is the class;
    ``module.required`` is the list of ``(pack_id, item_id)`` pairs the code
    under test asked the gate about, in order.
    """
    _invalidate_pack_probe_cache()

    module = types.ModuleType("sentence_transformers")
    # A bare ModuleType has ``__spec__ = None``, which makes ``find_spec``
    # raise ValueError -- harmless in itself, because ``state._module_available``
    # swallows it, but it would swallow it as "not importable". A real spec is
    # what makes a probe taken alongside the fake read "installed", which is
    # the whole point of installing it.
    module.__spec__ = importlib.machinery.ModuleSpec(
        "sentence_transformers", loader=None)
    module.SentenceTransformer = _FakeSentenceTransformer
    # (pack_id, item_id) pairs, in the order the gate was asked about them.
    module.required = []
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)

    model_path = tmp_path / "model"
    model_path.mkdir(parents=True, exist_ok=True)
    # Enough for code that sanity-checks the snapshot before loading it.
    (model_path / "config.json").write_text("{}", encoding="utf-8")

    def fake_pack_available(pack_id, item_id=None):
        module.required.append((pack_id, item_id))
        return True

    def fake_require_pack(pack_id, item_id=None):
        module.required.append((pack_id, item_id))
        return None

    from app.nodes.llm import _packs_bridge

    monkeypatch.setattr(_packs_bridge, "pack_available", fake_pack_available)
    monkeypatch.setattr(_packs_bridge, "require_pack", fake_require_pack)
    monkeypatch.setattr(_packs_bridge, "model_dir", lambda repo_id: model_path)

    try:
        yield module
    finally:
        # NOT redundant with the call on the way in: anything that probed
        # during the test cached an answer taken while the fake was installed.
        # ``monkeypatch`` undoes the ``sys.modules`` entry after this runs, so
        # what matters here is that the cache is left EMPTY -- the next probe
        # then recomputes against the real machine.
        _invalidate_pack_probe_cache()
        # The loaded-model cache is process-wide and keyed by path; a fake
        # model cached under this test's tmp_path would be handed to the
        # next test, after the directory it was loaded from is gone. Guarded
        # because the cache module arrives in a later task than this fixture.
        try:
            from app.nodes.llm import _sentence_models
        except ImportError:
            pass
        else:
            _sentence_models.clear_model_cache()


# ── offline transformers (a local causal LM) ─────────────────────────────
#
# The RAG stack pack is a gigabyte of Qwen2.5 weights on top of a
# transformers install that arrives with the Sentence embeddings pack. No
# test may download either, so ``HFTextGenerate`` and ``_hf_generators`` are
# tested against the fake below: the same call shape, arithmetic instead of
# a transformer.

#: The fake vocabulary. Ids ARE code points, so a generated string can be
#: read straight off them -- and 128 keeps that mapping inside ASCII, where
#: a failed assertion prints something legible.
_FAKE_VOCAB = 128

#: The id the fake calls end-of-turn. 0 because it decodes to NUL, which no
#: test text contains: an eos leaking into the output is visible rather than
#: plausible.
_FAKE_EOS_ID = 0

#: The score the fake puts on the token it wants. Large enough that softmax
#: gives it ~1.0 even at temperature 2, so a SAMPLED run is as predictable
#: as a greedy one -- a test can then assert the exact text either way, and
#: still exercise the multinomial draw.
_FAKE_LOGIT = 30.0


class _FakeKVCache:
    """Opaque stand-in for a transformers ``Cache`` object.

    The node under test is required to treat this as a black box -- take it
    off one forward pass and hand it to the next -- so it carries nothing
    but the bookkeeping that proves this happened.
    """

    def __init__(self, length: int):
        #: How many tokens the model had seen once it produced this.
        self.length = length
        #: Set by the model when this object is handed back to it.
        self.fed_back = False


class _FakeCausalOutput:
    """What the fake's forward returns: ``.logits`` and ``.past_key_values``."""

    __slots__ = ("logits", "past_key_values")

    def __init__(self, logits, past_key_values):
        self.logits = logits
        self.past_key_values = past_key_values


class _FakeGenerationConfig:
    """Only the one field a generator node reads off a real config."""

    def __init__(self, eos_token_id):
        self.eos_token_id = eos_token_id


class _FakeCausalLM(torch.nn.Module):
    """A causal LM whose next token is always ``(last id + 1) % vocab``.

    Deterministic and readable by eye: fed "abc" it continues "defg...".
    That is the whole point -- a randomly initialised transformer would make
    every assertion about the decode loop a coin toss, while this one lets a
    test write the expected string down.

    ``eos_at_step`` is which forward call (0-based) puts its mass on the
    end-of-turn id instead, i.e. how many tokens the model chooses to emit
    before it stops. None means "never stop on your own".
    """

    def __init__(self):
        super().__init__()
        #: A real causal LM builds its logits in the precision its weights
        #: are held in, so the fake needs one weight to be held in: this is
        #: what ``.to(dtype=)`` moves, and it is what makes a test that asks
        #: for float16 really exercise half-precision logits instead of
        #: float32 ones wearing the label.
        self.weight = torch.nn.Parameter(torch.zeros(1))
        #: Which forward call OF ONE GENERATION emits end-of-turn, i.e.
        #: how many tokens the model chooses to write. Counted per
        #: generation and not per instance: the loader caches the model, so
        #: a second run reuses this object and a running counter would make
        #: the second answer stop at a different length than the first.
        self.eos_at_step: int | None = None
        self.step = 0
        #: One dict per forward call: the ids it was fed, the cache it was
        #: handed, whether it was asked to keep one, and how many rows of
        #: logits it was asked for.
        self.calls: list[dict[str, Any]] = []
        #: ``(args, kwargs)`` of every ``.to()`` -- which device and which
        #: precision the loader asked for.
        self.to_calls: list[tuple[tuple, dict]] = []
        self.eval_calls = 0
        self.generation_config = _FakeGenerationConfig([_FAKE_EOS_ID])

    def to(self, *args, **kwargs):
        self.to_calls.append((args, dict(kwargs)))
        return super().to(*args, **kwargs)

    def eval(self):
        self.eval_calls += 1
        return super().eval()

    def forward(self, input_ids=None, *, past_key_values=None,
                use_cache=False, logits_to_keep=0):
        # No ``**_ignored``: every keyword a node may pass is named here, so
        # a call that would MISBEHAVE against the real library (a typo, a
        # keyword transformers dropped) raises TypeError in the test rather
        # than being silently swallowed. Same rule as
        # ``_FakeCausalTokenizer``. ``logits_to_keep=0`` is transformers'
        # own default and means "every position".
        #
        # A generation starts with no cache, and that is what resets the
        # step counter -- see `eos_at_step`.
        self.step = 0 if past_key_values is None else self.step + 1
        step = self.step
        self.calls.append({
            "input_ids": input_ids.tolist(),
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "logits_to_keep": logits_to_keep,
        })
        if past_key_values is not None:
            past_key_values.fed_back = True

        fed = [int(token) for token in input_ids[0].tolist()]
        # In the module's OWN precision, like a real model's LM head. It is
        # what gives the ``.float()`` in a decode loop something to do: a
        # fake that always built float32 logits would let a test say
        # "float16" and prove nothing about it.
        logits = torch.zeros(1, len(fed), _FAKE_VOCAB,
                             dtype=self.weight.dtype)
        for position, token in enumerate(fed):
            logits[0, position, (token + 1) % _FAKE_VOCAB] = _FAKE_LOGIT
        if self.eos_at_step is not None and step == self.eos_at_step:
            # The LAST position only: the earlier rows predict tokens the
            # caller already has, and a decode loop never reads them.
            logits[0, -1, :] = 0.0
            logits[0, -1, _FAKE_EOS_ID] = _FAKE_LOGIT
        if logits_to_keep:
            # What transformers does with the keyword: the LM head runs over
            # the last ``logits_to_keep`` positions only, so the caller gets
            # a shorter tensor back and must still read row -1.
            logits = logits[:, -int(logits_to_keep):, :]

        seen = past_key_values.length if past_key_values is not None else 0
        return _FakeCausalOutput(logits, _FakeKVCache(seen + len(fed)))


class _FakeCausalTokenizer:
    """Ids are code points, so a generated string reads off the ids.

    ``apply_chat_template`` renders the messages as ``<role>content`` runs
    followed by an empty ``<assistant>`` turn: close enough in shape to a
    real chat template that a test can assert the system message is there
    (or absent) without this file pretending to be Qwen's Jinja.

    Strict on purpose, like ``_FakeSentenceTransformer``: everything after
    the first argument is keyword-only and every accepted keyword is one a
    node may really pass, so a call that would MISBEHAVE against the real
    library fails here instead of passing.
    """

    def __init__(self, path, **init_kwargs):
        self.path = path
        self.init_kwargs = init_kwargs
        self.eos_token_id = _FAKE_EOS_ID
        #: One dict per ``apply_chat_template`` call.
        self.chat_calls: list[dict[str, Any]] = []
        #: Every string that was tokenized, in order.
        self.encoded: list[str] = []
        #: One dict per ``__call__``: the text and the keywords it came
        #: with, so a test can see that the chat template's own special
        #: tokens were not doubled by a second pass.
        self.encode_calls: list[dict[str, Any]] = []

    def apply_chat_template(self, messages, *, tokenize=False,
                            add_generation_prompt=False):
        rendered = "".join(f"<{m['role']}>{m['content']}" for m in messages)
        if add_generation_prompt:
            rendered += "<assistant>"
        self.chat_calls.append({
            "messages": [dict(message) for message in messages],
            "tokenize": tokenize,
            "add_generation_prompt": add_generation_prompt,
            "rendered": rendered,
        })
        return rendered

    def __call__(self, text, *, return_tensors=None, add_special_tokens=True):
        self.encoded.append(text)
        self.encode_calls.append({
            "text": text,
            "return_tensors": return_tensors,
            "add_special_tokens": add_special_tokens,
        })
        ids = [ord(char) % _FAKE_VOCAB for char in text]
        return {"input_ids": torch.tensor([ids], dtype=torch.int64)}

    def decode(self, ids, *, skip_special_tokens=False):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return "".join(
            chr(int(token)) for token in ids
            if not (skip_special_tokens and int(token) == _FAKE_EOS_ID))


@pytest.fixture
def fake_transformers(monkeypatch, tmp_path):
    """Install a fake ``transformers`` AND pretend the rag pack is downloaded.

    Both halves are needed for a node test to reach its own logic: without
    the pack patch the node stops at ``require_pack`` before it ever imports
    anything. The ``sys.modules`` entry is written with ``monkeypatch``, so
    it WINS over a real transformers install for the duration of the test
    and is put back (or removed) afterwards -- the rest of the suite still
    sees whether the library is really installed on this machine.

    Yields the fake MODULE, carrying everything a test asserts on:
    ``AutoTokenizer`` / ``AutoModelForCausalLM``, the ``tokenizers`` and
    ``models`` it handed out in creation order, the ``(path, kwargs)`` of
    every ``from_pretrained``, ``model_path`` (the tmp snapshot), and
    ``required`` -- the ``(pack_id, item_id)`` pairs the code under test
    asked the gate about, in order.
    """
    _invalidate_pack_probe_cache()

    module = types.ModuleType("transformers")
    # A bare ModuleType has ``__spec__ = None``, which makes ``find_spec``
    # raise -- swallowed by ``state._module_available``, but swallowed AS
    # "not importable". A real spec is what makes the rag pack's probe read
    # "installed" alongside the fake.
    module.__spec__ = importlib.machinery.ModuleSpec("transformers", loader=None)
    module.tokenizer_loads = []
    # Stamped onto every model the fake hands out, so a test can choose how
    # many tokens the model writes BEFORE the node loads it.
    module.eos_at_step = None
    module.model_loads = []
    module.tokenizers = []
    module.models = []
    # (pack_id, item_id) pairs, in the order the gate was asked about them.
    module.required = []

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(path, *, local_files_only=False):
            module.tokenizer_loads.append(
                (path, {"local_files_only": local_files_only}))
            tokenizer = _FakeCausalTokenizer(
                path, local_files_only=local_files_only)
            module.tokenizers.append(tokenizer)
            return tokenizer

    class AutoModelForCausalLM:
        @staticmethod
        def from_pretrained(path, *, local_files_only=False):
            module.model_loads.append(
                (path, {"local_files_only": local_files_only}))
            model = _FakeCausalLM()
            model.path = path
            model.eos_at_step = module.eos_at_step
            module.models.append(model)
            return model

    module.AutoTokenizer = AutoTokenizer
    module.AutoModelForCausalLM = AutoModelForCausalLM
    monkeypatch.setitem(sys.modules, "transformers", module)

    model_path = tmp_path / "generator"
    model_path.mkdir(parents=True, exist_ok=True)
    # Enough for code that sanity-checks the snapshot before loading it.
    (model_path / "config.json").write_text("{}", encoding="utf-8")
    module.model_path = model_path

    def fake_pack_available(pack_id, item_id=None):
        module.required.append((pack_id, item_id))
        return True

    def fake_require_pack(pack_id, item_id=None):
        module.required.append((pack_id, item_id))
        return None

    from app.nodes.llm import _packs_bridge

    monkeypatch.setattr(_packs_bridge, "pack_available", fake_pack_available)
    monkeypatch.setattr(_packs_bridge, "require_pack", fake_require_pack)
    monkeypatch.setattr(_packs_bridge, "model_dir", lambda repo_id: model_path)

    try:
        yield module
    finally:
        # NOT redundant with the call on the way in: anything that probed
        # during the test cached an answer taken while the fake was
        # installed, and that answer must not outlive it.
        _invalidate_pack_probe_cache()
        # The loaded-generator cache is process-wide; a fake model cached
        # under this test's tmp_path would be handed to the next test after
        # the directory it was loaded from is gone.
        from app.nodes.llm import _hf_generators

        _hf_generators.clear_generator_cache()
