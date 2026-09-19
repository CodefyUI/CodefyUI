"""What the pack-backed example graphs promise, checked without a pack.

Four shipped examples cannot run on a fresh install: Word-Embedding-Analogy
switches to a 69 MB GloVe table for its real vectors,
Sentence-Similarity-zhTW needs a 470 MB multilingual encoder before it does
anything at all, and the two RAG examples need that encoder plus a
generator -- a gigabyte of Qwen2.5 for the local one, a running Ollama or a
hosted API for the other. Three of them are never executed here -- CI has no
pack cache, and a box that HAS one should not load half a gigabyte of
weights inside the fast suite (which is why ``TextEmbedding``,
``HFTextGenerate`` and ``LLMChat`` all sit in
``test_builtin_examples._SLOW_NODE_TYPES``). So this file asserts everything
about them that does not need the model: what the gallery card says, how
many sentences the example carries, where the labels are wired, and -- for
the RAG pair -- that the two graphs really are the same retrieval chain
under two different generators, annotated with the same notes.

Word-Embedding-Analogy is the exception, and only as it SHIPS: its default
``demo-16d`` backend is a table written inline in the source, so running the
graph costs a second and downloads nothing. Its overview note tells the
reader which five words come back, and
``test_analogy_prints_the_five_words_its_note_names`` is where that promise
is checked against a real run. Switching that graph to ``glove-50d`` is the
download, and that test lives next door with the rest of them.

Why the card is worth a test. ``EmptyCanvasOverlay.tsx`` renders an
example's description as ``description.slice(0, 80) + '...'`` with no
``title`` attribute, so those 80 characters are the whole of the warning a
learner reads before pressing Run on a graph that will otherwise stop at a
missing pack. "The requirement is somewhere in the description" is
therefore not the property worth holding; "the requirement is inside the
first 80 characters, and the cut does not land mid-word" is.

The real run lives next door in ``test_pack_examples_real.py``, opt-in
behind ``CODEFYUI_PACK_NETWORK_TESTS=1`` because it needs the download this
file is written to avoid.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from app.config import settings
from app.core.graph_engine import execute_graph, is_note_node
from app.core.node_base import ParamType
from app.core.node_registry import NodeRegistry
from app.nodes.llm.llm_chat_node import LLMChatNode

from tests._example_graphs import graph_nodes

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES_ROOT = _REPO_ROOT / "examples"

_SENTENCE_EXAMPLE = (
    _EXAMPLES_ROOT / "LLM" / "Sentence-Similarity-zhTW" / "graph.json")
_ANALOGY_EXAMPLE = (
    _EXAMPLES_ROOT / "LLM" / "Word-Embedding-Analogy" / "graph.json")
_RAG_LOCAL_EXAMPLE = (
    _EXAMPLES_ROOT / "LLM" / "RAG-Local-Offline" / "graph.json")
_RAG_LLMCHAT_EXAMPLE = (
    _EXAMPLES_ROOT / "LLM" / "RAG-LLMChat-API" / "graph.json")

#: The two RAG graphs, which are the same retrieval chain under two
#: different generators. Several tests below hold both to the same rule, so
#: the pair is named once.
_RAG_EXAMPLES = [
    pytest.param(_RAG_LOCAL_EXAMPLE, id="RAG-Local-Offline"),
    pytest.param(_RAG_LLMCHAT_EXAMPLE, id="RAG-LLMChat-API"),
]

#: The bundled corpus both RAG examples read. ``DocumentLoader``'s
#: ``directory`` param names it relative to ``backend/``.
_RAG_CORPUS = _REPO_ROOT / "backend" / "data" / "samples" / "rag"

#: ``EmptyCanvasOverlay.tsx`` renders a gallery card's description as
#: ``description.slice(0, 80) + '...'``. Mirrors
#: ``test_builtin_examples._CARD_VISIBLE_CHARS``.
_CARD_VISIBLE_CHARS = 80

#: The sidebar ``TemplatesTab`` shows the full description in a native
#: ``title`` tooltip, and a tooltip of a thousand characters is its own kind
#: of unreadable. A generous ceiling rather than a style rule: the detail
#: belongs beside the graph, not on the card.
_DESCRIPTION_LIMIT = 600

#: Each pack-backed example and the phrases its card must show, because
#: without them the learner presses Run and gets an error instead of a
#: download. A TUPLE per example rather than one string: RAG-LLMChat-API
#: needs two things before it answers -- the encoder pack for the retrieval
#: half and a chat backend for the generation half -- and a card naming only
#: one of them still sends somebody to a failed run.
_PACK_BACKED_CARDS = [
    pytest.param(
        _SENTENCE_EXAMPLE,
        ("sentence-embeddings pack",),
        id="Sentence-Similarity-zhTW",
    ),
    pytest.param(
        # BOTH pack ids: installing ``rag`` fetches Qwen but not e5, which
        # comes from ``sentence-embeddings``, and a card naming one of them
        # sends the learner back to Package Center a second time -- after
        # the download they thought was the last one.
        _RAG_LOCAL_EXAMPLE,
        ("rag", "sentence-embeddings"),
        id="RAG-Local-Offline",
    ),
    pytest.param(
        _RAG_LLMCHAT_EXAMPLE,
        ("sentence-embeddings pack", "Ollama"),
        id="RAG-LLMChat-API",
    ),
]


def _payload(graph_path: Path) -> dict:
    return json.loads(graph_path.read_text(encoding="utf-8"))


def _notes(payload: dict) -> dict[str, dict]:
    """The graph's note nodes, keyed by id.

    The complement of ``graph_nodes``: that helper exists so a test about
    the pipeline reads past the prose, and this one so a test about the
    prose can find it. Keyed rather than listed because every assertion
    below is about a NAMED note -- the overview, the one on ``gen`` -- and
    an id is how the two RAG graphs line their notes up with each other.
    """
    return {node["id"]: node for node in payload.get("nodes", [])
            if is_note_node(node)}


#: What ``print_top`` shows on the shipped analogy graph, measured by
#: running it (``demo-16d``, 2026-09-19). The overview note lists these five
#: words to a reader who has not run anything yet, so the list is a fact
#: about the graph rather than a nicety, and
#: ``test_analogy_prints_the_five_words_its_note_names`` holds both to it.
_ANALOGY_TOP_K = ["queen", "duchess", "princess", "mother", "goddess"]


def _cuts_cleanly(description: str) -> bool:
    """Does the card's 80-character slice end between words?

    A boundary, not a length: the cut is clean when the text ends before it,
    or when whitespace sits on either side of it. Anything else splits a
    word, and the card appends its own ellipsis -- so "for real Glo..."
    reads as a bug rather than as a summary.
    """
    return (len(description) <= _CARD_VISIBLE_CHARS
            or description[_CARD_VISIBLE_CHARS].isspace()
            or description[_CARD_VISIBLE_CHARS - 1].isspace())


#: The CJK Unified Ideographs block, as codepoints rather than as the two
#: boundary characters: U+9FFF is a rare ideograph that a cp950 console
#: cannot print, so a literal would break the traceback on the Windows box
#: this suite is developed on -- and it reads as noise in source either way.
_CJK_FIRST, _CJK_LAST = 0x4E00, 0x9FFF


def _has_cjk(text: str) -> bool:
    """Any CJK ideograph, which is all "this line is Chinese" needs to mean."""
    return any(_CJK_FIRST <= ord(char) <= _CJK_LAST for char in text)


@pytest.fixture(scope="module")
def palette() -> set[str]:
    """The registry keys an install offers, built here rather than read.

    The session-wide singleton's contents depend on what earlier tests did
    to it (``test_plugin_api`` runs the app lifespan against a lockfile
    naming three packs, which drops the others), and a check about whether
    an example matches the palette must not pass or fail on collection
    order. Built-ins only, deliberately: a graph under ``examples/`` ships
    with the app and must never need a plugin node.
    """
    reg = NodeRegistry()
    reg.discover(settings.NODES_DIR, "app.nodes")
    keys = set(reg.nodes)
    assert "Start" in keys, (
        "the builtin scan found nothing -- the check would pass vacuously")
    return keys


# -- the gallery cards -----------------------------------------------------

@pytest.mark.parametrize("graph_path, requirements", _PACK_BACKED_CARDS)
def test_pack_example_cards_warn_inside_the_80_char_truncation(
    graph_path: Path, requirements: tuple[str, ...]
):
    """The pack requirement survives the card's cut, and reads as a sentence.

    An example that needs a download has exactly one chance to say so before
    the Run button is pressed, and it is these 80 characters. Asserted
    against the truncated string rather than the whole description, because
    the whole one is not what anybody reads.
    """
    description = _payload(graph_path)["description"]
    visible = description[:_CARD_VISIBLE_CHARS]

    for requirement in requirements:
        assert requirement in visible, (
            f"{requirement!r} is not inside the {_CARD_VISIBLE_CHARS} "
            f"characters the canvas gallery card shows. The card renders "
            f"{visible!r} and nothing more, so a learner without it presses "
            f"Run and gets an error instead of a download -- move the "
            f"requirement to the front of the description.")

    assert _cuts_cleanly(description), (
        f"the card renders {visible!r} and nothing more, cutting a word in "
        f"half; end the opening sentence inside {_CARD_VISIBLE_CHARS} "
        f"characters")

    assert len(description) <= _DESCRIPTION_LIMIT, (
        f"the description is {len(description)} characters; the sidebar shows "
        f"it in full as a tooltip, so keep the detail beside the graph")


def test_analogy_card_says_it_runs_offline():
    """The one fact that decides whether the analogy runs survives the cut.

    That it works offline on ``demo-16d``: a learner without the
    word-vectors pack needs to know the default backend ships with the app.
    It is the mirror image of the test above -- this example runs WITHOUT a
    pack, and the card has to say so.

    ``glove-50d`` used to be asserted here too. It is not a requirement but
    advice -- what to change to see the analogy on real vectors -- and
    advice on a card is what pushed these descriptions past the cut in the
    first place. It now has to be in a note instead, which the test below
    holds it to.
    """
    description = _payload(_ANALOGY_EXAMPLE)["description"]
    visible = description[:_CARD_VISIBLE_CHARS]

    assert "offline" in visible

    # The whole description, not its opening sentence: a one-line card is
    # self-contained because nothing of it is cut, which is the property the
    # old "the first sentence fits" assertion was reaching for by proxy.
    # Stronger than the old check and it replaces it rather than joining it:
    # a description the card shows WHOLE cannot be cut mid-word, so
    # ``_cuts_cleanly`` beside this line would be an assertion that cannot
    # fail. The pack-backed cards above still need it -- theirs are allowed
    # to run past the cut.
    assert len(description) == len(visible), (
        f"the card renders {visible!r} and cuts the rest; an example "
        f"description is one line the card can show whole")


def test_analogy_note_sends_the_reader_to_the_real_vectors():
    """``glove-50d`` is named on the canvas, where there is room for it.

    The toy table is what makes the example runnable and what makes it a
    toy: the analogy is exact on ``demo-16d`` because the 59 vectors were
    written so that it would be, and a reader who never hears about the real
    400k-word table leaves believing word vectors are cleaner than they are.
    That is a paragraph, not a card -- so it lives in a note, and the pack it
    comes from is named beside it or the advice ends at a greyed-out
    dropdown.
    """
    notes = _notes(_payload(_ANALOGY_EXAMPLE))
    assert notes, "the analogy example carries no notes at all"

    text = "\n".join(note["data"]["noteContent"] for note in notes.values())
    for phrase in ("glove-50d", "word-vectors"):
        assert phrase in text, (
            f"{phrase!r} is in none of the analogy's notes "
            f"({sorted(notes)}); the card no longer has room for it, so the "
            f"canvas is where it has to be said")


def test_analogy_prints_the_five_words_its_note_names():
    """The shipped graph is run, and the overview note is read against it.

    The one example in this file that executes here, and the exception is
    the point: its default ``demo-16d`` backend is inline, so this costs a
    second and no download, while the other three would each pull half a
    gigabyte. The note tells the reader which five words ``print_top``
    comes back with -- a promise about a run, made in a file nothing
    executes -- so the run is made and the note is held to it. A note that
    contradicts its graph is worse than no note.
    """
    payload = _payload(_ANALOGY_EXAMPLE)
    results = asyncio.run(execute_graph(
        payload["nodes"], payload["edges"], error_mode="fail_fast"))

    assert results["similarity"]["top_k_labels"] == [_ANALOGY_TOP_K], (
        f"the analogy now ranks "
        f"{results['similarity']['top_k_labels']} on demo-16d; the overview "
        f"note names {_ANALOGY_TOP_K}")

    note = _notes(payload)["note-overview"]["data"]["noteContent"]
    at = -1
    for word in _ANALOGY_TOP_K:
        found = note.find(word, at + 1)
        assert found > at, (
            f"the overview note does not name {word!r} after "
            f"{_ANALOGY_TOP_K[:_ANALOGY_TOP_K.index(word)]}, so it no longer "
            f"lists what print_top shows")
        at = found


# -- Sentence-Similarity-zhTW ---------------------------------------------

def test_sentence_similarity_has_eight_sentences_in_four_pairs():
    """Eight lines, seven of them Chinese and the last one English.

    The example's whole claim is that a pair written with almost no shared
    characters still comes back as each other's nearest neighbour, and the
    last pair makes that claim across two languages. Both are facts about
    the TEXT, invisible to ``validate_graph`` -- an edit that drops a line,
    or that translates the English one into Chinese, leaves a graph that
    validates and runs and no longer demonstrates anything.

    ``split_lines`` is checked in the same breath because it is what turns
    eight lines into eight vectors: with it off the whole block is embedded
    as one text, and every downstream assertion about pairs becomes
    meaningless rather than false.
    """
    payload = _payload(_SENTENCE_EXAMPLE)
    by_id = {node["id"]: node for node in payload["nodes"]}

    lines = [line for line in by_id["sentences"]["data"]["params"]["value"]
             .splitlines() if line.strip()]
    assert len(lines) == 8, (
        f"the example carries {len(lines)} sentences, not the four pairs its "
        f"description promises: {lines}")

    assert lines[-1].isascii(), (
        f"the last sentence is the English half of the cross-language pair, "
        f"and it is not ASCII: {lines[-1]!r}")
    for line in lines[:-1]:
        assert _has_cjk(line), (
            f"{line!r} carries no Chinese characters; seven of the eight "
            f"sentences are the zh-TW half of this example")

    assert by_id["embed"]["data"]["params"]["split_lines"] is True, (
        "split_lines is off, so the eight sentences are embedded as ONE "
        "vector and the pairs cannot be found at all")


def test_sentence_similarity_wires_labels_to_both_consumers(palette: set[str]):
    """The labels reach the top-k list AND the scatter, from the same node.

    Both consumers read ``labels`` positionally against a tensor they get
    from somewhere else (``CosineSimilarity`` pairs ``key_labels`` with
    ``keys``; ``EmbeddingScatter`` pairs ``labels`` with ``embeddings``), and
    both treat them as OPTIONAL -- so an unlabelled run does not fail. It
    prints a list of empty lists where the neighbouring sentences should be,
    and plots eight anonymous dots: the example minus the point of it.

    The node types are checked here too, and against a freshly built
    registry rather than the session-wide singleton: ``registry.get`` falls
    back to a suffix scan server-side while the canvas does an exact
    ``Map.get``, so a type this file did not catch would validate, execute,
    and still render as an empty box with no ports on the learner's screen.
    """
    payload = _payload(_SENTENCE_EXAMPLE)

    wiring = {
        (edge["source"], edge.get("sourceHandle"),
         edge["target"], edge.get("targetHandle"))
        for edge in payload["edges"]
        if edge.get("type", "data") != "trigger"
    }
    for consumer, port in (("similarity", "key_labels"), ("scatter", "labels")):
        assert ("embed", "labels", consumer, port) in wiring, (
            f"TextEmbedding.labels is not wired to {consumer}.{port}; the "
            f"port is optional, so the run still succeeds and shows blanks "
            f"where the sentences should be")

    unresolved = sorted({node["type"] for node in graph_nodes(payload)}
                        - palette)
    assert not unresolved, (
        f"{_SENTENCE_EXAMPLE.name} uses node types that are not registry "
        f"keys: {unresolved}. The canvas resolves a node type by exact "
        f"match, so each of these renders as an empty box with no ports.")


def test_sentence_similarity_runs_from_a_single_trigger():
    """One Start edge, into the one node that has no data input.

    ``TextInput`` is the only source in this graph; everything else is
    downstream of it through data edges, which is what the engine schedules
    on. A second trigger edge would therefore mark a node that already runs,
    and none at all would leave the whole chain unreachable from Start.
    """
    payload = _payload(_SENTENCE_EXAMPLE)
    triggers = [edge for edge in payload["edges"]
                if edge.get("type") == "trigger"]

    assert len(triggers) == 1, (
        f"expected exactly one trigger edge, got "
        f"{[edge.get('id') for edge in triggers]}")
    assert triggers[0]["sourceHandle"] == "trigger"
    assert (triggers[0]["source"], triggers[0]["target"]) == (
        "start", "sentences")


# -- the two RAG examples --------------------------------------------------

def _data_wiring(payload: dict) -> set[tuple[str, str, str, str]]:
    """``(source, sourceHandle, target, targetHandle)`` for the data edges.

    Trigger edges are excluded because they carry no value and are the one
    kind of edge that legitimately fans out; every assertion below is about
    where a VALUE came from.
    """
    return {
        (edge["source"], edge.get("sourceHandle"),
         edge["target"], edge.get("targetHandle"))
        for edge in payload["edges"]
        if edge.get("type", "data") != "trigger"
    }


@pytest.mark.parametrize("graph_path", _RAG_EXAMPLES)
def test_rag_examples_embed_question_and_documents_with_one_model(
    graph_path: Path,
):
    """Two encoders, one model, and the e5 prefixes the right way round.

    Retrieval compares the question's vector against the chunks' vectors, so
    the two ``TextEmbedding`` nodes must be the same model: different models
    put their vectors in different spaces, and a cosine between two spaces is
    a number with no meaning that nothing in the graph can reject. The index
    would still build, the search would still return three chunks, and they
    would be three arbitrary chunks.

    The prefixes are the other half of the same claim. ``multilingual-e5``
    was trained asymmetrically -- ``query: `` for questions and ``passage: ``
    for documents -- and swapping them, or dropping them, quietly costs
    retrieval quality without failing anything.
    """
    payload = _payload(graph_path)
    by_id = {node["id"]: node for node in payload["nodes"]}

    embedders = sorted(node["id"] for node in payload["nodes"]
                       if node["type"] == "TextEmbedding")
    assert embedders == ["embed_docs", "embed_q"], (
        f"expected the document encoder and the question encoder, got "
        f"{embedders}")

    docs_params = by_id["embed_docs"]["data"]["params"]
    question_params = by_id["embed_q"]["data"]["params"]

    assert docs_params["model"] == question_params["model"], (
        f"the chunks are embedded with {docs_params['model']!r} and the "
        f"question with {question_params['model']!r}. Two models are two "
        f"vector spaces, so every similarity the Retriever computes is "
        f"meaningless -- and nothing in the graph can tell.")

    assert docs_params["prefix"] == "passage: ", (
        f"the document encoder's prefix is {docs_params['prefix']!r}; "
        f"multilingual-e5 expects 'passage: ' on the indexed side")
    assert question_params["prefix"] == "query: ", (
        f"the question encoder's prefix is {question_params['prefix']!r}; "
        f"multilingual-e5 expects 'query: ' on the asking side")


@pytest.mark.parametrize("graph_path", _RAG_EXAMPLES)
def test_rag_examples_feed_the_question_to_both_embedder_and_prompt(
    graph_path: Path,
):
    """One TextInput reaches the encoder AND the prompt, from one port.

    These are the two places the question is used, and they are easy to get
    half right. Wire it only to the encoder and ``PromptBuilder.question``
    falls back to nothing, so the model is handed context and no question.
    Wire it only to the prompt and the Retriever searches with whatever else
    is connected. Both halves validate; only the pair actually asks the
    corpus the question the learner typed.
    """
    payload = _payload(graph_path)
    wiring = _data_wiring(payload)

    for target, port in (("embed_q", "text"), ("prompt", "question")):
        assert ("question", "text", target, port) in wiring, (
            f"TextInput.text does not reach {target}.{port}. The question "
            f"has to be both embedded and written into the prompt; wiring "
            f"only one of them still validates and still runs.")


@pytest.mark.parametrize("graph_path", _RAG_EXAMPLES)
def test_rag_examples_trigger_both_root_nodes(graph_path: Path):
    """Start fires ``loader`` AND ``question`` -- the two nodes with no input.

    Execution walks forward from the entry points along DATA edges, so a root
    with no trigger is simply pruned -- and ``validate_graph`` says nothing
    about it, because the OTHER trigger keeps the graph's entry-point check
    satisfied. Dropping ``start -> question`` leaves a file that validates
    clean while ``question`` and ``embed_q`` are no longer scheduled at all,
    so the failure surfaces at run time on ``Retriever``, two nodes away
    from the edge that is missing.

    The sibling check for Sentence-Similarity-zhTW asserts exactly ONE
    trigger, because that graph has one root. These two have two, and both
    of them have to be wired -- and, like that sibling, each edge has to
    leave Start's ``trigger`` handle and be typed as one. An edge that says
    ``"type": "trigger"`` while hanging off a data handle (or the reverse)
    loads as a graph the canvas draws differently from the one the runner
    walks.
    """
    payload = _payload(graph_path)
    # Selected by EITHER mark, so the loop below checks both directions: an
    # edge typed trigger that hangs off a data handle, and an edge off the
    # trigger handle that forgot the type, are both graphs the canvas draws
    # differently from the one the runner walks.
    trigger_edges = [edge for edge in payload["edges"]
                     if edge.get("type") == "trigger"
                     or edge.get("sourceHandle") == "trigger"]
    for edge in trigger_edges:
        assert edge.get("sourceHandle") == "trigger", (
            f"edge {edge.get('id')!r} is typed trigger but leaves the "
            f"{edge.get('sourceHandle')!r} handle")
        assert edge.get("type") == "trigger", (
            f"edge {edge.get('id')!r} leaves Start's trigger handle but is "
            f"typed {edge.get('type')!r}")
    triggers = {(edge["source"], edge["target"]) for edge in trigger_edges}

    assert triggers == {("start", "loader"), ("start", "question")}, (
        f"the trigger wiring is {sorted(triggers)}; both roots -- the "
        f"DocumentLoader and the TextInput -- need their own edge from "
        f"Start, and nothing else does. A root without one is pruned, and "
        f"validate_graph does not notice.")

    fed = {edge["target"] for edge in payload["edges"]
           if edge.get("type", "data") != "trigger"}
    assert not ({"loader", "question"} & fed), (
        "loader or question has an incoming data edge, so it is no longer a "
        "root and the trigger rule above is the wrong rule for this graph")


def test_rag_examples_share_the_retrieval_chain():
    """The two graphs differ in the generator and in nothing else.

    That is the whole teaching point of shipping both: the same nodes do the
    retrieval, and only the last box changes between a local model and an
    API. If the chains drift -- a different chunk size here, a different
    top_k there -- comparing the two answers stops being a comparison of
    GENERATORS, and the README's "compare the two examples" stops being
    true.
    """
    local = _payload(_RAG_LOCAL_EXAMPLE)
    api = _payload(_RAG_LLMCHAT_EXAMPLE)

    def shape(payload: dict) -> dict[str, tuple[str, dict]]:
        # Notes are out: each graph explains ITSELF, so the local one talks
        # about a model on disk and the API one about a chat endpoint. The
        # pipeline is what has to be identical, not the prose beside it.
        return {node["id"]: (node["type"], node["data"]["params"])
                for node in graph_nodes(payload) if node["id"] != "gen"}

    local_shape, api_shape = shape(local), shape(api)
    assert set(local_shape) == set(api_shape), (
        f"the two RAG graphs no longer hold the same nodes: only in the "
        f"local one {sorted(set(local_shape) - set(api_shape))}, only in the "
        f"API one {sorted(set(api_shape) - set(local_shape))}")
    for node_id in sorted(local_shape):
        assert local_shape[node_id] == api_shape[node_id], (
            f"node {node_id!r} differs between the two RAG examples: "
            f"{local_shape[node_id]} vs {api_shape[node_id]}. Everything "
            f"except `gen` is meant to be identical, so the pair compares "
            f"generators and not pipelines.")

    def retrieval_edges(payload: dict) -> set[tuple[str, str, str, str]]:
        return {edge for edge in _data_wiring(payload)
                if "gen" not in (edge[0], edge[2])}

    assert retrieval_edges(local) == retrieval_edges(api), (
        "the retrieval half is wired differently in the two RAG examples; "
        "only the edges into and out of `gen` may differ")

    generators = [next(node["type"] for node in payload["nodes"]
                       if node["id"] == "gen")
                  for payload in (local, api)]
    assert generators == ["HFTextGenerate", "LLMChat"], (
        f"the pair is supposed to show one local generator and one chat API, "
        f"got {generators}")


#: The two notes the RAG pair is allowed to disagree in: the overview, which
#: says what each example is FOR, and the one on the generator, which is the
#: only node that differs. Everything else on the canvas describes the shared
#: retrieval chain and has to read the same in both.
_RAG_NOTES_THAT_DIFFER = {"note-overview", "note-gen"}


def test_rag_examples_carry_the_same_notes_under_the_same_ids():
    """The annotations are part of "the same chain", and nothing sees it.

    ``test_rag_examples_share_the_retrieval_chain`` above compares the two
    graphs through ``graph_nodes``, which drops notes -- deliberately, so
    the pipeline comparison is about the pipeline. The consequence is that
    the prose beside it is compared by NOTHING: a top_k explained one way
    here and another way there would pass every test in this file while the
    reader is told two different things about one shipped parameter.

    So the same rule, one level up. Same note ids, same positions, same
    text -- except in the overview and on ``gen``, where the two examples
    genuinely differ and are REQUIRED to, or one of them is describing the
    other one's generator.
    """
    local = _notes(_payload(_RAG_LOCAL_EXAMPLE))
    api = _notes(_payload(_RAG_LLMCHAT_EXAMPLE))

    assert local, "RAG-Local-Offline carries no notes, so this compares nothing"
    assert set(local) == set(api), (
        f"the two RAG graphs no longer carry the same notes: only in the "
        f"local one {sorted(set(local) - set(api))}, only in the API one "
        f"{sorted(set(api) - set(local))}")

    for note_id in sorted(local):
        assert local[note_id]["position"] == api[note_id]["position"], (
            f"note {note_id!r} sits at {local[note_id]['position']} in the "
            f"local example and {api[note_id]['position']} in the API one; "
            f"the nodes they annotate are at identical positions, so the "
            f"notes are too")

    differ = {note_id for note_id in local
              if local[note_id]["data"]["noteContent"]
              != api[note_id]["data"]["noteContent"]}
    assert differ == _RAG_NOTES_THAT_DIFFER, (
        f"the notes that differ between the two RAG examples are "
        f"{sorted(differ)}, not {sorted(_RAG_NOTES_THAT_DIFFER)}. A note "
        f"about the shared retrieval chain must read the same in both; the "
        f"overview and the one on `gen` must not, or one example is "
        f"describing the other's generator.")


def test_rag_readmes_exist_and_corpus_has_five_documents():
    """Each RAG graph has its README, and the corpus behind them is intact.

    The chat-API's overview note ends by pointing at "README.md beside this
    graph", and the long form of either example is documented nowhere else,
    so a missing file leaves the reader with a graph, a note and no detail.
    The corpus count is the other half: the local example's overview note
    says five notes, both READMEs list the five by subject, and the opt-in
    real run next door asserts which TWO of the five fill the top three of
    the search -- a sixth file dropped into ``backend/data/samples/rag``
    makes all three wrong and breaks none of them on its own.
    """
    for graph_path in (_RAG_LOCAL_EXAMPLE, _RAG_LLMCHAT_EXAMPLE):
        readme = graph_path.parent / "README.md"
        assert readme.is_file(), (
            f"{readme} is missing, but the gallery card for "
            f"{graph_path.parent.name} tells the reader to go and read it")
        text = readme.read_text(encoding="utf-8")
        assert "## Before you run it" in text, (
            f"{readme} has no 'Before you run it' section; that is where the "
            f"pack, its size and the CPU timing are written down")

    documents = sorted(p.name for p in _RAG_CORPUS.glob("*")
                       if p.suffix.lower() in {".md", ".txt"})
    assert len(documents) == 5, (
        f"the bundled corpus holds {len(documents)} documents, not the five "
        f"both RAG cards promise: {documents}")


def test_rag_llmchat_example_carries_no_secret_params():
    """No key field is written into the shipped graph, empty or otherwise.

    ``LLMChat``'s two API keys are ``ParamType.SECRET``: the canvas strips
    them on save and they are never meant to reach disk. A shipped example
    that carries them -- even as ``""`` -- is a field inviting somebody to
    type a key into a file that gets committed, and it is the kind of habit
    that only shows up in a repository once it is too late.

    The names are read off the node class rather than listed here, so a
    third secret added to ``LLMChat`` tomorrow is covered by this test
    today.
    """
    secrets = {param.name for param in LLMChatNode.define_params()
               if param.param_type is ParamType.SECRET}
    assert secrets, (
        "LLMChat declares no SECRET params, so this test would pass "
        "vacuously -- check what changed in llm_chat_node.py")

    for node in graph_nodes(_payload(_RAG_LLMCHAT_EXAMPLE)):
        present = sorted(secrets & set(node["data"]["params"]))
        assert not present, (
            f"node {node['id']!r} carries the secret param(s) {present}. "
            f"Delete the key(s) from the file entirely: a SECRET field in a "
            f"committed example is somewhere to paste an API key by "
            f"accident.")
