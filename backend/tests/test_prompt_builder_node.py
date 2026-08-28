"""Tests for PromptBuilderNode -- where retrieval turns into a prompt.

There is no arithmetic in this node, and that is precisely why it needs
tests: string assembly is where a RAG chain breaks QUIETLY. A chunk of
documentation full of braces, a question whose own text is substituted a
second time, a citation printed under the wrong passage -- none of those
raise. They come back as an answer that reads perfectly and is wrong, which
is the failure mode a learner has no way to spot from the canvas.

So the assertions here are mostly about exact strings. The three that carry
the most weight are the two brace tests (a placeholder that arrives inside
retrieved TEXT must stay text, in both directions) and the alignment half of
``test_numbering_and_sources_are_rendered`` (a citation under the wrong
chunk is worse than no citation at all).
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from app.core.node_base import DataType, ParamType
from app.nodes.llm.prompt_builder_node import PromptBuilderNode

#: Every param the node ships, read off the node itself so each test names
#: only the ones it actually changes.
_DEFAULTS = {p.name: p.default for p in PromptBuilderNode.define_params()}

#: Two short chunks whose text contains no braces and no digits, so a
#: numbering or substitution bug cannot hide inside the content.
_CONTEXTS = [
    "A node is a box that computes.",
    "An edge carries a tensor between nodes.",
]
#: Deliberately NOT in the same order as the chunks they belong to, so a
#: rendering that sorted or reused sources would show.
_SOURCES = ["02-nodes-and-edges.md", "01-what-is-codefyui.md"]
_QUESTION = "What is a node?"


@dataclass
class FakeContext:
    """The one attribute this node reads off an ExecutionContext."""

    verbose: bool = False


def _run(*, inputs=None, context=None, **params) -> dict:
    p = dict(_DEFAULTS)
    p.update(params)
    wired = {"question": _QUESTION, "contexts": list(_CONTEXTS)}
    wired.update(inputs or {})
    return PromptBuilderNode().execute(wired, p, context=context)


def test_node_metadata():
    assert PromptBuilderNode.NODE_NAME == "PromptBuilder"
    assert PromptBuilderNode.CATEGORY == "LLM"
    # Nothing downloaded: assembling a prompt is string work over text that
    # already exists.
    assert PromptBuilderNode.REQUIRES_PACK is None
    # Pure: the same chunks and the same question build the same prompt.
    assert PromptBuilderNode.cacheable is True

    inputs = {p.name: p for p in PromptBuilderNode.define_inputs()}
    assert list(inputs) == ["question", "contexts", "sources", "template"]
    assert inputs["question"].data_type == DataType.STRING
    assert inputs["contexts"].data_type == DataType.LIST
    assert inputs["sources"].data_type == DataType.LIST
    assert inputs["template"].data_type == DataType.STRING
    assert not inputs["question"].optional
    assert not inputs["contexts"].optional
    # Citations and a hand-written template are both extras: a first RAG
    # graph wires two ports and works.
    assert inputs["sources"].optional
    assert inputs["template"].optional

    outputs = {p.name: p for p in PromptBuilderNode.define_outputs()}
    assert list(outputs) == ["prompt", "context"]
    assert outputs["prompt"].data_type == DataType.STRING
    assert outputs["context"].data_type == DataType.STRING

    params = {p.name: p for p in PromptBuilderNode.define_params()}
    assert list(params) == ["template", "separator", "number_contexts",
                            "max_context_chars"]
    assert params["template"].param_type == ParamType.STRING
    assert params["separator"].param_type == ParamType.SELECT
    assert params["separator"].default == "blank_line"
    assert params["separator"].options == ["blank_line", "newline", "rule"]
    assert params["number_contexts"].param_type == ParamType.BOOL
    assert params["number_contexts"].default is True
    assert params["max_context_chars"].param_type == ParamType.INT
    assert params["max_context_chars"].default == 0
    assert params["max_context_chars"].min_value == 0
    # A cap nobody touches until a local model gets slow -- behind Advanced
    # so the default view of the node stays three knobs.
    assert params["max_context_chars"].advanced is True


def test_default_template_contains_both_placeholders():
    template = _DEFAULTS["template"]
    assert "{context}" in template
    assert "{question}" in template

    # REAL newlines, not the two-character escape. The param renders in a
    # single-line text input, which makes a backslash-n look plausible in
    # source -- and it would reach the model as a backslash and an n,
    # turning the shipped prompt into one long line of instructions.
    assert "\n" in template
    assert "\\n" not in template

    # And the shipped default is a working prompt the moment the node lands
    # on the canvas, with nothing wired but a Retriever.
    result = _run(inputs={"sources": list(_SOURCES)})
    assert result["prompt"].startswith(
        "Answer the question using only the context below.")
    assert ("Context:\n[1] (02-nodes-and-edges.md) A node is a box that "
            "computes.") in result["prompt"]
    assert result["prompt"].endswith(
        "\n\nQuestion: What is a node?\nAnswer:")


def test_numbering_and_sources_are_rendered():
    numbered = _run(inputs={"sources": list(_SOURCES)})
    assert numbered["context"] == (
        "[1] (02-nodes-and-edges.md) A node is a box that computes.\n\n"
        "[2] (01-what-is-codefyui.md) An edge carries a tensor between nodes."
    )

    # Sources are read POSITIONALLY, so a list shorter than the chunks still
    # numbers every chunk and marks the ones it cannot cite. Sliding the
    # remaining citations up would put a real filename under the wrong
    # passage, which is worse than admitting the label is unknown.
    short = _run(inputs={"sources": ["only-one.md"]})
    assert short["context"].startswith("[1] (only-one.md) A node is a box")
    assert "[2] (?) An edge carries a tensor" in short["context"]

    # Nothing wired into sources at all: numbers, but no empty parenthetical
    # on every line of every prompt.
    unwired = _run()
    assert unwired["context"] == (
        "[1] A node is a box that computes.\n\n"
        "[2] An edge carries a tensor between nodes."
    )

    plain = _run(inputs={"sources": list(_SOURCES)}, number_contexts=False)
    assert plain["context"] == (
        "A node is a box that computes.\n\n"
        "An edge carries a tensor between nodes."
    )
    # The context output is the same text the prompt got, so a learner can
    # Print it and read exactly what the model was shown.
    assert plain["context"] in plain["prompt"]

    # A bare string on the contexts port is ONE chunk. ``list()`` of a
    # string is its characters, so the naive coercion would number this
    # prompt one letter at a time -- a failure that reads as a formatting
    # bug when it is really a wiring one. Same for a value that is not
    # iterable at all.
    assert _run(inputs={"contexts": "one whole chunk"})["context"] == (
        "[1] one whole chunk")
    assert _run(inputs={"contexts": 42})["context"] == "[1] 42"


def test_separator_options():
    joined = "A node is a box that computes.{sep}An edge carries a tensor " \
             "between nodes."

    for name, sep in (("blank_line", "\n\n"), ("newline", "\n"),
                      ("rule", "\n---\n")):
        result = _run(separator=name, number_contexts=False)
        assert result["context"] == joined.replace("{sep}", sep), name

    # A separator is cosmetic. An unrecognised one falls back to the default
    # rather than failing a run that has already paid for the embeddings --
    # unlike TextChunker's strategy, which changes what the chunks ARE.
    assert (_run(separator="nonsense", number_contexts=False)["context"]
            == _run(separator="blank_line", number_contexts=False)["context"])


def test_a_rule_separator_still_numbers_and_cites():
    """The two knobs are independent, and ``rule`` is the one that looks
    like markup.

    A divider line between numbered entries is what a small model needs
    when a blank line is not enough separation, and the numbering has to
    survive it intact -- the ``[n]`` is what makes "cite your source"
    answerable at all.
    """
    result = _run(separator="rule", number_contexts=True,
                  inputs={"sources": list(_SOURCES)})

    assert result["context"] == (
        "[1] (02-nodes-and-edges.md) A node is a box that computes."
        "\n---\n"
        "[2] (01-what-is-codefyui.md) An edge carries a tensor between nodes."
    )
    assert result["context"] in result["prompt"]

    # The divider is not mistaken for a placeholder or a chunk: two chunks
    # mean exactly one divider.
    assert result["context"].count("\n---\n") == 1


def test_template_input_overrides_param():
    wired = _run(inputs={"template": "Q: {question}\nC: {context}"},
                 template="the param: {context} {question}")
    assert wired["prompt"] == (
        "Q: What is a node?\n"
        "C: [1] A node is a box that computes.\n\n"
        "[2] An edge carries a tensor between nodes."
    )

    # An unwired port arrives as None and a TextInput the learner emptied
    # arrives blank. Neither is a template, so both fall back to the param
    # instead of raising about placeholders missing from a value the
    # learner cannot see on the node.
    for empty in (None, "", "   "):
        fallback = _run(inputs={"template": empty},
                        template="the param: {context}|{question}")
        assert fallback["prompt"].startswith("the param: [1] "), repr(empty)


def test_a_blank_connected_template_says_so_in_the_log():
    """A wired TextInput with nothing in it is a node with no effect.

    Falling back to the param is right -- raising about placeholders missing
    from a value that is not visible on the node would be worse -- but the
    learner built that TextInput on purpose, and is entitled to know that
    the prompt came from somewhere else.
    """
    for blank in ("", "   "):
        result = _run(inputs={"template": blank},
                      template="the param: {context}|{question}")
        assert ("the connected `template` input is blank, so the template "
                "param was used") in result["__log__"], repr(blank)

    # An UNWIRED port is the ordinary case and says nothing about itself.
    assert "template" not in _run(inputs={"template": None})["__log__"]
    assert "template" not in _run()["__log__"]

    # And a connected template that HAS something in it is simply used.
    used = _run(inputs={"template": "Q {question} C {context}"})
    assert "template" not in used["__log__"]


def test_a_stray_brace_pair_in_the_template_is_just_text():
    """Substitution is ``str.replace``, and this is what that buys.

    A template asking for JSON -- ``reply as {"answer": ...}`` -- is an
    ordinary thing to write, and under ``str.format`` it is a ``KeyError``
    from inside a node the learner did not write, triggered by nothing
    worse than a brace.
    """
    template = (
        'Reply as {"answer": "...", "cited": [1]}.\n'
        "Context:\n{context}\nQuestion: {question}"
    )

    result = _run(template=template, number_contexts=False,
                  inputs={"contexts": ["A node is a box that computes."]})

    assert result["prompt"] == (
        'Reply as {"answer": "...", "cited": [1]}.\n'
        "Context:\nA node is a box that computes.\n"
        "Question: What is a node?"
    )
    # The stray pair is untouched -- not substituted, not escaped, not eaten.
    assert '{"answer": "...", "cited": [1]}' in result["prompt"]


def test_missing_placeholder_is_an_error():
    with pytest.raises(ValueError) as excinfo:
        _run(template="Answer this: {question}")
    message = str(excinfo.value)
    assert "missing {context}" in message
    # BOTH are named even when only one is missing: a learner who wrote
    # their own template has no other way to learn what the other is called.
    assert "{context} and {question}" in message

    with pytest.raises(ValueError) as excinfo:
        _run(template="Context: {context}")
    assert "missing {question}" in str(excinfo.value)

    # An emptied param is not a fallback to the default -- it is a template
    # with no placeholders, and saying so teaches what a template needs.
    with pytest.raises(ValueError) as excinfo:
        _run(template="")
    assert "missing {context} and {question}" in str(excinfo.value)


def test_braces_inside_context_survive():
    # A retrieved chunk of documentation about this very node. str.format
    # would raise KeyError on it, and a second substitution pass would eat
    # the literal text and paste the question inside the chunk.
    chunk = "The template needs {context} and {question} in it."
    result = _run(inputs={"contexts": [chunk]}, number_contexts=False)

    assert result["context"] == chunk
    assert result["prompt"].endswith(
        "Context:\nThe template needs {context} and {question} in it."
        "\n\nQuestion: What is a node?\nAnswer:"
    )
    # Exactly once: the {question} inside the CHUNK was never a placeholder,
    # only text that looks like one.
    assert result["prompt"].count("What is a node?") == 1


def test_braces_inside_the_question_survive():
    # The mirror image, and the one an implementation that substitutes
    # {context} first gets wrong: by the time {question} is replaced the
    # question text is already in the string, so a {context} inside it would
    # be substituted on the way past and the whole context block would land
    # in the middle of the question.
    question = "Why does a template need {context} in it?"
    result = _run(inputs={"question": question, "contexts": ["Because."]},
                  number_contexts=False)

    assert result["prompt"].endswith(
        "Context:\nBecause.\n\nQuestion: Why does a template need "
        "{context} in it?\nAnswer:"
    )
    assert result["prompt"].count("Because.") == 1


def test_max_context_chars_truncates():
    long_chunk = "x" * 500

    capped = _run(inputs={"contexts": [long_chunk]}, number_contexts=False,
                  max_context_chars=100)
    assert capped["context"] == "x" * 100 + "..."
    assert capped["context"] in capped["prompt"]
    assert "truncated" in capped["__log__"]

    # 0 is the shipped default and means no cap at all.
    uncapped = _run(inputs={"contexts": [long_chunk]}, number_contexts=False)
    assert uncapped["context"] == long_chunk
    assert "truncated" not in uncapped["__log__"]

    # A block that already fits is left exactly alone: an ellipsis on a
    # complete context tells the model something was cut when nothing was.
    exact = _run(inputs={"contexts": [long_chunk]}, number_contexts=False,
                 max_context_chars=500)
    assert exact["context"] == long_chunk

    # A hand-edited graph can carry a value the INT widget would never
    # produce; a nonsense cap means no cap rather than an empty prompt.
    for junk in (-10, "lots", None):
        assert _run(inputs={"contexts": [long_chunk]}, number_contexts=False,
                    max_context_chars=junk)["context"] == long_chunk


def test_empty_contexts_produce_placeholder_and_log():
    result = _run(inputs={"contexts": []})

    # The prompt is still built -- the run continues and the learner gets to
    # watch what a model does with nothing -- but the block says so out
    # loud, where both the model and the reader can see it.
    assert result["context"] == "(no context retrieved)"
    assert "Context:\n(no context retrieved)\n\nQuestion:" in result["prompt"]
    assert "[PromptBuilder] " in result["__log__"]
    assert "no chunks reached this node" in result["__log__"]

    # An unwired LIST port arrives as None, not [], and must not crash on
    # the way to the same placeholder.
    assert _run(inputs={"contexts": None})["context"] == (
        "(no context retrieved)")

    # The normal case says nothing alarming.
    assert "[PromptBuilder] " not in _run()["__log__"]


def test_blank_chunks_are_not_chunks():
    """A retrieval that produced only blanks retrieved nothing.

    ``contexts=""`` is what an exported script writes when the upstream node
    produced nothing, and a list holding ``""`` is what a hand-built one
    writes. Either way the prompt must say ``(no context retrieved)`` and
    warn -- a ``[1]`` with nothing after it looks like a retrieval that
    worked.
    """
    for nothing in ("", [""], ["", None], [None]):
        result = _run(inputs={"contexts": nothing})
        assert result["context"] == "(no context retrieved)", repr(nothing)
        assert "no chunks reached this node" in result["__log__"], repr(nothing)

    # A blank among real chunks is dropped, and the numbering closes up
    # rather than printing an empty [2].
    mixed = _run(inputs={"contexts": ["first", "", "second"]})
    assert mixed["context"] == "[1] first\n\n[2] second"

    # ``sources`` is the one list read POSITIONALLY, so a blank entry there
    # stays where it is and prints as ``?`` -- closing the gap would put a
    # real filename under the wrong passage.
    cited = _run(inputs={"contexts": ["first", "second"],
                         "sources": ["", "b.md"]})
    assert cited["context"] == "[1] (?) first\n\n[2] (b.md) second"


def test_a_whitespace_only_chunk_is_blank_too():
    """Three spaces read as empty to everyone looking at the prompt.

    ``_template`` already calls a whitespace-only connected template blank;
    a chunk of spaces is the same thing arriving on a different port, and
    counting it as content spends a ``[2]`` on nothing and pushes the real
    passages down a number.
    """
    result = _run(inputs={"contexts": ["first", "   ", "third"],
                          "sources": ["a.md", "b.md", "c.md"]})

    assert result["context"] == (
        "[1] (a.md) first"
        "\n\n"
        "[2] (c.md) third"
    )
    assert "b.md" not in result["context"]

    # Every chunk whitespace-only is a retrieval that retrieved nothing.
    empty = _run(inputs={"contexts": ["  ", "\n", "\t"]})
    assert empty["context"] == "(no context retrieved)"
    assert "no chunks reached this node" in empty["__log__"]


def test_blank_context_does_not_shift_citations():
    """A dropped chunk takes its own citation with it.

    The two lists are paired by index, so filtering one and not the other
    renumbers one side only: ``['first', '', 'third']`` against
    ``['a.md', 'b.md', 'c.md']`` then prints ``third`` under ``b.md``. That
    is worse than the blank chunk the filter was removing -- a citation
    under the wrong passage is exactly the failure the whole RAG chain
    carries ``source`` around to prevent, and nothing on screen shows it.
    """
    shifted = _run(inputs={"contexts": ["first", "", "third"],
                           "sources": ["a.md", "b.md", "c.md"]})

    assert shifted["context"] == (
        "[1] (a.md) first"
        "\n\n"
        "[2] (c.md) third"
    )
    assert "b.md" not in shifted["context"], (
        "the citation of the dropped chunk survived it")

    # An unwired ``sources`` stays unwired: dropping a chunk must not start
    # printing ``(?)`` on every line.
    assert _run(inputs={"contexts": ["first", "", "third"]})["context"] == (
        "[1] first\n\n[2] third")

    # A SHORT sources list still fills its gaps by position, after the pair
    # has been filtered: 'third' was index 2, which the list does not reach.
    short = _run(inputs={"contexts": ["first", "", "third"],
                         "sources": ["a.md"]})
    assert short["context"] == "[1] (a.md) first\n\n[2] (?) third"

    # And the mirror image, restated here because it is the reason the
    # filter cannot simply drop blanks from both lists: a blank in SOURCES
    # alone keeps its slot and prints ``?`` without moving its neighbours.
    only_source_blank = _run(inputs={"contexts": ["first", "second", "third"],
                                     "sources": ["a.md", "", "c.md"]})
    assert only_source_blank["context"] == (
        "[1] (a.md) first"
        "\n\n"
        "[2] (?) second"
        "\n\n"
        "[3] (c.md) third"
    )


def test_step_trace_when_verbose():
    result = _run(inputs={"sources": list(_SOURCES)},
                  context=FakeContext(verbose=True))

    assert [s.name for s in result["__steps__"]] == ["contexts", "prompt"]
    steps = {s.name: s for s in result["__steps__"]}
    assert steps["contexts"].scalars["chunks"] == 2.0
    assert steps["contexts"].scalars["chars"] == float(len(result["context"]))
    assert steps["prompt"].scalars["chars"] == float(len(result["prompt"]))

    quiet = _run(inputs={"sources": list(_SOURCES)})
    assert "__steps__" not in quiet
