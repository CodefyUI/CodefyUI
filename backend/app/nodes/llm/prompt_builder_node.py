"""PromptBuilder -- the retrieved chunks and the question, in one string.

The last node of the RAG chain::

    ... -> VectorStore -> Retriever -> PromptBuilder -> HFTextGenerate

**This node is the whole trick.** Everything upstream of it exists to
decide WHICH paragraphs to paste here, and everything downstream is an
ordinary generator being shown an ordinary prompt. No weight anywhere in
the graph was changed to make the model know about the corpus; it is told,
at inference time, in the prompt. A learner who has been told "RAG gives an
LLM your documents" and then reads these forty lines of string assembly has
usually learned more than the rest of the chain teaches put together --
which is why the node builds the prompt in the open, on a ``context``
output they can wire to Print, rather than only handing it downstream.

**Why substitution is ``str.replace`` and never ``str.format``.** The text
being pasted in is RETRIEVED, so its content is not ours: a chunk of a
README, a JSON sample, a LaTeX formula, this very docstring. Any of those
can contain a brace, and ``"...{context}...".format(context=chunk)`` does
not care where the braces came from -- an unknown name raises ``KeyError``
and ``{}`` raises ``IndexError``, both from inside a node the learner did
not write, and both triggered by nothing worse than the corpus containing a
code sample.

**Why both placeholders are substituted in one pass.** The obvious two-line
version -- replace ``{context}``, then replace ``{question}`` -- has the
same bug one level up. After the first replace, the retrieved text is IN
the template, so a chunk containing the literal text ``{question}`` becomes
a placeholder the second replace fills, and the learner's question appears
inside a paragraph of the corpus. Swapping the order just moves the bug to
questions containing ``{context}``. So :func:`_fill` splits on one
placeholder and replaces the other inside the pieces, before either value
is joined in: every placeholder it substitutes came from the TEMPLATE, and
nothing that arrived as data is ever scanned.

**Why the template is both a param and an input port.** ``ParamType.STRING``
renders as a single-line text input, and a prompt template is the one string
in this graph a learner genuinely wants to write across several lines. The
param carries a working default so the node runs the moment it lands on the
canvas; the ``template`` input overrides it, so a ``TextInput`` (which has
a real textarea) is how you write your own.

**Why an empty retrieval still builds a prompt.** Raising here would hide
the lesson. A RAG chain that retrieved nothing and answered anyway is the
central failure of the technique, and the way to teach it is to let the run
finish with ``(no context retrieved)`` sitting where the paragraphs should
be -- visible in the prompt, in the ``context`` output, and as a warning in
the log -- so the learner reads the model's confident answer and can see
exactly what it was based on.
"""

from __future__ import annotations

import logging
from typing import Any

from ...core.advisories import emit_advisory, join_notes
from ...core.node_base import (
    BaseNode,
    DataType,
    ParamDefinition,
    ParamType,
    PortDefinition,
)
from ...core.step_trace import StepRecorder

logger = logging.getLogger(__name__)

#: The two spellings the template is required to contain. Named constants
#: because :func:`_fill` and the error message must agree with the shipped
#: default exactly -- a mismatch would be a template that validates and
#: substitutes nothing.
CONTEXT_PLACEHOLDER = "{context}"
QUESTION_PLACEHOLDER = "{question}"

#: The shipped prompt. Three instructions, in this order and for these
#: reasons: answer from the context only (the point of RAG), say you do not
#: know when it is not there (without this a model fills the gap from its
#: own memory and the retrieval was decorative), and answer in the language
#: of the question (this project's corpus and its learners are bilingual,
#: and an English-only instruction pulls the answer into English).
DEFAULT_TEMPLATE = (
    "Answer the question using only the context below. If the context does "
    "not contain the answer, say you do not know. Answer in the language of "
    "the question.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
)

#: What goes BETWEEN chunks. Chunks arrive as unrelated paragraphs from
#: unrelated files, and running them together makes the model read them as
#: one argument; a blank line is the weakest separation that reliably reads
#: as "these are separate". ``rule`` is for models that need it spelled out.
SEPARATORS = {
    "blank_line": "\n\n",
    "newline": "\n",
    "rule": "\n---\n",
}
DEFAULT_SEPARATOR = "blank_line"

#: What the context block says when nothing was retrieved. Phrased as a
#: statement to the model, not an empty string: an empty block leaves
#: "Context:" followed by a blank, which reads as an accident.
NO_CONTEXT = "(no context retrieved)"

#: The source label for a chunk whose citation is unknown -- the same ``?``
#: ``Retriever`` puts on the wire, so the two agree on screen.
UNKNOWN_SOURCE = "?"

#: Appended when ``max_context_chars`` cuts the block, so the model (and
#: the learner reading the prompt) can tell a truncated paragraph from one
#: that simply ended.
TRUNCATION_MARKER = "..."

#: Advisory delivery (``core.advisories``): the Log tab has no severity of
#: its own, so the prefix is what distinguishes this from a Print node's
#: output, and the kind is the token a client may branch on.
PROMPT_NOTE_PREFIX = "[PromptBuilder] "
NO_CONTEXT_WARNING_KIND = "prompt_builder_no_context"


def _text(value: Any) -> str:
    """A port's value as a string; ``None`` (an unwired port) as ``""``."""
    if value is None:
        return ""
    return value if isinstance(value, str) else str(value)


def _string_list(value: Any) -> list[str]:
    """A LIST port's value as strings, in the order it arrived.

    The bare-string guard is not paranoia about the canvas, which type-checks
    LIST against STRING; it is about ``run_graph.py`` and exported scripts,
    where ``contexts="one chunk"`` is an easy thing to write. ``list()`` of a
    string is its CHARACTERS, so without this the prompt would come back
    numbered one letter at a time -- a wrong answer that looks like a
    formatting bug rather than a wiring one.

    ``""`` is an EMPTY list rather than a list holding one blank: a port
    carrying nothing is a port carrying nothing, and a ``contexts`` of ``""``
    has to reach the ``(no context retrieved)`` branch while a ``sources`` of
    ``""`` has to stay un-cited rather than print ``(?)`` on every line.

    Nothing is dropped here, deliberately -- blanks are removed by
    :func:`_drop_blank_contexts`, which removes each one's SOURCE with it.
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    try:
        items = list(value)
    except TypeError:
        return [str(value)]
    return [_text(item) for item in items]


def _fill(template: str, *, context: str, question: str) -> str:
    """Substitute both placeholders in ONE pass over *template*.

    See the module docstring for why the two-replace version is wrong. The
    invariant here is worth stating on its own: the split happens on the
    TEMPLATE, so every piece handed to ``replace`` is template text, and
    *context* is joined in afterwards and never scanned at all. A brace in
    either value is therefore text in both directions -- a chunk containing
    ``{question}`` keeps it, and a question containing ``{context}`` keeps
    it too.
    """
    pieces = [piece.replace(QUESTION_PLACEHOLDER, question)
              for piece in template.split(CONTEXT_PLACEHOLDER)]
    return context.join(pieces)


def _template(inputs: dict[str, Any],
              params: dict[str, Any]) -> tuple[str, bool]:
    """The template in force, and whether a CONNECTED one was passed over.

    Blank counts as absent. An unwired port is ``None`` and a ``TextInput``
    whose box the learner cleared is ``""`` or whitespace -- neither is a
    template, and falling back to the param beats raising about placeholders
    missing from a value that is not visible on the node.

    The second half of the answer is what the log needs. An UNWIRED port is
    the ordinary case and says nothing; a wired one carrying nothing is a
    node the learner built on purpose, silently having no effect, and they
    are entitled to be told which template actually ran.
    """
    supplied = inputs.get("template")
    text = _text(supplied)
    if text.strip():
        return text, False
    return _text(params.get("template")), supplied is not None


def _validate(template: str) -> None:
    """Refuse a template that cannot hold what this node was asked to put in.

    Named as an error rather than repaired, because there is no sensible
    repair: a template without ``{context}`` builds a prompt with no
    retrieval in it, which is a graph that runs, answers from the model's
    memory, and looks like RAG.
    """
    missing = [placeholder
               for placeholder in (CONTEXT_PLACEHOLDER, QUESTION_PLACEHOLDER)
               if placeholder not in template]
    if not missing:
        return
    # Concatenation rather than an f-string: half of this message IS braces,
    # and doubling them to escape the formatter would be one typo away from
    # printing a placeholder that does not exist.
    raise ValueError(
        "PromptBuilder: the template is missing " + " and ".join(missing)
        + ". It must contain both {context} and {question} -- the retrieved "
        "chunks go where the first one is, and the question where the second "
        "one is. Edit the template param, or wire a TextInput into the "
        "template input to write a multi-line one."
    )


def _separator(params: dict[str, Any]) -> str:
    """The string between chunks, defaulting rather than raising.

    ``TextChunker`` refuses an unknown ``strategy`` because the strategy
    decides what the chunks ARE. This one decides what goes between them, so
    a value from a hand-edited graph is worth a default and not a failed run
    that has already paid for the embeddings.
    """
    name = str(params.get("separator") or DEFAULT_SEPARATOR)
    return SEPARATORS.get(name, SEPARATORS[DEFAULT_SEPARATOR])


def _max_context_chars(params: dict[str, Any]) -> int:
    """The cap on the joined block; 0, and anything nonsensical, mean none.

    Clamped rather than refused, like ``Retriever``'s integers: the INT
    widget already bounds this at 0, so an out-of-range value arrived from a
    hand-edited graph or a generated script, and no cap is a better answer
    than an empty context block or a failed run.
    """
    raw = params.get("max_context_chars", 0)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = 0
    return max(0, value)


def _source_at(sources: list[str], position: int) -> str:
    """The citation for chunk *position*, by INDEX and never by shifting.

    ``Retriever`` emits one source per context and already writes ``?``
    where it has no metadata, so a short or ragged list means the wiring is
    unusual. Filling the gap in place keeps every other citation under the
    passage it belongs to; consuming the list in order would slide them all
    up and print a real filename under the wrong text, which is a worse
    lie than admitting one label is unknown.
    """
    source = sources[position] if position < len(sources) else ""
    return source or UNKNOWN_SOURCE


def _drop_blank_contexts(
    contexts: list[str], sources: list[str],
) -> tuple[list[str], list[str]]:
    """Remove blank chunks and the citation belonging to each one.

    A blank entry is not a chunk: it would spend a ``[2]`` and a separator
    saying nothing, and a ``contexts`` port carrying only blanks has to
    reach the ``(no context retrieved)`` branch and its warning rather than
    build a prompt out of punctuation and look like it retrieved something.

    The two lists are filtered AS A PAIR, which is the whole point of doing
    it here rather than inside :func:`_string_list`. They are paired by
    INDEX -- ``sources[i]`` is the citation for ``contexts[i]`` -- so
    dropping a context without dropping its source renumbers one side and
    not the other, and every citation after the gap ends up under the wrong
    passage. That is the failure ``_source_at`` exists to prevent, and it is
    strictly worse than the blank chunk it was trying to remove.

    The surviving sources are resolved through :func:`_source_at`, so a
    short or ragged list still fills its gaps IN PLACE with ``?``. An empty
    ``sources`` stays empty: an unwired port must not start printing
    ``(?)`` on every line.
    """
    kept = [index for index, chunk in enumerate(contexts) if chunk]
    if not sources:
        return [contexts[index] for index in kept], []
    return ([contexts[index] for index in kept],
            [_source_at(sources, index) for index in kept])


def _render(
    contexts: list[str],
    sources: list[str],
    *,
    numbered: bool,
    separator: str,
) -> str:
    """The chunks as one block of text.

    The numbers are what make citation possible at all: a model asked to
    cite ``[2]`` can, and a learner can then count down to the second chunk
    and check. The source in parentheses only appears when the ``sources``
    port is wired -- an unwired one would otherwise print ``(?)`` on every
    line of every prompt, spending tokens to say nothing.
    """
    if not numbered:
        return separator.join(contexts)
    entries: list[str] = []
    for position, chunk in enumerate(contexts):
        label = f"[{position + 1}]"
        if sources:
            label += f" ({_source_at(sources, position)})"
        entries.append(f"{label} {chunk}")
    return separator.join(entries)


class PromptBuilderNode(BaseNode):
    NODE_NAME = "PromptBuilder"
    CATEGORY = "LLM"
    DESCRIPTION = (
        "Assemble the final prompt: the retrieved chunks pasted into a "
        "template together with the question, with an instruction to answer "
        "only from that context. This is the whole trick of RAG -- the model "
        "is not fine-tuned, it is simply shown the right paragraphs. The "
        "template must contain {context} and {question}; wire a TextInput "
        "into the template input to write your own."
    )

    # Nothing downloaded: the text on every port already exists.
    REQUIRES_PACK = None

    # Pure: the same chunks, question and params assemble the same string,
    # and nothing outside the node is touched.
    cacheable = True

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="question",
                data_type=DataType.STRING,
                description=(
                    "the question, usually the same TextInput that was "
                    "embedded for the Retriever"
                ),
            ),
            PortDefinition(
                name="contexts",
                data_type=DataType.LIST,
                description="from Retriever.contexts",
            ),
            PortDefinition(
                name="sources",
                data_type=DataType.LIST,
                optional=True,
                description=(
                    "from Retriever.sources, used for [n] (source) citations"
                ),
            ),
            PortDefinition(
                name="template",
                data_type=DataType.STRING,
                optional=True,
                description=(
                    "overrides the template param; use a TextInput for "
                    "multi-line editing"
                ),
            ),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="prompt",
                data_type=DataType.STRING,
                description=(
                    "the full prompt to send to HFTextGenerate or LLMChat"
                ),
            ),
            PortDefinition(
                name="context",
                data_type=DataType.STRING,
                description="just the joined context block, for Print",
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="template",
                param_type=ParamType.STRING,
                default=DEFAULT_TEMPLATE,
                description=(
                    "The prompt around the chunks; must contain {context} "
                    "and {question}. The template input wins when connected."
                ),
            ),
            ParamDefinition(
                name="separator",
                param_type=ParamType.SELECT,
                default=DEFAULT_SEPARATOR,
                options=list(SEPARATORS),
                description=(
                    "What goes between chunks: blank_line a blank line, "
                    "newline a line break, rule a --- divider."
                ),
            ),
            ParamDefinition(
                name="number_contexts",
                param_type=ParamType.BOOL,
                default=True,
                description=(
                    "Prefix each chunk with [1], [2], ... and its source "
                    "when connected"
                ),
            ),
            ParamDefinition(
                name="max_context_chars",
                param_type=ParamType.INT,
                default=0,
                min_value=0,
                description=(
                    "Truncate the joined context block to this many "
                    "characters (0 = no cap). A small local model gets slow "
                    "past a few thousand."
                ),
                advanced=True,
            ),
        ]

    def execute(
        self,
        inputs: dict[str, Any],
        params: dict[str, Any],
        progress_callback: Any | None = None,
        *,
        context: Any = None,
    ) -> dict[str, Any]:
        # Validated before anything is assembled: a template that cannot
        # hold the retrieval is worth failing on immediately, and the check
        # costs nothing.
        template, blank_template_input = _template(inputs, params)
        _validate(template)

        question = _text(inputs.get("question"))
        # Coerced first, filtered together second. The two lists are paired
        # by index, so a blank chunk has to take its own citation with it --
        # dropping one side alone slides every later citation onto the wrong
        # passage, which is the exact lie ``_source_at`` exists to prevent.
        contexts, sources = _drop_blank_contexts(
            _string_list(inputs.get("contexts")),
            _string_list(inputs.get("sources")),
        )
        limit = _max_context_chars(params)

        warning = None
        truncated = False
        if contexts:
            block = _render(
                contexts, sources,
                numbered=bool(params.get("number_contexts", True)),
                separator=_separator(params),
            )
            # Cut the block, not the placeholder below: a cap of 5 over an
            # empty retrieval would turn "(no context retrieved)" into
            # "(no c...", which says nothing to anybody. The marker takes
            # the total three characters over the cap, which is the cheaper
            # of the two mistakes -- the cap exists to keep a local model
            # responsive, not to hit a byte budget.
            if limit and len(block) > limit:
                block = block[:limit] + TRUNCATION_MARKER
                truncated = True
        else:
            block = NO_CONTEXT
            warning = emit_advisory(
                "no chunks reached this node, so the prompt says "
                f"'{NO_CONTEXT}' -- the model will answer from its own "
                "memory or refuse. Check the Retriever's min_score, and "
                "that the corpus actually loaded",
                kind=NO_CONTEXT_WARNING_KIND,
                prefix=PROMPT_NOTE_PREFIX,
                context=context,
                logger=logger,
            )

        prompt = _fill(template, context=block, question=question)

        note = (f"{len(contexts)} chunk(s) -> {len(block):,}-char context, "
                f"{len(prompt):,}-char prompt")
        if truncated:
            note += f" (context truncated to max_context_chars {limit})"
        # Not an advisory: nothing is wrong, and the node did the sensible
        # thing. But a learner who wired a TextInput and then emptied it is
        # looking at a prompt built from a template they cannot see on the
        # node, and that is worth one line.
        fallback_note = (
            "the connected `template` input is blank, so the template param "
            "was used" if blank_template_input else None
        )

        result: dict[str, Any] = {
            # A plain STRING output: ``core.output_entries`` summarises it
            # to its first 200 characters, which is the beginning of the
            # instructions and the top of the context -- enough to see on
            # the canvas that the prompt is the shape it should be, with
            # the whole thing one click away in the Inspector.
            "prompt": prompt,
            "context": block,
            # The one result key the canvas Log tab renders; dunder keys are
            # filtered out of recorded outputs and port summaries.
            "__log__": join_notes(note, fallback_note, warning),
        }

        if context is not None and getattr(context, "verbose", False):
            result["__steps__"] = self._trace(
                chunks=len(contexts), block=block, prompt=prompt)

        logger.info("PromptBuilder: %d chunk(s) into a %d-char prompt",
                    len(contexts), len(prompt))
        return result

    @staticmethod
    def _trace(*, chunks: int, block: str, prompt: str) -> list[Any]:
        """The two steps the Teaching Inspector shows for one prompt.

        Character counts rather than the strings themselves: a step carries
        tensors and scalars, and both texts are already on ports the
        Inspector renders in full. The counts are the part that is hard to
        see by eye and easy to be wrong about -- a context block that is a
        tenth the size expected is a retrieval problem, not a prompt one.
        """
        recorder = StepRecorder()
        recorder.record(
            "contexts",
            f"{chunks} retrieved chunk(s) joined into one {len(block)}-"
            "character context block. This is the entire world the model "
            "is allowed to answer from.",
            scalars={"chunks": float(chunks), "chars": float(len(block))},
        )
        recorder.record(
            "prompt",
            f"Block and question substituted into the template: "
            f"{len(prompt)} characters go to the generator. None of this "
            "was learned -- it is read at inference time, which is what "
            "makes RAG cheap.",
            scalars={"chars": float(len(prompt))},
        )
        return recorder.steps
