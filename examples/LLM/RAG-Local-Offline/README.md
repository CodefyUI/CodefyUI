# RAG, fully local: retrieve then generate

Retrieval-augmented generation with nothing leaving the machine. The graph reads
a folder of plain-text notes, turns them into vectors, finds the three passages
closest to your question, pastes those into a prompt, and lets a small local
model answer from them.

The point of the example is the *middle*, not the model. A 0.5B model knows
almost nothing about CodefyUI; wired to five notes about it, it answers
correctly anyway. That gap is the whole argument for retrieval.

Two chains meet at the `Retriever`:

```
loader -> chunker -> embed_docs -> store ---+
                     ("passage: ")          |
                                            v
question -> embed_q ---------------------> retriever -> prompt -> gen -> print_ans
            ("query: ")
```

Three things join that from the side:

- `chunker.chunks` and `chunker.metadata` also go into `store`. The vectors
  alone are not an index -- the store keeps the chunk *texts* and their
  sources beside them, which is what lets a hit be cited.
- `question.text` also goes into `prompt.question`, so the model is given the
  question as well as the passages.
- `retriever.contexts` also goes into `print_ctx`, so you can read what was
  retrieved *before* the model writes anything about it.

The documents go down one side, the question goes down the other, and both are
encoded by the **same model** -- that is the one invariant the graph cannot
survive breaking.

## The stages

| Node | What it does | The knob to try |
|---|---|---|
| `loader` `DocumentLoader` | Reads every `.md` and `.txt` in `data/samples/rag` and emits `{text, source}` per file | `directory` -- point it at your own folder |
| `chunker` `TextChunker` | Cuts each note into 400-character windows overlapping by 80, carrying `source` and the character offsets | `chunk_size` -- smaller is more precise, larger keeps more context per hit |
| `embed_docs` `TextEmbedding` | One vector per chunk, prefixed `passage: ` for multilingual-e5 | `model` -- the whole index is rebuilt when you change it |
| `store` `VectorStore` | Stacks the vectors into one `[N, D]` matrix with the chunk texts and metadata beside it | `metric` -- `cosine` ignores vector length, `dot` does not |
| `question` `TextInput` | The question, used twice: embedded, and written into the prompt | `value` -- this is the field to edit |
| `embed_q` `TextEmbedding` | The question's vector, prefixed `query: ` | `prefix` -- must stay `query: ` while `embed_docs` says `passage: ` |
| `retriever` `Retriever` | The 3 nearest chunks and the files they came from | `top_k`, `min_score` |
| `print_ctx` `Print` | Shows what was retrieved *before* the model sees it | `label` |
| `prompt` `PromptBuilder` | Numbers the chunks, cites their sources, and wraps them in an answer-only-from-the-context template | `template`, `number_contexts` |
| `gen` `HFTextGenerate` | Qwen2.5-0.5B-Instruct writes the answer on the CPU | `max_new_tokens`, `temperature` |
| `print_ans` `Print` | The answer | `label` |

`start` fires `loader` and `question` -- the two nodes with no incoming data
edge. Everything else is reached along data edges from those two.

## Before you run it

| | |
|---|---|
| Packs | **Two items, from two packs**, in Package Center (Settings in the toolbar). `qwen2.5-0.5b-instruct` from **RAG stack**, and `multilingual-e5-small` from **Sentence embeddings**. RAG stack *depends* on Sentence embeddings, so Package Center will not let you install it until that pack's Python packages are in place -- but the dependency brings the library, **not** the encoder. Pick the e5 model yourself, or install the whole Sentence embeddings pack and get all four encoders. |
| Disk | About 1.5 GB for those two items: roughly 1.0 GB for Qwen2.5-0.5B-Instruct (Apache-2.0) plus 470 MB for `multilingual-e5-small` (MIT). Installing all of Sentence embeddings instead is about 1.1 GB of models. |
| Network | Needed **once**, for the download. The run itself is offline -- no request leaves the machine, which is the difference between this example and `RAG-LLMChat-API` next door. |
| GPU | Not required. Set `gen.device` and `embed_docs.device` / `embed_q.device` to `cuda` if you have one; `auto` follows the global device selector. |
| Time | The first run also pays a few seconds to read the weights off disk; both models stay cached afterwards. Generating 160 tokens from a 0.5B model on a laptop CPU is roughly 20-40 seconds. **That figure is an estimate from the model size and the token budget, not a recorded measurement** -- see [Provenance](#provenance). |

Nothing is downloaded by pressing Run: a missing pack stops the graph with an
error naming the pack, rather than fetching a gigabyte behind your back.

## Try this

**Ask something else.** Edit `question.value` and run again. Three that the
bundled corpus can answer, and that exercise different parts of it:

- `What is the difference between a data edge and a trigger edge?`
- `Why does a training run need a DataLoader?`
- `什麼是套件中心？` -- every note has a Traditional Chinese half, and a
  multilingual encoder is supposed to put this question next to the Chinese
  half of `05-optional-packs.md` rather than next to the English word "pack".
  The prompt template says "Answer in the language of the question", so the
  answer should come back in Chinese too.

**Ask something the corpus cannot answer** -- `What is the capital of France?`
-- and watch what happens. The retriever still returns its three nearest
chunks, because "nearest" is always defined; the template is what stops the
model inventing an answer from them. Raise `retriever.min_score` until nothing
clears the bar and the retrieval comes back empty instead, at which point
`PromptBuilder` writes `(no context retrieved)` into the context block. The
Retriever's own log line reports the best score it saw, which is how you find
out where to put the bar for this corpus.

**Swap the embedding model.** Change `model` on **both** `embed_docs` and
`embed_q` to `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (or
`BAAI/bge-small-zh-v1.5` for a Chinese-focused one) and compare which chunks
come back. Those models were not trained with prefixes, so clear `prefix` on
both nodes when you switch -- they ignore it, but leaving `passage: ` in place
is a habit that costs quality the day you switch back to e5.

Changing the model on only one of the two is the interesting failure: the graph
runs, the Retriever returns three chunks, and they are three arbitrary chunks.
Two models are two vector spaces, and a cosine between them is a number with no
meaning that nothing in the graph can reject.

**Raise `top_k`.** Three chunks is about 1,200 characters of context. Try 5 or
8 and watch two things move in opposite directions: the answer gets more to
work with, and the 0.5B model gets slower and more easily distracted by the
chunk that turned out to be irrelevant. `max_context_chars` on `PromptBuilder`
is the cap for when that gets out of hand.

## The corpus

`data/samples/rag` ships five short notes -- what CodefyUI is, nodes and edges,
training basics, embeddings and RAG, and optional packs -- each with an English
half and a Traditional Chinese half.

**`DocumentLoader` reads every `.md` and `.txt` in the folder it is pointed at**
(and only those two extensions -- no PDF, no HTML, no DOCX). So the fastest way
to make this example about your own material is to put a folder of notes
somewhere and set `loader.directory` to it; nothing else in the graph changes.
`recursive` adds subfolders, and `max_docs` caps how many files are read, which
is how you try a big folder cheaply. A file that is not valid UTF-8 is refused
by name rather than read with replacement characters, because mojibake embeds
silently and shows up only as a retrieval result nobody can explain.

## Notes for anyone editing this graph

- **Both `TextEmbedding` nodes must name the same model.** See above -- this is
  the failure that produces plausible nonsense rather than an error.
- **The prefixes are not decoration.** `multilingual-e5` was trained
  asymmetrically: `query: ` on the asking side, `passage: ` on the indexed
  side. Swapping them, or dropping them, quietly costs retrieval quality.
- **`question` feeds two nodes.** `embed_q.text` and `prompt.question`. Wire
  only the first and the model gets context with no question; wire only the
  second and the search runs on something else. Both halves validate.
- **`chunker.chunks` also feeds `store.chunks`.** The vectors alone are not an
  index -- the store keeps the chunk *texts* beside them, and `metadata` is what
  lets the Retriever name the file each hit came from.
- **`gen.prompt` is empty on purpose.** The param is only the fallback for when
  nothing is connected; `PromptBuilder.prompt` is what this graph sends.
- **Both root nodes need a trigger edge.** Execution walks forward from the
  entry points along data edges, so `loader` and `question` are each wired
  directly from `start`. Leaving one out prunes the whole branch behind it --
  drop `start -> question` and `question` and `embed_q` are never scheduled,
  so the run fails on `Retriever` with no query, two nodes from the edge that
  is actually missing. `validate_graph` does not catch it, because the other
  trigger keeps the entry-point check satisfied.

## Continuous integration

`backend/tests/test_rag_examples.py` holds this graph to its shape without
downloading anything: that the card names the rag pack inside the 80 characters
the gallery shows, that both encoders share one model with the right prefixes,
that the question reaches both consumers, and that this README exists.

`backend/tests/test_builtin_examples.py` validates the graph structurally and
**skips executing it** -- `HFTextGenerate` and `TextEmbedding` are both in
`_SLOW_NODE_TYPES`, so CI never tries to load a gigabyte of weights it does not
have.

The real run is opt-in, in `backend/tests/test_pack_examples_real.py`:

```
CODEFYUI_PACK_NETWORK_TESTS=1 pytest tests/test_pack_examples_real.py -q -s
```

`test_rag_local_example_answers_for_real` skips unless both pack items are
installed, then runs this graph and asserts the closest chunk to the shipped
question came from `02-nodes-and-edges.md` and that the generated answer is not
empty.

## Provenance

No measured run is recorded for this example. The timing in "Before you run it"
is an estimate from the model size and `max_new_tokens`, not a stopwatch. If you
run it, the wall clock and the machine belong here.

Qwen2.5-0.5B-Instruct is Qwen Team, Alibaba Cloud (Apache-2.0);
`intfloat/multilingual-e5-small` is Wang et al., *Multilingual E5 Text
Embeddings* (2024), MIT. Both are fetched by Package Center at install time and
neither is redistributed with CodefyUI.
