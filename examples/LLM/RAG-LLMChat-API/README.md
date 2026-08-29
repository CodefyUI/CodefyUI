# RAG with a chat API: retrieve locally, generate anywhere

The same retrieval chain as
[RAG-Local-Offline](../RAG-Local-Offline/README.md), with the last box swapped.
`HFTextGenerate` loads a model into this process; `LLMChat` sends the prompt to
a server. Everything before it -- the loader, the chunker, both encoders, the
store, the retriever and the prompt builder -- is identical node for node and
edge for edge, and a test asserts that it stays that way.

That is the lesson. **Retrieval is local work and generation is a swappable
back end.** The chunks that reach the model are chosen on your machine by a
470 MB encoder, and which model writes the sentence afterwards is a dropdown.

Out of the box `gen` talks to a local Ollama server, so this variant still
sends nothing off the machine; switching `provider` to `ChatGPT API` or
`Claude API` and supplying a key sends the identical prompt to a hosted model
instead.

```
loader -> chunker -> embed_docs -> store ---+
                     ("passage: ")          |
                                            v
question -> embed_q ---------------------> retriever -> prompt -> gen -> print_ans
            ("query: ")                                          (LLMChat)
```

Three things join that from the side, exactly as in the local example:
`chunker.chunks` and `chunker.metadata` into `store`, `question.text` into
`prompt.question`, and `retriever.contexts` into `print_ctx`.

## The stages

| Node | What it does | The knob to try |
|---|---|---|
| `loader` `DocumentLoader` | Reads every `.md` and `.txt` in `data/samples/rag` and emits `{text, source}` per file | `directory` -- point it at your own folder |
| `chunker` `TextChunker` | Cuts each note into 400-character windows overlapping by 80, carrying `source` and the character offsets | `chunk_size` |
| `embed_docs` `TextEmbedding` | One vector per chunk, prefixed `passage: ` for multilingual-e5 | `model` -- change it on **both** encoders or not at all |
| `store` `VectorStore` | Stacks the vectors into one `[N, D]` matrix with the chunk texts and metadata beside it | `metric` |
| `question` `TextInput` | The question, used twice: embedded, and written into the prompt | `value` -- this is the field to edit |
| `embed_q` `TextEmbedding` | The question's vector, prefixed `query: ` | `prefix` -- must stay `query: ` |
| `retriever` `Retriever` | The 3 nearest chunks and the files they came from | `top_k`, `min_score` |
| `print_ctx` `Print` | Shows what was retrieved *before* the model sees it | `label` |
| `prompt` `PromptBuilder` | Numbers the chunks, cites their sources, and wraps them in an answer-only-from-the-context template | `template` |
| `gen` `LLMChat` | Sends that prompt to a chat model and streams the answer back | `provider`, `model`, `max_tokens` |
| `print_ans` `Print` | The answer | `label` |

`prompt.prompt` lands on `gen.`**`text`** here, not on `gen.prompt`.
`LLMChat`'s `prompt` *param* is prepended to whatever arrives on the `text`
input, so it is deliberately left **empty** in this graph -- a non-empty param
would glue a second instruction on top of the retrieval prompt the
`PromptBuilder` just assembled.

## Before you run it

| | |
|---|---|
| Pack | The `multilingual-e5-small` item of the **Sentence embeddings** pack, from Package Center. Only the retrieval half needs a download here -- the RAG stack pack is *not* required, because no model is loaded into this process. |
| Disk | About 470 MB for that one item; about 1.1 GB if you install the whole pack and get all four encoders. |
| Generator | A running [Ollama](https://ollama.com), with the model pulled: `ollama pull qwen2.5:0.5b`. `gen.ollama_base_url` defaults to `http://127.0.0.1:11434/v1`, the OpenAI-compatible endpoint Ollama serves. |
| Network | Once for the encoder download. The Ollama path then stays on localhost; the hosted providers do not. |
| Time | The encode is a couple of seconds on CPU. The generation is however long the server takes -- `qwen2.5:0.5b` on Ollama is a few seconds, a hosted model a second or two of network. **Estimates, not recorded measurements.** |

### Using a hosted model instead

Set `gen.provider` to `ChatGPT API` or `Claude API` and put a model id in
`gen.model` (`gpt-5.2`, `claude-sonnet-4-6`, ...). The key comes from the
environment -- `OPENAI_API_KEY` / `CODEFYUI_OPENAI_API_KEY`, or
`ANTHROPIC_API_KEY` / `CODEFYUI_ANTHROPIC_API_KEY` -- or from the node's own
key field, which is a `SECRET` param: the canvas clears it on save and it is
never written to disk.

**This file carries no key field at all**, not even an empty one, and a test
enforces that. A `SECRET` param sitting in a committed example is somewhere to
paste a key by accident.

Remember that the hosted path sends the retrieved chunks to a third party. That
is fine for the five bundled notes about CodefyUI, and it is a decision worth
making on purpose the moment `loader.directory` points at your own material.

## Try this

**Compare the two examples.** Run `RAG-Local-Offline` and this one on the same
question and read the two answers next to each other. The retrieved contexts
are identical -- same encoder, same index, same `top_k` -- so every difference
you see is the generator. A bigger model usually writes a tidier sentence; it
does not retrieve better, because it does not retrieve at all.

**Ask something else.** Edit `question.value`. Three the bundled corpus can
answer:

- `What is the difference between a data edge and a trigger edge?`
- `Why does a training run need a DataLoader?`
- `什麼是套件中心？` -- every note has a Traditional Chinese half, and the prompt
  template says "Answer in the language of the question", so the answer should
  come back in Chinese. Whether it does is a property of the *generator*, which
  makes it a good question to run in both examples.

**Swap the embedding model.** Change `model` on **both** `embed_docs` and
`embed_q` to `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` and
clear `prefix` on both (only the e5 models were trained with prefixes).
Changing one and not the other is the interesting failure: two models are two
vector spaces, so the search still returns three chunks and they are three
arbitrary chunks. Nothing in the graph can tell.

**Raise `top_k`, then switch provider.** Out of the box `gen` runs the same
0.5B model as the local example, so 8 or 10 chunks costs the same there as it
does here. Point it at a hosted model and the extra context becomes affordable
-- that is the comparison worth making, because it is the first thing the
bigger model actually buys you. Watch whether the answer improves: more context
is also more room to answer from the wrong chunk. `max_tokens` is the *output*
budget and does not change any of this.

## The corpus

`data/samples/rag` ships five short notes -- what CodefyUI is, nodes and edges,
training basics, embeddings and RAG, and optional packs -- each with an English
half and a Traditional Chinese half.

**`DocumentLoader` reads every `.md` and `.txt` in the folder it is pointed at**
(those two extensions only). Point `loader.directory` at a folder of your own
notes and nothing else in the graph changes. `recursive` adds subfolders and
`max_docs` caps how many files are read.

## Notes for anyone editing this graph

- **Keep the retrieval half identical to `RAG-Local-Offline`.**
  `test_rag_examples_share_the_retrieval_chain` compares every node except
  `gen`, plus every edge that does not touch it. If the two chains drift, the
  comparison above stops being a comparison of generators.
- **`gen.prompt` stays empty.** See the note under the stage table.
- **The key params stay out of the file.** Use the environment, or type the key
  into the node for the session.
- **`Ollama` is the OpenAI-compatible path.** The provider maps to the same
  adapter as `ChatGPT API`, which is why `ollama_base_url` ends in `/v1` and
  why no key is required for it.
- **Both root nodes need a trigger edge.** `loader` and `question` are each
  wired from `start`; execution walks forward along data edges from there.

## Continuous integration

`backend/tests/test_rag_examples.py` holds this graph to its shape with no
download and no network: the card names both the pack and Ollama inside the 80
characters the gallery shows, the two encoders share one model with the right
prefixes, the question reaches both consumers, no `SECRET` param appears
anywhere in the file, and the retrieval chain matches `RAG-Local-Offline`
node-for-node.

`backend/tests/test_builtin_examples.py` validates the graph structurally and
**skips executing it** -- `LLMChat` and `TextEmbedding` are both in
`_SLOW_NODE_TYPES`, so no smoke run ever opens a socket or spends anyone's
money.

There is no opt-in real run for this variant: it would need a live Ollama or a
funded key, neither of which a maintainer's machine can be assumed to have.
`test_rag_local_example_answers_for_real` covers the shared retrieval half
against the real encoder, and that half is the same one this graph uses.

## Provenance

No measured run is recorded for this example; the timings above are estimates.
`intfloat/multilingual-e5-small` is Wang et al., *Multilingual E5 Text
Embeddings* (2024), MIT. Ollama and the hosted providers are third-party
services and nothing about them is redistributed here.
