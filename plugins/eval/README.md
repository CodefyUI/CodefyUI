# Eval — evaluation and inference pipelines

Nine nodes that turn "run these two checkpoints over this eval set and tell
me where they differ" into a graph whose every number can be traced back to
the exact weights, tokenizer, configuration and file that produced it.

```bash
cdui plugin install eval
```

Installing the pack also installs `transformers` (its `[python_deps]`,
`>=5.0,<6`) into the CodefyUI environment. The node modules import without it,
so the palette never breaks; only the two loaders need it, and they say how to
get it when it is missing. The pack declares no capability. Its sources pass
the plugin AST gate at Tier 0 except for one import, `threading` in
`nodes/_eval_hub.py`: transformers resolves its classes lazily, which is not
safe when the two loaders of an A/B graph start at once, so the pack resolves
them once under a lock. Built-in packs skip the gate at install; a fork
published from GitHub would declare `allowed_modules = ["threading"]`.

## The pipeline

```
Eval-JSONLDataset ─► (Eval-PromptTemplate) ─┐
Eval-HFModelLoader ─► Eval-HFProcessorLoader ┴► Eval-BatchInference ─► Eval-ExactMatch / Eval-F1Score / Eval-TopKAccuracy
                                                         │
                         (a second run, same shape) ─────┴► Eval-CompareRuns ─► Stats-TableView / Stats-ChartView
```

## Node reference

Ports marked *(opt)* are optional. "Record table" and "eval dataset" are
defined in [the contracts](#the-contracts) below.

### `Eval-JSONLDataset`

| | |
|---|---|
| **In** | — |
| **Out** | `dataset` ANY *(eval dataset)* · `labels` LIST |

One JSON object per line. Each item gets a stable `id` (the record's
`id_field`, or `line-N` when the file has none), an `input` (`input_field`,
must be non-empty text) and a `label` (`label_field`, optional per row;
integral numbers become `"1"`, booleans `"true"`). Duplicate ids, blank inputs
and malformed lines are refused with the line number. The dataset's identity --
`path`, `sha256` of the file's bytes, `rows` -- travels into every prediction
table built from it.

A path starting with `eval:` names a file inside this pack:
`eval:samples/sst2-mini-20.jsonl` is the 20-sentence sentiment set the example
uses (original sentences written for this pack, 10 positive, 10 negative). Any
other path resolves the way `CSVReader` resolves one: an uploaded file by bare
name, else relative to the backend working directory (or the project directory
in project mode).

| Param | Type | Default |
|---|---|---|
| `path` | data file | `""` |
| `input_field` | string | `text` |
| `label_field` | string | `label` |
| `id_field` | string | `id` |
| `limit` *(advanced)* | int | `0` *(all)* |

### `Eval-PromptTemplate`

| | |
|---|---|
| **In** | `dataset` ANY |
| **Out** | `dataset` ANY |

`{input}` is the item's current input, `{name}` any field of its original
record; `{{` and `}}` are literal braces. Only plain field names are allowed
(no `{x.attr}` or `{x[0]}`). The rendered text replaces the input -- so it is
what the model sees and what `input_hash` records -- and the template's text
and sha256 join the dataset's identity. A template that uses the label field
is refused: it would show the model the answer.

### `Eval-HFModelLoader`

| | |
|---|---|
| **In** | — |
| **Out** | `model` MODEL · `provenance` ANY |

| Param | Type | Default | Values |
|---|---|---|---|
| `model_id` | string | `""` | Hub id (`org/name`) or a local directory |
| `revision` | string | `main` | branch, tag or commit sha |
| `task` | select | `auto` | `auto`, `classification`, `generation` |
| `dtype` *(advanced)* | select | `float32` | `float32`, `auto`, `float16`, `bfloat16` |
| `device` *(advanced)* | select | `auto` | `auto`, `cpu`, `cuda`, `mps` |

See [revisions, the cache and offline mode](#revisions-the-cache-and-offline-mode).
`task = auto` reads the config's `architectures`: `...ForSequenceClassification`
is classification, `...ForCausalLM` / `...LMHeadModel` is generation. Loaded
models are kept in-process (two at a time, for an A/B pair).

### `Eval-HFProcessorLoader`

| | |
|---|---|
| **In** | `model` MODEL *(opt)* |
| **Out** | `processor` ANY · `provenance` ANY |

Connect the model and leave `model_id` and `revision` empty: the tokenizer is
loaded from the model's repository at the commit the model resolved to. A
`revision` that resolves to a different commit of the model's own repository
is refused. `kind` (advanced) is `tokenizer` (AutoTokenizer) or `auto`
(AutoProcessor).

### `Eval-BatchInference`

| | |
|---|---|
| **In** | `model` MODEL · `processor` ANY · `dataset` ANY |
| **Out** | `predictions` ANY *(prediction table)* · `answers` LIST · `labels` LIST |

| Param | Type | Default | Notes |
|---|---|---|---|
| `task` | select | `auto` | follows the loader's recorded task |
| `batch_size` | int | `8` | progress and Stop are checked between batches |
| `normalize` | select | `strip_lower` | `strip_lower`, `strip`, `none` |
| `label_map` | string | `""` | `LABEL_0=negative, LABEL_1=positive` |
| `max_new_tokens` | int | `32` | generation |
| `do_sample`, `temperature`, `top_p`, `seed` *(advanced)* | | `false`, `1.0`, `1.0`, `0` | generation |
| `max_length`, `top_k` *(advanced)* | int | `256`, `5` | classification |

* **classification** -- softmax over the logits, argmax named by the model's
  `config.id2label` (`LABEL_i` when it has none); the `topk` column keeps up to
  five ranked label names for `Eval-TopKAccuracy`, `score` the top probability.
* **generation** -- greedy or sampled decoding with a KV cache, one item at a
  time. Sampling draws from a local `torch.Generator` seeded per item from
  `sha256(seed:id)`: an item's sample does not depend on the batch size, the
  item order or anything else that touched the global RNG, which is never used.
* `label_map` renames the raw answer before normalising; `raw_answer` keeps
  what the model said.
* `answers` / `labels` are the normalised answers and labels of the `ok`,
  labelled rows, aligned -- ready for `Stats-ConfusionMatrix`.

### `Eval-ExactMatch`, `Eval-F1Score`, `Eval-TopKAccuracy`

| | |
|---|---|
| **In** | `predictions` ANY |
| **Out** | `value` SCALAR · `table` TENSOR `[1, 3]` · `columns` LIST · `row_labels` LIST · `metrics` ANY *(record table)* |

Each scores the rows whose `status` is `ok` and whose `label` is present, and
reports `counted` and `skipped` beside the value: an error, a cancelled item or
an unlabelled row is skipped, never scored as wrong. Parity with scikit-learn
is tested: `accuracy_score`, `f1_score(average="micro" | "macro",
zero_division=0)` and `top_k_accuracy_score` (no ties). `Eval-TopKAccuracy`
reads the `topk` column and skips rows without one (generation rows).

The value is also logged as a run metric (`context.log_metric`) under
`metric_name` (default `exact_match`, `f1_macro`, `top2_accuracy`, ...) at
`step`, so a sweep can rank on it. Give the A and B runs different names.

### `Eval-CompareRuns`

| | |
|---|---|
| **In** | `table_a` ANY · `table_b` ANY |
| **Out** | `diff` ANY · `diff_table` TENSOR · `diff_columns` LIST · `diff_ids` LIST · `summary_table` TENSOR `[4, 2]` · `summary_columns` LIST · `summary_rows` LIST · `agreement` SCALAR · `summary` ANY · `chart` ANY *(`media=chart`)* |

Rows are matched on `key_column` (default `id`), never by position. Refused,
naming the first offenders:

* a key that appears twice in either table;
* key sets that differ (`N id(s) only in table_a (first: [...])`);
* the same key with different `input_hash` (turn `require_same_input` off to
  compare two prompts on purpose);
* the same key with different labels.

An item is compared only when both runs answered it (`ok` on both sides); the
others stay in the diff with their statuses and empty comparison cells. The
diff columns are `id, answer_a, answer_b, label, agree, a_correct, b_correct,
status_a, status_b`. `summary_table` rows are `both_correct`, `only_a_correct`,
`only_b_correct`, `both_wrong`, with columns `count` and `share` (of the
compared, labelled items) -- connect the three `summary_*` ports to
`Stats-ChartView` or `Stats-TableView`. `chart` is the agreement matrix, A's
answers by row against B's by column (a bar chart of the four outcomes when
there are more than 20 distinct answers). The Log shows the diff with
disagreements first.

---

## The contracts

### Record tables

The [stats pack's table contract](../stats/README.md#the-table-contract) is a
numeric 2D `TENSOR` plus a `columns` LIST, and it keeps non-numeric data out of
`table`. A prediction is text, so this pack carries its tables as one JSON-safe
dict on an `ANY` port, with the same column/row vocabulary:

```jsonc
{
  "format": "codefyui.eval.table",
  "version": 1,
  "columns": ["id", "input", "input_hash", "raw_answer", "answer", "label", "status", "error", "score", "topk"],
  "rows": [["s01", "a warm, funny film ...", "3f0c...", "positive", "positive", "positive", "ok", null, 0.98, ["positive", "negative"]]],
  "meta": { ... }
}
```

Every node that produces a record table also produces its numeric projection
in the stats contract (`TENSOR` + `columns` + `row_labels`, `NaN` where a cell
is missing), which is what `Stats-TableView` and `Stats-ChartView` render.

### Prediction tables

The fixed columns, first and in this order:

| Column | Meaning |
|---|---|
| `id` | the item's stable id from the dataset |
| `input` | the exact text the model saw (after any template) |
| `input_hash` | first 16 hex digits of the sha256 of `input` -- the input's identity |
| `raw_answer` | what the model produced, before any mapping |
| `answer` | `raw_answer` after `label_map` and normalisation; `null` unless `ok` |
| `label` | the normalised label, or `null` |
| `status` | `ok`, `error` or `cancelled` |
| `error` | why, for `error` and `cancelled` rows |

Extras follow: `score`, `topk` (classification) or `new_tokens` (generation).

`meta` records where the rows came from:

| Key | Contents |
|---|---|
| `producer` | `node`, `model_id`, `requested_revision`, `model_revision`, `revision_source`, `processor_id`, `processor_revision`, `processor_revision_source`, `model_class`, `processor_class`, `dtype`, `device`, `normalize`, `label_map` |
| `inference` | the effective `task`, `batch_size` and (classification) `max_length`, `top_k` |
| `generation` | generation only: `decoding`, `max_new_tokens`, `do_sample`, `temperature`, `top_p`, `seed`, `seed_scope`, `eos_token_id` |
| `seed` | the seed param |
| `dataset` | `path`, `sha256`, `rows`, `used`, `template` (text and sha256, if one was applied) |
| `counts` | rows per status |
| `interrupted_at_batch` | the batch a Stop landed before, else `null` |
| `reproducibility` | `{"reproducible": bool, "reasons": [...]}` |

`reproducible` is true only when the model revision and the processor revision
are each pinned to a commit sha or a local content hash, the dataset has a
content hash, and sampled generation has a seed. A model id alone never
qualifies; `reasons` lists what is missing.

### Failure behaviour

* An item's record is written once. A finished batch's items are `ok` and stay
  `ok` whatever happens to later batches.
* A batch that raises is retried one item at a time, so one bad input marks
  one item `error` with the exception text, and its batch mates still finish.
* Stop is honoured between batches: every item not yet started is
  `cancelled`, the table still holds every item, and the node returns as
  interrupted (`__interrupted__`), so the partial table stays inspectable on
  the canvas. The metrics skip the cancelled rows and say how many.

## Revisions, the cache and offline mode

A requested revision is resolved to one exact snapshot before anything loads,
and both the request and the answer are recorded:

| `revision_source` | Meaning |
|---|---|
| `hub` | the Hub was asked what the branch or tag points at now |
| `pinned` | the revision was already a full commit sha |
| `cache` | the Hub was off or unreachable, so the cached snapshot's sha is used; `revision_note` says why |
| `local` | the id is a local directory, identified as `local-sha256:<hash>` of its model and tokenizer files |

The snapshot is fetched with `huggingface_hub.snapshot_download` -- only the
configs, vocabularies and one weights format (safetensors when the repository
has it, else `pytorch_model*.bin`) -- and the model and tokenizer are loaded
from that `snapshots/<sha>` directory, never by repo id. Loading a `.bin`-only
repository by id would also start transformers' background request asking the
Hub's converter bot to open a safetensors pull request on somebody else's
repository; loading from the directory does not.

Downloads go to CodefyUI's own Hugging Face cache, `<cache>/hf` under the user
data directory (the directory the Package Center uses, honouring
`CODEFYUI_USER_DATA_DIR`), never to `HF_HOME`. Start the server with
`HF_HUB_OFFLINE=1` and nothing is fetched: a branch name resolves to the
snapshot cached under it, a commit sha to its snapshot, and anything not cached
is an error that says so. huggingface_hub reads the variable once at import,
so set it before starting the server.

## Example

`examples/Eval/AB-Text-Classification/` -- **A/B text classifier eval**. Two
two-layer BERT sentiment classifiers, `philschmid/tiny-bert-sst2-distilled`
(A) and `M-FAC/bert-tiny-finetuned-sst2` (B), each pinned to a commit, run over
the bundled 20-sentence set; B's config has no label names, so its
`label_map` renames `LABEL_0`/`LABEL_1`. Expected, CPU, in a few seconds once
downloaded (about 18 MB each):

| | A | B |
|---|---|---|
| exact match | 1.0000 | 0.9500 |
| macro F1 | 1.0000 | 0.9499 |

They agree on 19 of 20 items (agreement 0.9500); the one difference is `s15`,
which A calls positive and B negative, so the outcomes are 19 both correct, 1
only A correct, 0 only B, 0 both wrong. The models are tiny: the example
demonstrates the pipeline, not either model's quality. The outcome chart and
table are `stats` pack nodes, so install `stats` as well.

## Tests

* `backend/tests/test_eval_nodes.py` -- offline, CPU, no transformers: metric
  parity with scikit-learn, CompareRuns alignment and refusals, Stop and
  per-item errors, provenance and reproducibility, revision resolution.
* `backend/tests/test_eval_hf_local.py` -- real tiny random-init
  `BertForSequenceClassification` / `GPT2LMHeadModel` saved to a temp dir;
  skipped without transformers. The Hub checks (the example's numbers, and a
  branch resolved offline from the cache) run only with
  `CODEFYUI_PACK_NETWORK_TESTS=1`.
* `backend/tests/test_eval_pack.py` -- catalog, manifest, discovery, the AST
  gate, the zh-TW node entries and the example graph's wiring.

## Not in this version

BLEU and ROUGE-L, a VQA-style example, and a "build your own eval pipeline"
tutorial page are tracked in issue #139.
