# Eval — evaluation and inference pipelines

Eleven nodes that turn "run these two checkpoints over this eval set and tell
me where they differ" into a graph whose every number can be traced back to
the exact weights, tokenizer, configuration and file that produced it. Text
classifiers, causal LMs and visual question answering models (ViLT-style,
image + question) are supported.

A step-by-step tutorial, **Build Your Own Eval Pipeline**, is in the
CodefyUI docs (`docs/docs/usage/eval-pipelines.md`).

```bash
cdui plugin install eval
```

Installing the pack also installs `transformers` (its `[python_deps]`,
`>=5.0,<6`) into the CodefyUI environment. The node modules import without it,
so the palette never breaks; only the two loaders need it, and they say how to
get it when it is missing. BLEU and ROUGE-L are computed by the pack itself
(`nodes/_eval_text_metrics.py`), so they add no dependency; the tests check
them against sacrebleu and rouge-score. The pack declares no capability. Its sources pass
the plugin AST gate at Tier 0 except for one import, `threading` in
`nodes/_eval_hub.py`: transformers resolves its classes lazily, which is not
safe when the two loaders of an A/B graph start at once, so the pack resolves
them once under a lock. Built-in packs skip the gate at install; a fork
published from GitHub would declare `allowed_modules = ["threading"]`.

## The pipeline

```
Eval-JSONLDataset ─► (Eval-PromptTemplate) ─┐
Eval-HFModelLoader ─► Eval-HFProcessorLoader ┴► Eval-BatchInference ─► Eval-ExactMatch / Eval-F1Score / Eval-TopKAccuracy
                                                         │              Eval-BLEU / Eval-ROUGE-L (generation)
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
| `image_field` | string | `""` *(no images)* |
| `limit` *(advanced)* | int | `0` *(all)* |

With `image_field` set, each record names an image file, relative to the JSONL
file's folder; an absolute path, a drive letter or a `..` segment is refused,
so a dataset folder is self-contained. Every image is read and hashed at load:
a missing or unreadable image is refused with the line number, each item
carries `image` (the path as written) and `image_sha256`, and the dataset's
identity gains `images_sha256`, a hash over every item's id and image hash.
The JSONL's own `sha256` cannot see an image that changed under an unchanged
name; `images_sha256` can. `eval:samples/shapes-vqa-12.jsonl` is the VQA
example's set: 12 questions about 7 PNG drawings in `samples/shapes-vqa/`,
drawn for this pack.

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
| `task` | select | `auto` | `auto`, `classification`, `generation`, `vqa` |
| `dtype` *(advanced)* | select | `float32` | `float32`, `auto`, `float16`, `bfloat16` |
| `device` *(advanced)* | select | `auto` | `auto`, `cpu`, `cuda`, `mps` |

See [revisions, the cache and offline mode](#revisions-the-cache-and-offline-mode).
`task = auto` reads the config's `architectures`: `...ForSequenceClassification`
is classification, `...ForCausalLM` / `...LMHeadModel` is generation, and
`ViltForQuestionAnswering` (spelled `ViltForVisualQuestionAnswering` in the
published ViLT configs) is vqa, loaded with `AutoModelForVisualQuestionAnswering`.
`vqa` means visual question answering as classification over the model's
answer vocabulary (`config.id2label`); BLIP's question-answering model, which
generates its answer, is not one. Loaded models are kept in-process (two at a
time, for an A/B pair).

### `Eval-HFProcessorLoader`

| | |
|---|---|
| **In** | `model` MODEL *(opt)* |
| **Out** | `processor` ANY · `provenance` ANY |

Connect the model and leave `model_id` and `revision` empty: the tokenizer is
loaded from the model's repository at the commit the model resolved to. A
`revision` that resolves to a different commit of the model's own repository
is refused. `kind` (advanced) is `tokenizer` (AutoTokenizer) or `auto`
(AutoProcessor). A `vqa` model needs `auto`, which loads its image processor
with its tokenizer; `tokenizer` is refused for one.

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
| `generation_config` *(advanced)* | select | `node` | `node` or `model`; generation |
| `do_sample`, `temperature`, `top_p`, `repetition_penalty`, `seed` *(advanced)* | | `false`, `1.0`, `1.0`, `1.0`, `0` | generation |
| `max_length`, `top_k` *(advanced)* | int | `256`, `5` | classification, vqa |

* **classification** -- softmax over the logits, argmax named by the model's
  `config.id2label` (`LABEL_i` when it has none); the `topk` column keeps up to
  five ranked label names for `Eval-TopKAccuracy`, `score` the top probability.
* **vqa** -- the same over the model's answer vocabulary, for an image and a
  question: the processor gets `images=` and `text=` together, as ViLT's does.
  The dataset needs `image_field`. Each item's image is opened when its batch
  runs, so one unreadable image is one `error` row. Rows add `image` and
  `image_sha256`.
* **generation** -- greedy or sampled decoding with a KV cache, one item at a
  time. Sampling draws from a local `torch.Generator` seeded per item from
  `sha256(seed:id)`: an item's sample does not depend on the batch size, the
  item order or anything else that touched the global RNG, which is never used.
* **generation settings** -- `generation_config = node` (the default, and what
  every graph saved before the option ran) uses this node's `do_sample`,
  `temperature`, `top_p` and `repetition_penalty`, with no top-k filter.
  `generation_config = model` takes `do_sample`, `temperature`, `top_p`,
  `top_k` and `repetition_penalty` from the model's `generation_config`, the
  settings `model.generate` would use, filling a field the checkpoint leaves
  unset with transformers' default; a model without a generation config is
  refused. `repetition_penalty` is transformers' rule: every token already in
  the prompt or the answer has its logit divided by the penalty when positive
  and multiplied by it when negative. Greedy decoding with a penalty matches
  `model.generate(do_sample=False, repetition_penalty=...)` (tested).
  `meta.generation` records the settings that ran and, under
  `model_generation_config`, what the model's own config suggests.
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
`metric_name` (default `exact_match`, `f1_macro`, `top2_accuracy`, `bleu`,
`rouge_l`, ...) at `step`, so a sweep can rank on it. Give the A and B runs
different names. The `metrics` record table's `meta.metric` names the metric
node and its params (its producer), and BLEU and ROUGE-L add `details`.

### `Eval-BLEU`, `Eval-ROUGE-L`

Same ports and the same skipping rules as the metrics above; meant for
generation, where an answer is free text.

| Node | Param | Default | Values |
|---|---|---|---|
| `Eval-BLEU` | `tokenize` | `13a` | `13a`, `zh`, `char`, `none` |
| | `smooth_method` *(advanced)* | `exp` | `exp`, `floor`, `add-k`, `none` |
| `Eval-ROUGE-L` | `measure` | `fmeasure` | `fmeasure`, `precision`, `recall` |
| | `tokenize` | `rouge` | `rouge`, `char`, `whitespace` |

* **`Eval-BLEU`** is corpus BLEU on a 0 to 100 scale, exactly
  `sacrebleu.corpus_bleu(answers, [labels], tokenize=..., smooth_method=...)`
  with sacrebleu's other defaults (case-sensitive, n-grams up to 4, no
  effective order). The answer and label are the normalised ones, so
  `normalize = strip_lower` on BatchInference makes it case-insensitive. Use
  `zh` or `char` for Chinese. `details` records sacrebleu's signature without
  its version field (`nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp`), the four
  precisions, the brevity penalty and the n-gram counts.
* **`Eval-ROUGE-L`** is the mean over the counted rows of
  `rouge_score.rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False).score(label, answer)`
  (0 to 1). `tokenize = rouge` is rouge-score's tokenizer, which lowercases and
  treats anything outside `a-z0-9` as a separator, so text with no ASCII
  letters or digits has no tokens and scores 0; `char` scores such text by
  characters. `details` counts the rows that scored 0.

Both are implemented in `nodes/_eval_text_metrics.py`. The parity tests in
`backend/tests/test_eval_text_metrics.py` compare them with sacrebleu and
rouge-score (installed by the backend's `dev` extra, for the tests only) over
random fixtures that hit every tokenizer rule, including the two entries of
sacrebleu's `zh` range table that are written as two-character strings.

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
| `input_hash` | first 16 hex digits of the sha256 of `input` -- the input's identity; for a vqa row, of `input` and the image's sha256 together |
| `raw_answer` | what the model produced, before any mapping |
| `answer` | `raw_answer` after `label_map` and normalisation; `null` unless `ok` |
| `label` | the normalised label, or `null` |
| `status` | `ok`, `error` or `cancelled` |
| `error` | why, for `error` and `cancelled` rows |

Extras follow: `score`, `topk` (classification); `score`, `topk`, `image`,
`image_sha256` (vqa); or `new_tokens` (generation).

`meta` records where the rows came from:

| Key | Contents |
|---|---|
| `producer` | `node`, `model_id`, `requested_revision`, `model_revision`, `revision_source`, `processor_id`, `processor_revision`, `processor_revision_source`, `model_class`, `processor_class`, `dtype`, `device`, `normalize`, `label_map` |
| `inference` | the effective `task`, `batch_size` and (classification, vqa) `max_length`, `top_k` |
| `generation` | generation only: `decoding`, `source` (`node` or `model`), `max_new_tokens`, `do_sample`, `temperature`, `top_p`, `top_k`, `repetition_penalty`, `seed`, `seed_scope`, `eos_token_id`, and `model_generation_config` (what the model's own config suggests, with the fields it left `unset`) |
| `seed` | the seed param |
| `dataset` | `path`, `sha256`, `rows`, `used`, `template` (text and sha256, if one was applied), and `images_sha256` when the items have images |
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

## Examples

### A/B text classifier eval

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

### VQA eval skeleton

`examples/Eval/VQA-Shapes/` -- **VQA eval skeleton**, the image-and-question
stand-in for a video-LM evaluation. ViLT (`dandelin/vilt-b32-finetuned-vqa`,
pinned to commit `d0a1f6a`, about 470 MB, Apache-2.0) answers the 12 questions
of `samples/shapes-vqa-12.jsonl` about the 7 drawings in
`samples/shapes-vqa/`, scored by `Eval-ExactMatch` and `Eval-TopKAccuracy`
(`k = 3`). The processor loader uses `kind = auto`. Expected, CPU:

| | value |
|---|---|
| exact match | 0.5000 (6 of 12) |
| top-3 accuracy | 0.8333 (10 of 12) |

ViLT names the triangle, answers the yes/no question about a square, counts
one and two balls, and gets red and orange balls right. It misses three
colours (red read as orange twice, blue as yellow), two counts and one yes/no
question. ViLT learned from VQAv2 photographs, and flat drawings on white are
far from them, so read the numbers as the pipeline at work. The example needs
no other pack.

To evaluate your own model the same way, point `Eval-JSONLDataset` at your
JSONL (with `image_field`) and the loader at your checkpoint. The `vqa` task
covers models that choose from an answer vocabulary; a generative
vision-language model, and video frames, are not tasks of this pack yet.

## Tests

* `backend/tests/test_eval_nodes.py` -- offline, CPU, no transformers: metric
  parity with scikit-learn, CompareRuns alignment and refusals, Stop and
  per-item errors, provenance and reproducibility, revision resolution, the
  generation settings (`generation_config`, `repetition_penalty`, top-k).
* `backend/tests/test_eval_text_metrics.py` -- BLEU and ROUGE-L parity with
  sacrebleu and rouge-score on random fixtures, and the two metric nodes.
* `backend/tests/test_eval_vqa.py` -- offline, no transformers: images in the
  dataset and its identity, the vqa rows and their input hash, per-item image
  errors, Stop, and the VQA example graph run end to end through the engine on
  its bundled images with a ViLT-shaped stand-in for the two loaders.
* `backend/tests/test_eval_hf_local.py` -- real tiny random-init
  `BertForSequenceClassification` / `GPT2LMHeadModel` /
  `ViltForQuestionAnswering` saved to a temp dir; skipped without
  transformers. Generation with a repetition penalty and with
  `generation_config = model` is compared with `model.generate`. The Hub
  checks (both examples' numbers, and a branch resolved offline from the
  cache) run only with `CODEFYUI_PACK_NETWORK_TESTS=1`.
* `backend/tests/test_eval_pack.py` -- catalog, manifest, discovery, the AST
  gate, the zh-TW node entries and the A/B example graph's wiring.
