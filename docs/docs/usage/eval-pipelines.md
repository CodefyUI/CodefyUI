---
sidebar_position: 3.95
title: Build Your Own Eval Pipeline
description: Evaluate a Hugging Face model on your own data with the eval pack — an eval set with ids and hashes, a pinned model, a prediction table, metrics and an item-aligned A/B comparison.
---

# Build Your Own Eval Pipeline

The `eval` plugin pack turns a model evaluation into a graph: read an eval set, load a model at one exact revision, run it over every item, score the answers and compare two runs item by item. Every number the graph prints can be traced back to the weights, the tokenizer, the settings and the file that produced it.

This page builds that graph from your own data, one node at a time. The two example graphs the pack ships, **A/B text classifier eval** and **VQA eval skeleton**, are the finished versions of what it describes; open either one from the Examples gallery to follow along. The node-by-node reference, including every column of the prediction table, is the pack's [README](https://github.com/CodefyUI/CodefyUI/blob/main/plugins/eval/README.md).

## Before you start {/* #before-you-start */}

Install the pack from the [Plugin Center](./plugin-center.md), or from a terminal:

```bash
cdui plugin install eval
cdui plugin install stats   # optional: charts and tables for the results
```

Installing `eval` also installs `transformers`, the one package the Hugging Face loaders need. Models download on first use into CodefyUI's own cache under the user data directory. Start the server with `HF_HUB_OFFLINE=1` to use only what is already cached; nothing is downloaded then, and a model that is not cached is an error that says so.

## 1. Write the eval set {/* #write-the-eval-set */}

An eval set is a JSONL file: one JSON object per line, one line per item.

```json
{"id": "s01", "text": "A warm, funny film with a big heart.", "label": "positive"}
{"id": "s02", "text": "Two hours I will not get back.", "label": "negative"}
```

Add an **Eval-JSONLDataset** node and set `path` to the file: an uploaded file by its name, a path relative to the backend working directory (the project directory in project mode), or `eval:samples/...` for a file the pack ships. Name the fields with `input_field` (the text the model sees), `label_field` (the expected answer; optional per row) and `id_field`.

- **Ids are the item's identity.** Two runs are compared by id, never by row position, so a duplicated id is refused at load. A file without ids gets `line-N`, which changes when lines move; give your items ids.
- **The file has an identity too.** The dataset records the path, the sha256 of the file's bytes and the row count, and every prediction table built from it carries them.
- **Images.** For visual question answering, add an image path per record and set `image_field` to that field. The path is relative to the JSONL file's folder and may not contain `..` or be absolute, so a dataset folder is self-contained. Each image is read and hashed at load; a missing image is refused with its line number, and the set gains `images_sha256`, so an image that changed under an unchanged name changes the set's identity.

```json
{"id": "q01", "image": "shapes-vqa/red-ball.png", "question": "What color is the ball?", "answer": "red"}
```

## 2. Template the prompt (optional) {/* #template-the-prompt */}

**Eval-PromptTemplate** rewrites each item's input, for example `Review: {input}\nSentiment:` for a causal LM. `{input}` is the current input and `{field}` any field of the record. The rendered text is what the model sees and what the input hash records; the template's text and hash join the dataset's identity. A template that uses the label field is refused, because it would show the model the answer.

## 3. Load the model at a fixed revision {/* #load-the-model */}

**Eval-HFModelLoader** takes a Hugging Face Hub id (`org/name`) or a local model directory, and a `revision`. The revision is resolved to a commit sha before anything loads, and the model is loaded from that exact snapshot. Set `revision` to a commit sha to keep a graph's numbers fixed: `main` moves whenever the model's owner pushes. A local directory is identified by the sha256 of its model and tokenizer files.

`task` decides how the model is used. `auto` reads the model's config:

| `task` | Models | Answer |
|---|---|---|
| `classification` | `...ForSequenceClassification` | the most likely label, named by the model's `id2label` |
| `generation` | causal LMs (`...ForCausalLM`, `...LMHeadModel`) | the text decoded after the prompt |
| `vqa` | `ViltForQuestionAnswering` | the most likely answer from the model's answer vocabulary, for an image and a question |

Connect the model to an **Eval-HFProcessorLoader** and leave its `model_id` and `revision` empty: the tokenizer is then loaded from the model's repository at the same commit. A processor from another commit of the same repository is refused. For a `vqa` model, set `kind` to `auto` so the image processor is loaded with the tokenizer; a bare tokenizer is refused for it.

## 4. Run batched inference {/* #run-batched-inference */}

**Eval-BatchInference** takes the model, the processor and the dataset, and produces a **prediction table**: one row per item with `id`, `input`, `input_hash`, `raw_answer`, the normalised `answer`, `label`, `status` and `error`. Classification and VQA rows add `score` and `topk`; VQA rows also record `image` and `image_sha256`, and their `input_hash` covers the image as well as the question. For VQA, `max_length` is capped at the model's text length (`config.max_position_embeddings`, 40 for ViLT), so a longer question is truncated.

- `normalize` (default `strip_lower`) is applied to answers and labels before any comparison. `label_map` renames raw answers first, for a model whose config has no label names: `LABEL_0=negative, LABEL_1=positive`.
- **Status.** A finished batch's items are `ok` and stay `ok`. A batch that raises is retried one item at a time, so one bad input (an unreadable image, say) is one `error` row. Press Stop and the run ends between batches: the items not yet started are `cancelled`, the table still holds every item, and the metrics skip the unfinished rows and say how many.
- **Generation settings.** By default generation is greedy and uses this node's parameters (`do_sample`, `temperature`, `top_p`, `repetition_penalty`, `seed`). Set `generation_config` to `model` to take `do_sample`, `temperature`, `top_p`, `top_k` and `repetition_penalty` from the model's `generation_config`, the settings `model.generate` would use: transformers reads it from the checkpoint's `generation_config.json`, or derives it from `config.json` when the checkpoint has none. Either way the table records the settings that ran and, beside them, the ones the model suggests. Sampling draws from a generator seeded per item from `seed` and the item's id, so an item's answer does not depend on the batch size or the item order.

## 5. Score it {/* #score-it */}

Connect the prediction table to one or more metric nodes. Each scores the rows that are `ok` and have a label, reports `counted` and `skipped` beside the value, and logs the value as a run metric under `metric_name`.

| Node | Value | For | Checked against |
|---|---|---|---|
| **Eval-ExactMatch** | share of answers equal to the label | every task | `sklearn.metrics.accuracy_score` |
| **Eval-F1Score** | micro or macro F1 | classification, VQA | `sklearn.metrics.f1_score` |
| **Eval-TopKAccuracy** | share of labels in the top `k` answers | classification, VQA | `sklearn.metrics.top_k_accuracy_score` |
| **Eval-BLEU** | corpus BLEU, 0 to 100 | generation | `sacrebleu.corpus_bleu` |
| **Eval-ROUGE-L** | mean ROUGE-L F-measure (or precision, recall), 0 to 1 | generation | `rouge_score` `RougeScorer(["rougeL"])` |

The pack computes BLEU and ROUGE-L itself, so it installs no extra packages for them, and its tests compare every value with the reference implementation. Two settings matter for non-English text:

- **Eval-BLEU** `tokenize` is sacrebleu's `13a` by default; use `zh` (Chinese characters split) or `char` for Chinese. BLEU scores the normalised answers and labels, and Eval-BatchInference's default `normalize = strip_lower` lowercases both, so by default the score is case-insensitive; `normalize = strip` or `none` keeps the case. The metrics table records sacrebleu's signature to match, such as `nrefs:1|case:lc|eff:no|tok:13a|smooth:exp` for the default and `case:mixed` otherwise.
- **Eval-ROUGE-L** `tokenize` `rouge` is rouge-score's tokenizer, which keeps only `a-z` and `0-9`: a Chinese answer has no tokens and scores 0. Use `char` for such text.

Give each run's metric its own `metric_name` (`exact_match_a`, `exact_match_b`), so a sweep or the run history can tell them apart.

## 6. Compare two runs {/* #compare-two-runs */}

Build the same chain a second time with another model, or the same model at another revision, and connect both prediction tables to **Eval-CompareRuns**. Rows are matched on `id`, and the comparison is refused, naming the first offenders, when:

- an id appears twice in either table;
- the id sets differ;
- the same id has a different `input_hash` (turn `require_same_input` off to compare two prompts on purpose);
- the same id has a different label.

The diff lists every item with both answers, whether they agree and which side is correct; an item that did not finish on one side stays in the diff with its status. `summary_table`, `summary_columns` and `summary_rows` follow the stats pack's table contract, so they connect to **Stats-ChartView** and **Stats-TableView** unchanged; `chart` is the agreement matrix.

## Reading the provenance {/* #reading-the-provenance */}

The prediction table's `meta` holds what a reader needs to trust or reproduce it: the model and processor ids with the requested and resolved revisions, the effective inference and generation settings, the seed, and the dataset's path, sha256, row counts, template and (for images) `images_sha256`. `meta.reproducibility` is true only when both revisions are pinned to a commit or a content hash, the dataset has a hash, and sampled generation has a seed; otherwise `reasons` lists what is missing. A model id alone never counts as reproducible.

## The two shipped examples {/* #the-shipped-examples */}

| Example | What it shows | Result |
|---|---|---|
| **A/B text classifier eval** | two tiny BERT sentiment checkpoints over 20 sentences, compared item by item | exact match 1.0000 and 0.9500; they disagree only on `s15` |
| **VQA eval skeleton** | ViLT (`dandelin/vilt-b32-finetuned-vqa`) over 12 questions about 7 drawn pictures | exact match 0.5000, top-3 accuracy 0.8333 |

Both pin their checkpoints to a commit. The first run downloads about 18 MB per sentiment checkpoint, or about 470 MB for ViLT. ViLT learned from photographs, and the flat drawings in the sample are far from them, so the VQA example's score says how the pipeline reports a model and little about ViLT itself.

## From VQA to a video-LM evaluation {/* #from-vqa-to-video */}

The VQA example is the skeleton of a video-language-model evaluation: a visual input and a question in, an answer out, scored against a label, with the visual input hashed into each item's identity. To evaluate your own model, replace the JSONL and its images with your own set, and the loader's `model_id` and `revision` with your checkpoint.

The `vqa` task covers models that choose an answer from a fixed vocabulary, as ViLT does. A generative vision-language model, which writes its answer as text, and video input (frames sampled from a clip) are not tasks of this pack yet.

## Checks and limits {/* #checks-and-limits */}

- The example graphs and the metrics run on CPU. Large models work on a GPU through the loader's `device`.
- The HF loaders keep the two most recently loaded models in memory, for an A/B pair.
- `Eval-BatchInference` is not cacheable: a run always re-evaluates, because its model input is a live object.
- What the tests pin, and how to run the opt-in Hub checks, is listed in the README's Tests section.
