# Stats — descriptive statistics and charts

Eight nodes that answer the ordinary questions about a dataset in one node
each, instead of hand-wiring `Mean` / `Reduce` / `Softmax` every time.

```bash
cdui plugin install stats
```

Every node is pure, cacheable and computes on CPU with numpy. The pack imports
**numpy and torch only** — no pandas, no scikit-learn at runtime — and passes
CodefyUI's AST security gate with **no `[security]` overrides**, which is the
point: the default plugin policy is workable for real statistics code.

## Node reference

Ports marked *(opt)* are optional. Every `table` / `matrix` output is a
`float32` 2D `TENSOR`; see [the table contract](#the-table-contract) for the
companion list ports. The two display nodes are the exceptions — `TableView`
returns a `STRING` and `ChartView` returns a spec `dict`.

### `Stats-Describe`

| | |
|---|---|
| **In** | `table` TENSOR · `columns` LIST *(opt)* |
| **Out** | `table` TENSOR `[statistics, columns]` · `columns` LIST · `row_labels` LIST |

`row_labels` are pandas' names: `count`, `mean`, `std`, `min`, the requested
percentiles, `max`.

| Param | Type | Default | Values |
|---|---|---|---|
| `axis` | select | **`columns`** | `columns`, `rows`, `all` |
| `percentiles` | string | `25,50,75` | comma-separated, 0–100 |

### `Stats-GroupByAggregate`

| | |
|---|---|
| **In** | `table` TENSOR · `columns` LIST *(opt)* · `keys` LIST *(opt)* |
| **Out** | `table` TENSOR `[groups, columns]` · `columns` LIST · `row_labels` LIST *(group keys)* · `counts` TENSOR `[groups]` |

`counts` is each group's **row** count. An `agg` of `count` instead gives the
number of *present* (non-NaN) values per column, so the two differ wherever
there are holes. Output column names carry the aggregate applied — `petal
length (cm) [mean]`.

| Param | Type | Default | Values |
|---|---|---|---|
| `group_by` | string | `""` *(use `keys`)* | comma-separated column names or indices |
| `agg` | select | `mean` | `mean`, `sum`, `count`, `min`, `max`, `std` |
| `agg_overrides` | string | `""` | `col=agg` pairs, comma-separated |

### `Stats-Histogram`

| | |
|---|---|
| **In** | `tensor` TENSOR |
| **Out** | `table` TENSOR `[bins, 3]` · `columns` LIST · `row_labels` LIST *(bin intervals)* · `chart` ANY *(`media=chart`)* · `dropped` SCALAR |

`columns` is `["bin_start", "bin_end", "count"]`, or `"density"` when `density`
is on. `dropped` counts the non-finite values excluded before binning.

| Param | Type | Default | Values |
|---|---|---|---|
| `bins` | int | `20` | 1–1000 |
| `range_mode` | select | `auto` | `auto`, `manual` |
| `range_min` | float | `0.0` | *(manual only)* |
| `range_max` | float | `1.0` | *(manual only)* |
| `density` | bool | `false` | |
| `title` | string | `Histogram` | |

### `Stats-Percentile`

| | |
|---|---|
| **In** | `tensor` TENSOR · `columns` LIST *(opt)* |
| **Out** | `percentiles` TENSOR · `columns` LIST · `row_labels` LIST *(`25%`, `50%`, …)* |

`percentiles` is 1D `[q]` for `axis = all`, else 2D `[q, series]`.

| Param | Type | Default | Values |
|---|---|---|---|
| `q` | string | `50` | comma-separated, 0–100 |
| `axis` | select | **`all`** | `all`, `columns`, `rows` |

> **The two `axis` params differ.** `Stats-Describe` defaults to `columns`
> (describe each column, as pandas does) and lists `columns, rows, all`.
> `Stats-Percentile` defaults to `all` (reduce everything to one number per q)
> and lists `all, columns, rows`. Same name, different default, different order
> in the dropdown.

### `Stats-Correlation`

| | |
|---|---|
| **In** | `table` TENSOR · `columns` LIST *(opt)* |
| **Out** | `matrix` TENSOR `[columns, columns]` · `columns` LIST · `row_labels` LIST *(same names)* · `chart` ANY *(`media=chart`)* |

| Param | Type | Default | Values |
|---|---|---|---|
| `method` | select | `pearson` | `pearson`, `spearman` |
| `drop_nan` | bool | `true` | pairwise-complete, as pandas' `corr` |
| `title` | string | `Correlation` | |

### `Stats-ConfusionMatrix`

| | |
|---|---|
| **In** | `predictions` ANY · `labels` ANY |
| **Out** | `matrix` TENSOR `[classes, classes]` · `columns` LIST *(predicted)* · `row_labels` LIST *(true)* · `accuracy` SCALAR · `chart` ANY *(`media=chart`)* |

Each input takes a list of labels, a 1D tensor of class indices, or a 2D
`[samples, classes]` score matrix (row argmax). `accuracy` is the diagonal's
share of the **raw counts**, so it does not move when `normalize` does.

| Param | Type | Default | Values |
|---|---|---|---|
| `normalize` | select | `none` | `none`, `true` *(row → recall)*, `pred` *(column → precision)*, `all` |
| `class_names` | string | `""` *(every class seen, sorted)* | comma-separated |
| `title` | string | `Confusion matrix` | |

### `Stats-TableView`

| | |
|---|---|
| **In** | `table` TENSOR · `columns` LIST *(opt)* · `row_labels` LIST *(opt)* |
| **Out** | `text` STRING — the rendered table, **not** a tensor |

Also emits the reserved `__log__` key, which the `output_kind` channel turns
into a `text` entry, so the table appears in the Results panel.

| Param | Type | Default | Values |
|---|---|---|---|
| `max_rows` | int | `20` | 0–1000; `0` means every row |
| `precision` | int | `4` | 0–12 |
| `title` | string | `""` | |

### `Stats-ChartView`

| | |
|---|---|
| **In** | `chart` ANY *(opt)* · `table` TENSOR *(opt)* · `columns` LIST *(opt)* · `row_labels` LIST *(opt)* |
| **Out** | `chart` ANY *(`media=chart`)* — a spec **dict**, not a tensor |

Connect either a `chart` payload from another Stats node or a `table`. With
`kind = auto` a table becomes a bar chart when `row_labels` are present (the
rows are categories) and a line chart otherwise.

| Param | Type | Default | Values |
|---|---|---|---|
| `kind` | select | `auto` | `auto`, `bar`, `line`, `scatter`, `heatmap` |
| `title` | string | `""` *(keep the payload's)* | |
| `columns_filter` | string | `""` *(all columns)* | comma-separated names or indices |

## Examples

`examples/Stats/` ships three graphs, reachable from the Examples gallery:

- **Iris describe()** — `CSVReader → Stats-Describe → Stats-TableView`.
- **Group by species, then chart it** — `CSVReader → Stats-GroupByAggregate → Stats-ChartView`.
- **Confusion matrix as a heatmap** — decision tree on held-out iris → `Stats-ConfusionMatrix`.

---

## The table contract

Third-party packs that want to interoperate with these nodes should read this
section. It documents what CodefyUI **actually** does, which is not quite what
the phrase "a table" suggests.

### There is no `TABLE` port type

`DataType` in `backend/app/core/node_base.py` is a closed `Enum`
(`TENSOR`, `LIST`, `SCALAR`, `STRING`, `ANY`, …). A plugin cannot add a member,
and there is no `TABLE`, `DATAFRAME` or `RECORD`. A table therefore travels as
**several ports that agree with each other**, not as one value:

| Port | Type | Meaning |
|---|---|---|
| `table` | `TENSOR` | 2D `[rows, columns]`, numeric |
| `columns` | `LIST` | one name per column, in column order |
| `row_labels` | `LIST` | one name per row, in row order (optional) |
| `keys` / `labels` | `LIST` | one categorical value per row (optional) |

**Who speaks it in core today:**

- **Produce** `tensor` + `labels` + `columns`: `CSVReader` and
  `SyntheticDataset`. Those are the only two.
- **Consume the column-name list:** `ColumnSelector` (`columns`, optional) —
  the only one. It emits no `columns` of its own, so the names stop there.
- **Consume the per-row categorical list:** `TrainTestSplit`, `Accuracy`, the
  classifier nodes (`y_train`), `ScatterPlot2D`, `EmbeddingScatter`.
- **`RowSelector`** takes a `labels` LIST, but those are **row** labels — the
  row-axis analogue of `row_labels` below, not column names. It is the closest
  core precedent for the row-name port this pack adds.
- `Edu-ColumnStats` takes a bare 2D `TENSOR` and no list port at all.

### Rules

1. **Values are numeric and 2D.** `float32` on output; any numeric dtype
   accepted on input. A 1D input is read as a single column. Non-numeric data
   never enters `table` — it rides a `LIST` port instead. That is why
   `CSVReader` sends the species column to `labels` rather than to `tensor`.
2. **`columns` is `list[str]`, positionally aligned with the columns.** Emit
   strings; when consuming, re-cast with `str()` — `CSVReader` forwards raw
   pandas labels, which are usually but not always strings.
3. **`columns` is optional and often absent.** `ColumnSelector` consumes the
   names and emits none, so a table arriving from mid-graph usually has no
   names at all. Generate `c0`, `c1`, … rather than failing. These nodes do,
   and they disambiguate duplicates as `a`, `a#1`, `a#2`.
4. **A short or over-long `columns` list is not an error.** Pad with generated
   names, trim the excess.
5. **NaN means missing; ±Inf is a value.** This is pandas' rule, and it is what
   makes `Stats-Describe` reproduce `DataFrame.describe()` exactly: `count`
   ignores NaN but counts Inf, so a column holding an Inf has an Inf mean.
   There is no separate mask port and no sentinel value.
6. **`row_labels` is this pack's addition.** No core node round-trips row
   names, which is why a `describe()` table would otherwise print without
   telling you which row is the median. Every node here that produces a table
   emits it; every node that consumes one treats it as optional.

### What a columnar dict is (and is not)

`backend/app/core/port_stats.py` accepts a columnar dict —
`{"age": [30, 40], "city": ["taipei", "tainan"]}` — and so the Node Detail
Modal's **Stats** tab can summarise one. That is an *inspection* shape, not a
transport contract:

- it is not a declared port type, so nothing type-checks it;
- on the WebSocket and the outputs REST API it degrades to a truncated
  `repr()` string, because neither serialiser has a branch for it;
- it is only recognised when **every** value is a `list`/`tuple` — including
  length-1 columns.

Do not build a pipeline on it. Use the split table above.

---

## The chart contract

`Stats-Histogram`, `Stats-Correlation`, `Stats-ConfusionMatrix` and
`Stats-ChartView` all declare `media=MEDIA_CHART` on their `chart` output:

```python
from app.core.node_base import MEDIA_CHART, DataType, PortDefinition

PortDefinition(name="chart", data_type=DataType.ANY, media=MEDIA_CHART)
```

The value is a plain dict — a **chart spec** — which rides the structured
`output_kind` channel to the browser as
`{"output_kind": "chart", "port": "chart", "chart": {...}}` and is drawn by
the editor's own SVG components. Nothing is rendered server-side, so the
picture is themed, hoverable and sharp at any zoom.

### Spec shape

```jsonc
{
  "kind": "bar" | "line" | "scatter" | "heatmap",   // required
  "title": "Mean petal length by species",
  "x_label": "species",
  "y_label": "cm",
  "note": "12 non-finite value(s) excluded",        // shown under the title

  // kind: "bar"
  "bars": [{ "label": "setosa", "value": 1.462 }],

  // kind: "line"
  "series": [{ "name": "loss", "points": [[0, 1.2], [1, 0.8]] }],

  // kind: "scatter"
  "points": [{ "x": 1.0, "y": 2.0, "label": "a", "cluster": 0 }],

  // kind: "heatmap"
  "matrix": [[13, 0], [1, 12]],
  "row_labels": ["setosa", "versicolor"],
  "col_labels": ["setosa", "versicolor"],
  "vmin": 0, "vmax": 13,
  "colormap": "viridis" | "blues" | "RdBu"
}
```

Those are the only keys the renderer reads. Anything else travels to the
browser untouched and is ignored — do not add a field expecting an effect.

### Rules

1. **Every number must be a finite, plain Python `float` or `int`.** Run
   events are serialised with `allow_nan=False`, so a NaN would be rewritten
   to `null` and drawn as a hole — and a `np.float32` is not JSON-serialisable
   at all (only `np.float64` subclasses Python `float`, so it slips past the
   run store's `json_safe`). Substitute a real value and say so in `note`.
2. **Keep the spec small — budget about 64 KB, not 128 KB.** When a
   `node_status` payload exceeds `RUN_EVENT_PAYLOAD_CAP_BYTES` (128 KB) the
   run service degrades it in three stages: first each output entry gets a
   share of the cap (`cap ÷ number of entries`) and any entry over its share
   becomes an `{"elided": true, ...}` marker keeping `output_kind` and `port`
   but **dropping the payload**; then long strings are truncated; only if it is
   still too big does the whole payload collapse to one marker. A chart node
   emits at least a `chart` entry and a `tensor_summary` sibling, so the
   realistic share for one chart is **half the cap**. This pack caps at 200
   bars, 8 series × 2000 points, 2000 scatter points and a 64×64 heatmap.
3. **Never assign `note` directly.** Use `add_note()` from `_stats_core`. A
   builder may already have recorded a downsampling notice, and overwriting it
   turns a disclosed truncation into a silent one — a 500-bin histogram of data
   containing NaN would report the excluded values and quietly lose "showing
   the first 200 of 500 bars".
4. **`vmin`/`vmax` are the colour scale, not the data range.** A correlation
   heatmap pins −1…1 so a weak matrix cannot masquerade as a strong one; a raw
   confusion matrix uses its own maximum.
5. **A spec the editor cannot draw degrades to a caption, never to an error.**
   Three distinct cases, and they say different things:
   - an **unknown `output_kind`** (a media kind this editor predates) is
     dropped by the WebSocket handler and never reaches the log at all;
   - an **unknown chart `kind`** renders "This chart kind (`x`) needs a newer
     editor" — the reader should upgrade;
   - a **known kind missing its payload** (a `heatmap` with no `matrix`)
     renders "This heatmap chart arrived without its data" — the reader should
     look at the node that produced it, not at their editor version.

### Adding your own media kind

`MEDIA_CHART` is not special-cased in core. `declared_media_ports()` keys on
whatever string a port declares, and any port whose value is a non-empty dict
is shipped through untouched. A pack declaring `media="waveform"` reaches the
browser as `{"output_kind": "waveform", ...}` with no core change; only
rendering it needs one.

Two constraints on the kind string. It must not collide with the keys an entry
uses for its own bookkeeping — `output_kind`, `port`, `elided`, `bytes`,
`cap_bytes` — because the payload is stored under a key *named by the kind*, so
a `media="elided"` port would forge the truncation marker. Colliding kinds are
refused with a logged warning rather than quietly renamed, so the mistake
surfaces to the pack author. And the payload must be a non-empty `dict`;
anything else is skipped, which is how a declared port that produced nothing
this run stays absent instead of arriving empty.
