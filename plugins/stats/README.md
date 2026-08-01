# Stats — descriptive statistics and charts

Eight nodes that answer the ordinary questions about a dataset in one node
each, instead of hand-wiring `Mean` / `Reduce` / `Softmax` every time.

```bash
cdui plugin install stats
```

| Node | In → Out | Params |
|---|---|---|
| `Stats-Describe` | table → table | `axis`, `percentiles` |
| `Stats-GroupByAggregate` | table, keys → table | `group_by`, `agg`, `agg_overrides` |
| `Stats-Histogram` | tensor → table + chart | `bins`, `range_mode`, `range_min`, `range_max`, `density`, `title` |
| `Stats-Percentile` | tensor → tensor | `q`, `axis` |
| `Stats-Correlation` | table → matrix + chart | `method`, `drop_nan`, `title` |
| `Stats-ConfusionMatrix` | predictions, labels → matrix + chart | `normalize`, `class_names`, `title` |
| `Stats-TableView` | table → display | `max_rows`, `precision`, `title` |
| `Stats-ChartView` | chart *or* table → display | `kind`, `title`, `columns_filter` |

Every node is pure and cacheable, computes on CPU with numpy, and returns
`float32` tensors. The pack imports **numpy and torch only** — no pandas, no
scikit-learn at runtime — and passes CodefyUI's AST security gate with **no
`[security]` overrides**, which is the point: the default plugin policy is
workable for real statistics code.

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

This is what `CSVReader` and `SyntheticDataset` already emit (`tensor` +
`labels` + `columns`) and what `ColumnSelector`, `RowSelector`, `FilterRows`
and `Edu-ColumnStats` already accept. This pack produces and consumes exactly
that shape, so `CSVReader → Stats-Describe` needs no adapter.

### Rules

1. **Values are numeric and 2D.** `float32` on output; any numeric dtype
   accepted on input. A 1D input is read as a single column. Non-numeric data
   never enters `table` — it rides a `LIST` port instead. That is why
   `CSVReader` sends the species column to `labels` rather than to `tensor`.
2. **`columns` is `list[str]`, positionally aligned with the columns.** Emit
   strings; when consuming, re-cast with `str()` — `CSVReader` forwards raw
   pandas labels, which are usually but not always strings.
3. **`columns` is optional and often absent.** Any transform node in core
   drops it (`ColumnSelector` has no `columns` output at all), so a table
   arriving from mid-graph usually has no names. Generate `c0`, `c1`, … rather
   than failing. These nodes do, and they disambiguate duplicates as `a`,
   `a#1`, `a#2`.
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
  "colormap": "viridis" | "blues" | "RdBu",
  "value_format": "0.00"
}
```

### Rules

1. **Every number must be a finite, plain Python `float` or `int`.** Run
   events are serialised with `allow_nan=False`, so a NaN would be rewritten
   to `null` and drawn as a hole — and a `np.float32` is not JSON-serialisable
   at all (only `np.float64` subclasses Python `float`, so it slips past the
   run store's `json_safe`). Substitute a real value and say so in `note`.
2. **Keep the spec small.** A `node_status` payload over
   `RUN_EVENT_PAYLOAD_CAP_BYTES` (128 KB) is replaced wholesale by an elision
   marker. This pack caps at 200 bars, 8 series × 2000 points, 2000 scatter
   points and a 64×64 heatmap, downsampling with a `note` rather than
   truncating silently.
3. **`vmin`/`vmax` are the colour scale, not the data range.** A correlation
   heatmap pins −1…1 so a weak matrix cannot masquerade as a strong one; a raw
   confusion matrix uses its own maximum.
4. **An unknown `kind` renders as nothing, not as an error.** The frontend
   ignores kinds it does not know, exactly as it ignores unknown
   `output_kind`s — so adding a fifth kind is a frontend change, and old
   editors keep working against a new backend.

### Adding your own media kind

`MEDIA_CHART` is not special-cased in core. `declared_media_ports()` keys on
whatever string a port declares, and any port whose value is a non-empty dict
is shipped through untouched. A pack declaring `media="waveform"` reaches the
browser as `{"output_kind": "waveform", ...}` with no core change; only
rendering it needs one.
