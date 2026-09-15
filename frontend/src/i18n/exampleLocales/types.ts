/** Per-example translated copy, keyed by the `path` `/api/examples/list`
 * returns: `Model_Architecture/ConvNeXt-CNN` for a builtin, and
 * `plugin:<id>/<rest>` for one a plugin pack ships.
 *
 * Only the description is translated. An example's `name` stays in English in
 * every locale: the names are the canonical handles readers meet again in the
 * docs, in `examples/` on disk and in `run_graph.py` arguments, and a title
 * that changes with the language stops being a handle.
 */
export interface ExampleTranslation {
  description?: string;
}

export type ExampleTranslations = Record<string, ExampleTranslation>;
