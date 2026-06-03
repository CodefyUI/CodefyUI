# Feature: Export tab architecture diagram (nodes + connections only)

Goal: help students see an AI algorithm's structure by exporting the active
tab as a clean diagram — **nodes + connecting lines only, no parameters**.

## Approach
Generate a standalone **SVG** directly from the live `nodes`/`edges` data
(not a screenshot of the canvas, which carries all the param/viz clutter the
request explicitly excludes). Pure function → fully unit-testable → vector
output ideal for slides/handouts.

## Changes
1. `frontend/src/utils/exportDiagram.ts` — `graphToSvg(nodes, edges, opts)`:
   - skip note nodes (annotations, not architecture)
   - each node → rounded rect (category-colored border + faint tint) with its
     label centered; **no params**
   - each edge → bezier path + arrowhead (data-flow direction), colored by the
     edge's data-type stroke (educational) with a neutral fallback
   - bounding box + padding → viewBox; light & dark themes
2. `frontend/src/utils/index.ts` — re-export.
3. `Toolbar.tsx` — `handleExportDiagram` + "Export as Diagram" menu item.
4. i18n: `toolbar.exportDiagram[.title|.empty]` in en + zh-TW.
5. Tests: `exportDiagram.test.ts` (full coverage) + Toolbar menu tests.

## Verify
- `pnpm test` from `frontend/` (100% coverage maintained)
- `pnpm build` (tsc) green
- Manually open a generated SVG to eyeball it
