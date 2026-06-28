# {{plugin_name}}

A [CodefyUI](https://github.com/treeleaves30760/CodefyUI) plugin, scaffolded with `cdui plugin new`.

## Develop

With the CodefyUI server running (`cdui start` or `cdui dev`):

```bash
# from the CodefyUI repo:
cdui plugin dev path/to/{{plugin_id}}     # link + watch + hot-reload on every edit
```

Replace `nodes/example_node.py` with your node(s). Every `.py` under `nodes/`
whose classes subclass `BaseNode` and define `NODE_NAME` is auto-registered and
appears in the editor palette.

## Frontend UI (optional)

If you scaffolded with `--ui`, an editor tool panel + a custom node renderer
live in `ui/` (React + TypeScript):

```bash
cd ui
pnpm install
pnpm build      # emits ../frontend/index.js (commit it)
pnpm dev        # rebuild on save — pair with `cdui plugin dev`
```

The typed SDK is vendored under `ui/src/sdk/` (clone-and-own). It mirrors the
host plugin API, so you get autocomplete for `defineTool`, the hooks, and
`defineNodeRenderer`.

## Test

```bash
uv run --directory path/to/CodefyUI/backend pytest path/to/{{plugin_id}}/tests/
```

## Publish

Push to GitHub, tag a release, and your users install with:

```bash
cdui plugin install <your-username>/<your-repo>
```
