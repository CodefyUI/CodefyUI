const en = {
  // Toolbar
  'toolbar.run': 'Run',
  'toolbar.running': 'Running...',
  'toolbar.stop': 'Stop',
  'toolbar.run.title': 'Execute the pipeline (Run)',
  'toolbar.stop.title': 'Stop execution',
  'toolbar.reloadNodes': 'Reload Nodes',
  'toolbar.reloadNodes.title': 'Reload node definitions from backend',
  'toolbar.reload.fail': 'Reload failed: {error}',

  // Menu: File
  'toolbar.menu.file': 'File',
  'toolbar.save': 'Save',
  'toolbar.save.title': 'Save graph',
  'toolbar.save.prompt': 'Enter a name for this graph:',
  'toolbar.save.success': 'Graph "{name}" saved successfully.',
  'toolbar.save.fail': 'Save failed: {error}',
  'toolbar.save.overwriteConfirm': 'A graph named {name} already exists and will be overwritten - continue?',
  'toolbar.saveAs': 'Save As...',
  'toolbar.saveAs.title': 'Save under a new name',
  'toolbar.load': 'Load',
  'toolbar.load.title': 'Load a saved graph',
  'toolbar.load.fail': 'Load failed: {error}',
  'toolbar.load.loading': 'Loading...',
  'toolbar.load.empty': 'No saved graphs',
  'toolbar.load.toCanvas': 'Load into this canvas tab',
  'toolbar.load.toCanvas.title':
    'Replace what is on this canvas without binding the tab to the file — the next Save asks where to put it',
  'toolbar.load.toCanvas.confirm': 'Replace this canvas with "{name}"?',
  'toolbar.load.toCanvas.confirmAction': 'Replace',
  'toolbar.load.andSave': 'Load and save',
  'toolbar.load.andSave.title':
    'Load it and bind this tab to the file — Save then writes straight back over it',
  'toolbar.load.search': 'Search saved graphs…',
  'toolbar.load.noMatch': 'No graph matches "{query}"',
  'toolbar.import': 'Import JSON...',
  'toolbar.import.fail': 'Import failed: {error}',
  'toolbar.clear': 'Clear Canvas',
  'toolbar.clear.title': 'Clear the canvas',
  'toolbar.clear.confirm': 'Clear the canvas? All unsaved work will be lost.',

  // Menu: Export
  'toolbar.menu.export': 'Export',
  'toolbar.export.empty': 'Canvas is empty — add some nodes before exporting.',
  'toolbar.exportJson': 'Export as JSON',
  'toolbar.exportJson.title': 'Download graph as JSON file (includes subgraphs)',
  'toolbar.exportJson.empty': 'Canvas is empty — add some nodes before exporting.',
  'toolbar.export': 'Export as Subgraph',
  'toolbar.export.title': 'Export current graph as a reusable subgraph/preset',
  'toolbar.export.prompt': 'Enter a name for this subgraph:',
  'toolbar.export.success': 'Subgraph "{name}" exported successfully! It now appears in the Nodes panel.',
  'toolbar.export.fail': 'Export failed: {error}',
  // core#137: a preset carries only nodes + edges, so it cannot hold the
  // definition an instance node points at. Refusing names the blocks so the
  // user knows exactly which ones to expand first.
  'toolbar.export.subgraphRefused': 'Cannot export as a subgraph while the canvas contains collapsed blocks ({names}). A preset cannot carry their contents — expand them first, then export.',
  'toolbar.exportPython': 'Export as Python',
  'toolbar.exportPython.title': 'Download a headless Python runner (requires the CodefyUI backend environment)',
  'toolbar.exportPython.empty': 'Canvas has no executable nodes — add a node before exporting.',
  'toolbar.exportPython.fail': 'Python export failed: {error}',
  'toolbar.exportDiagram.svg': 'Export Diagram (SVG)',
  'toolbar.exportDiagram.png': 'Export Diagram (PNG)',
  'toolbar.exportDiagram.title': 'Download the architecture (nodes, input/output ports & connections — no parameter values) as an image',
  'toolbar.exportDiagram.empty': 'Canvas is empty — add some nodes before exporting.',
  'toolbar.exportDiagram.fail': 'Diagram export failed: {error}',

  // Status
  'status.idle': 'Idle',
  'status.running': 'Running',
  'status.completed': 'Completed',
  'status.error': 'Error',
  'status.skipped': 'Skipped',
  'status.cached': 'Cached',
  'status.interrupted': 'Interrupted',

  // Connection (WebSocket reconnect surface)
  'connection.lost': 'Connection lost — reconnecting…',
  'connection.restored': 'Connection restored',
  'connection.failed': 'Could not reconnect to the execution server',
  'connection.tooLarge':
    'Graph too large to send — the server refused the message and closed the '
    + 'connection. Simplify the graph, or raise the server’s WebSocket '
    + 'message limit (CODEFYUI_WS_MAX_MESSAGE_BYTES).',

  // Node Palette
  'palette.title': 'Nodes',
  'palette.search': 'Search nodes...',
  'palette.loading': 'Loading nodes...',
  'palette.loadFail': 'Failed to load nodes: {error}',
  'palette.retry': 'Retry',
  'palette.noMatch': 'No matching nodes',
  'palette.empty': 'No nodes available',
  'palette.hint': 'Drag nodes onto the canvas',
  'palette.searchPresets': 'Search presets...',
  'palette.presets.empty': 'No presets available',
  'palette.presets.noMatch': 'No matching presets',
  'palette.presets.hint': 'Drag presets onto the canvas',

  // Sidebar rail (#126)
  'sidebar.rail.aria': 'Sidebar sections',
  'sidebar.tab.nodes': 'Nodes',
  'sidebar.tab.presets': 'Presets',
  'sidebar.tab.templates': 'Templates',
  'sidebar.tab.custom': 'Custom & Plugins',
  'sidebar.tab.git': 'Source Control',
  'sidebar.collapse': 'Collapse sidebar',
  'sidebar.expand': 'Expand sidebar',
  'sidebar.resize': 'Resize sidebar',
  'sidebar.jumpTo': 'Jump to category',
  'sidebar.expandAll': 'Expand all',
  'sidebar.collapseAll': 'Collapse all',
  'sidebar.refresh': 'Refresh',

  // Sidebar: Templates tab (#126)
  'templates.search': 'Search examples...',
  'templates.loading': 'Loading examples...',
  'templates.loadFail': 'Failed to load examples: {error}',
  'templates.empty': 'No examples available',
  'templates.noMatch': 'No matching examples',
  'templates.hint': 'Drag an example onto the canvas, or click to add it',

  // Template gallery modal (core#128)
  'gallery.open': 'Templates',
  // Used where the word "Templates" is already on screen as a heading (the
  // sidebar tab, the empty-canvas overlay), so the two never read as the same
  // control.
  'gallery.browse': 'Browse all templates',
  'gallery.open.title': 'Browse every built-in and plugin example',
  'gallery.title': 'Template Gallery',
  'gallery.subtitle': 'Open one in a new tab, or insert it into this canvas',
  'gallery.search': 'Search templates...',
  'gallery.close': 'Close template gallery',
  'gallery.list': 'Template list',
  'gallery.detail': 'Template details',
  'gallery.detailEmpty': 'Select a template to see what it contains',
  'gallery.noDescription': 'This template ships no description.',
  'gallery.edgeCount': '{count} connections',
  'gallery.sourceBuiltin': 'Built-in example',
  'gallery.sourcePlugin': 'From plugin pack "{plugin}"',
  'gallery.openNewTab': 'Open in new tab',
  'gallery.insert': 'Insert into this canvas',
  'gallery.insertHint': 'Inserted nodes get fresh ids and are placed below your current graph, so nothing is overwritten. One undo removes them.',

  // Sidebar: Custom & Plugins tab (#126)
  'customTab.section.nodes': 'Custom Nodes',
  'customTab.section.plugins': 'Plugins',
  'customTab.manage': 'Manage...',
  'customTab.plugins.open': 'Plugin Center...',
  'customTab.nodes.empty': 'No custom nodes yet',
  // No hint under the empty plugins section: the header button one line above
  // it IS the Plugin Center, so a sentence naming that destination is the
  // same fact twice. The packs hint stays because it carries another one --
  // what a pack is.
  'customTab.plugins.empty': 'No plugins installed',
  'customTab.loadFail': 'Failed to load: {error}',

  // Config Panel
  'config.title': 'Node Config',
  'config.selectNode': 'Select a node to configure',
  'config.parameters': 'Parameters',
  'config.noParams': 'No configurable parameters',
  'config.advanced': 'Advanced',
  'config.ports': 'Ports',
  'config.inputs': 'Inputs',
  'config.outputs': 'Outputs',
  'config.optional': 'optional',
  'config.execution': 'Execution',
  'config.range': 'Range: {min} — {max}',

  // Node
  'node.opt': 'opt',
  'node.running': 'Running...',
  'node.completed': 'Completed',
  'node.cached': 'Cached',
  'node.skipped': 'Skipped',
  'node.error': 'Error: {error}',
  'node.bypassed': 'BYPASS',
  'node.bypassed.title': 'Bypassed: this node is skipped and passes its input straight through',
  'node.code.empty': '(no code yet)',
  'node.code.moreLines': '+{count} more lines',

  // Results Panel
  'results.title': 'Execution Log',
  'results.training': 'Training',
  'results.trainingConfig': 'Parameters',
  'results.trainingEmpty': 'No training data yet.',
  'results.clear': 'Clear',
  'results.empty': 'No log entries. Run the pipeline to see output.',

  // Preset
  'preset.badge': 'PRESET',
  'preset.configure': 'Configure Preset',
  'preset.nodeCount': '{count} nodes inside',
  'preset.nodesInside': 'nodes inside',
  'preset.apply': 'Apply',
  'preset.cancel': 'Cancel',
  'preset.generalGroup': 'General',

  // Empty Canvas
  'empty.title': 'Build your first deep learning model',
  'empty.subtitle': 'Pick an example to get started quickly',
  'empty.hint': 'or drag a node from the left palette',
  'empty.loading': 'Loading examples...',
  'empty.loadError': 'Failed to load example',
  'empty.section.quickstart': 'Quick Start',
  'empty.section.advanced': 'Advanced Examples',
  'empty.section.plugin': 'Plugin Examples',
  'empty.section.architecture': 'Model Architectures',

  // Context Menu
  'contextMenu.rename': 'Rename',
  'contextMenu.duplicate': 'Duplicate',
  'contextMenu.delete': 'Delete',
  'contextMenu.rename.prompt': 'Enter a new name for this node:',
  'contextMenu.addTextNote': 'Add Text Note',
  'contextMenu.addImageNote': 'Add Image Note',
  'contextMenu.bypass': 'Bypass',
  'contextMenu.unbypass': 'Remove Bypass',
  'contextMenu.collapseToSubgraph': 'Collapse to subgraph',
  'contextMenu.enterSubgraph': 'Enter subgraph',
  'contextMenu.expandSubgraph': 'Expand subgraph here',

  // Subgraphs (core#137)
  'subgraph.badge': 'Subgraph',
  'subgraph.nodeCount': '{count} nodes',
  'subgraph.missing': 'definition missing',
  // Last-resort name for a block with neither a name nor an id -- an
  // instance whose type is a bare `subgraph:`, which only a hand-edited or
  // plugin-produced file can produce. Naming it "()" would be worse.
  'subgraph.unnamed': 'an unnamed block',
  'subgraph.breadcrumb.root': 'Main',
  'subgraph.breadcrumb.back': 'Back',
  'subgraph.breadcrumb.jump': 'Go back to this level',
  'subgraph.breadcrumb.exitAll': 'Back to the main graph',
  'subgraph.rename.hint': 'Click to rename this subgraph',
  'subgraph.rename.label': 'Subgraph name',
  'subgraph.collapse.too-few':
    'Select at least two nodes to collapse them into a subgraph',
  'subgraph.collapse.contains-start':
    'A Start node cannot go inside a subgraph - it marks where the whole graph begins',
  'subgraph.collapse.contains-note':
    'Notes are annotations, not part of a subgraph - deselect them first',
  'subgraph.collapse.read-only': 'This graph is open read-only',
  'subgraph.collapse.namePrompt': 'Name this subgraph',
  'subgraph.collapse.notConvex':
    'These nodes sit between the ones you selected, so the block would feed back into itself: {nodes}. Add them to the selection.',
  // Overflow tail for the list above. Error toasts never auto-dismiss and the
  // list is not scrollable, so a graph with fifty blockers (or one node with
  // a very long label) would otherwise paint a wall of text over the canvas
  // with no way to get rid of it. The blockers are all added to the selection
  // anyway, so the message only has to name enough of them to be recognisable.
  'subgraph.collapse.andMore': 'and {count} more',
  'subgraph.detail.interface': 'Subgraph interface',
  'subgraph.detail.inputs': 'Inputs',
  'subgraph.detail.outputs': 'Outputs',
  'subgraph.detail.enter': 'Enter subgraph',
  'subgraph.detail.empty': 'This subgraph exposes no ports',

  // Notes
  'note.placeholder': 'Click to edit...',
  'note.imagePlaceholder': 'Click to upload image',
  'note.bind': 'Bind to Nearest Node',
  'note.unbind': 'Unbind Note',
  'note.changeColor': 'Change Color',
  'note.layoutWarning': 'Unbound notes were not repositioned by auto-layout.',
  'note.boundToNode': 'Bound to node',

  // Tabs
  'tabs.add': 'New tab',
  'tabs.closeRunning': 'This tab is still running. Close it anyway?',
  'tabs.close.confirmTitle': 'Close "{name}"?',
  'tabs.close.confirmMessage':
    'This tab has a graph in it ({count} nodes). Closing discards it, and there is no undo for a closed tab: anything you have not saved to a graph file is gone. Cancel and save it first if you want it back later.',
  'tabs.close.confirmButton': 'Close tab',

  // Tab bar chrome (#341)
  'tabBar.readOnly': 'Read-only',
  'tabBar.sourceTitle': 'Opened by {plugin}',

  // Subgraph Editor (SequentialModel)
  'layersEditor.title': 'Model Architecture Editor',
  'layersEditor.palette': 'Layers',
  'layersEditor.apply': 'Apply',
  'layersEditor.cancel': 'Cancel',
  'layersEditor.import': 'Import',
  'layersEditor.export': 'Export',
  'layersEditor.import.title': 'Import a saved model architecture',
  'layersEditor.export.title': 'Export current architecture as JSON',
  'layersEditor.empty': 'Drag layers from the left panel to build your model',
  'layersEditor.layerCount': '{count} layers',
  'layersEditor.params': 'Parameters',
  'layersEditor.noParams': 'No parameters',
  'layersEditor.deleteLayer': 'Delete',
  'layersEditor.hint': 'Double-click to edit architecture',
  'layersEditor.import.fail': 'Import failed: {error}',
  'layersEditor.import.selectModel': 'Select SequentialModel to Import',
  'layersEditor.import.noContent': 'No importable layers or SequentialModel nodes found in this file.',
  'layersEditor.searchLayers': 'Search layers...',
  'layersEditor.snapOn': 'Snap: ON',
  'layersEditor.snapOff': 'Snap: OFF',
  'layersEditor.snapTitle': 'Toggle grid snap',
  'layersEditor.autoLayout': 'Auto Layout',
  'layersEditor.autoLayoutTitle': 'Arrange nodes top-to-bottom by connection order',
  'layersEditor.category.io': 'I/O',
  'layersEditor.category.merge': 'Merge',
  'layersEditor.validation.cycle': 'Graph contains a cycle',
  'layersEditor.validation.noInput': 'Graph must have exactly one Input node',
  'layersEditor.validation.noOutput': 'Graph must have exactly one Output node',
  'layersEditor.port.add': '+ Add port',
  'layersEditor.port.remove': 'Remove',
  'layersEditor.port.namePlaceholder': 'port name',
  'layersEditor.port.duplicate': 'Duplicate port name',
  'layersEditor.port.list': 'Ports',
  'layersEditor.layerNode.moreParams': '+{count} more',

  // Tooltips
  'toolbar.tooltips.on': 'Tips ON',
  'toolbar.tooltips.off': 'Tips OFF',
  'toolbar.tooltips.title': 'Toggle node description tooltips',

  // Custom Node Manager
  'customNodes.title': 'Custom Node Manager',
  'customNodes.loading': 'Loading...',
  'customNodes.empty': 'No custom nodes. Upload a .py file to get started.',
  'customNodes.enabled': 'Enabled',
  'customNodes.disabled': 'Disabled',
  'customNodes.delete': 'Delete',
  'customNodes.delete.confirm': 'Delete "{name}"? This cannot be undone.',
  'customNodes.upload': 'Upload .py',
  'toolbar.customNodes': 'Custom Nodes',
  'toolbar.customNodes.title': 'Manage custom nodes',

  // ParamField (file picker for model / image params)
  'paramField.upload.model': 'Upload model file',
  'paramField.upload.image': 'Upload image file',
  'paramField.upload.data': 'Upload data file (CSV)',
  'paramField.download': 'Download selected file',
  'paramField.refresh': 'Refresh file list',
  'paramField.selectFile': '-- select file --',
  'paramField.uploadFailed': 'Upload failed',
  'paramField.downloadFailed': 'Download failed',
  'paramField.secretHint': 'Session only - cleared on save. Prefer the environment variable.',

  // ParamField (in-canvas Python, core#131)
  'paramField.code.checking': 'checking...',
  'paramField.code.ok': 'policy OK',
  'paramField.code.rejected': 'rejected',
  'paramField.code.unavailable': 'check unavailable',
  'paramField.code.atLine': 'Line {line}:',
  'paramField.code.noRun': 'No run(inputs, params) defined yet - this node will fail when the graph runs.',
  'paramField.code.allowed': 'Imports allowed: {modules}',

  // Grid Snap
  'toolbar.gridSnap.on': 'Snap ON',
  'toolbar.gridSnap.off': 'Snap OFF',
  'toolbar.gridSnap.title': 'Toggle grid snapping for node alignment',

  // Auto Layout
  'toolbar.autoLayout': 'Auto Layout',
  'toolbar.autoLayout.experiments': 'Layout Experiments',
  'toolbar.autoLayout.all': 'Layout All',
  'toolbar.autoLayout.selected': 'Layout Selected ({count})',

  // Execution errors
  'execution.error.noEntryPoints': 'No entry points defined. Drag a Start node from the palette and connect it to the node you want to start execution from.',

  // Beginner-facing rewrites of raw Python/PyTorch exceptions (see
  // utils/errorMessages.ts). Each one names what to change, not just what broke.
  'error.missingTensorInput': "This node expected a 'tensor' input but did not receive one. Check that every required input is connected.",
  'error.missingInput': "Missing required input '{key}'. Check that it is connected.",
  'error.linearShapeMismatch': 'Size mismatch: this layer received {got} features but is configured for {expected}. Set the layer\'s in_features to {got}, or change the previous layer so it outputs {expected}.',
  'error.invalidReshape': 'Cannot reshape to {shape}: the tensor has {size} elements, which does not divide evenly into that shape. Check the batch size and the dimensions of the previous layer.',
  'error.channelMismatch': 'Channel mismatch: this layer is configured for {expected} input channels but received {got}. Set its in_channels to {got}, or change what feeds it.',

  // Re-attach (#121): the run kept going while the tab was closed.
  'execution.reattached': 'Reconnected to a run that is still in progress',

  // Refused submit (#123). NOT a failure: nothing started, and the run this
  // tab is already following keeps going.
  'execution.rejected': 'This run was not started — the server is already busy with this tab\'s run. It is still going; wait for it to finish or press Stop.',

  // Node palette — control category / start node
  'palette.category.control': 'Control',
  'palette.start.description': 'Marks an execution entry point. Connect to the first node of a script.',

  // Keyboard Shortcuts
  'shortcuts.title': 'Keyboard Shortcuts',
  'shortcuts.undo': 'Undo',
  'shortcuts.redo': 'Redo',
  'shortcuts.redoAlt': 'Redo (alt)',
  'shortcuts.copy': 'Copy selected nodes',
  'shortcuts.paste': 'Paste nodes',
  'shortcuts.delete': 'Delete selected',
  'shortcuts.commit': 'Commit (Source Control message box)',
  'shortcuts.bypass': 'Bypass / un-bypass the selected node(s)',
  'shortcuts.toggleSidebar': 'Collapse / expand sidebar (when no node is selected)',
  'shortcuts.toggleSidebarAlways': 'Collapse / expand sidebar (always)',
  'shortcuts.quickSearch': 'Quick node search',
  'shortcuts.help': 'Show this help',
  'shortcuts.doubleClickKey': 'Double-click',

  // Training Summary
  'results.epoch': 'Epoch',
  'results.currentLoss': 'Loss',
  'results.bestLoss': 'Best',
  'results.lossCurve': 'Loss Curve',
  'results.waitingEpoch': 'Waiting for first epoch...',
  'results.epochsHeader': 'Epochs ({current}/{total})',
  'results.col.loss': 'Loss',
  'results.col.delta': 'Delta',
  'results.col.time': 'Time',

  // Beginner Mode
  'toolbar.beginnerMode.on': 'Beginner',
  'toolbar.beginnerMode.off': 'All Nodes',
  'toolbar.beginnerMode.title': 'Toggle beginner mode (show only basic node categories)',

  // Results Panel — expandable errors
  'results.clickToExpand': 'Click to expand error details',
  'results.clickToHighlight': 'Click to highlight node',

  // Language
  'lang.label': 'EN',

  // Teaching Inspector — Record toggle
  'toolbar.record.on': 'Rec ON',
  'toolbar.record.off': 'Rec OFF',
  'toolbar.record.title': 'Record node outputs (captured data is kept even when turned off)',

  // Teaching Inspector — Compare Segment
  'toolbar.compareSegment': 'Compare',
  'toolbar.clearSegment': 'Clear Segment',
  'toolbar.clearActiveSegment': 'Clear Active',
  'toolbar.compareSegment.title': 'Select two nodes, then click to compare head-input with tail-output. Multiple segments can coexist; the × on each bubble removes just that one.',
  'toolbar.compareSegment.needTwo': 'Select exactly two nodes first',
  'segment.noPath': 'Segment: no path from head to tail',

  // Inspector panel
  'inspector.title': 'Inspector',
  'inspector.collapse': 'Collapse inspector',
  'inspector.expand': 'Expand inspector',
  'inspector.collapsedStub': 'INSPECTOR',
  'inspector.segmentBadge': 'SEGMENT',
  'inspector.emptyPorts': 'This node has no ports.',
  'inspector.empty.notRun': 'Run the graph to capture data',
  'inspector.empty.notRunHint': 'Make sure Rec is ON, then click ▶ Run',
  'inspector.empty.noSelection': 'Select a node or segment to inspect',
  'inspector.empty.noSelectionHint': 'Click any node, or shift-select two and press Compare',
  'inspector.segment.inputs': 'Segment inputs ({count})',
  'inspector.segment.outputs': 'Segment outputs ({count})',
  'inspector.node.inputs': 'Inputs ({count})',
  'inspector.node.outputs': 'Outputs ({count})',
  'inspector.node.inputsEmpty': 'No inputs connected',
  'inspector.node.outputsEmpty': 'Run the graph to see outputs',
  'segment.removeThis': 'Remove this segment',

  // A1 — Verbose / step-trace mode
  'toolbar.verbose.on': 'Verbose',
  'toolbar.verbose.off': 'Quiet',
  'toolbar.verbose.title': 'Show step-by-step algorithm internals (Q, K, V, scores, …) in the Inspector',
  'inspector.tabs.forward': 'Forward',
  'inspector.tabs.steps': 'Steps',
  'inspector.tabs.backward': 'Backward',
  'inspector.steps.empty': 'This node does not record steps',
  'inspector.steps.requireVerbose': 'Enable Verbose mode and re-run to see steps',

  // A2 — Per-node weight persistence
  'toolbar.weights.on': 'Persist',
  'toolbar.weights.off': 'Fresh',
  'toolbar.weights.title': 'Keep layer weights between runs (so a Conv2d / Linear / Attention learns instead of resetting)',
  'toolbar.weights.resetAll': 'Reset All Weights',
  'toolbar.weights.resetAllConfirm': 'Reset all persisted weights for this graph?',
  'toolbar.weights.resetAllOk': 'Persisted weights cleared',
  'contextMenu.resetWeights': 'Reset Weights',
  'node.weightsPersistedBadge': 'Weights persisted',

  // A3 — Backward / gradient inspector
  'toolbar.backward.on': '∂ Grad ON',
  'toolbar.backward.off': '∂ Grad OFF',
  'toolbar.backward.title': 'Capture gradients on the next run (forward pass + .backward())',
  'toolbar.autoBackward.on': 'Auto Loss',
  'toolbar.autoBackward.off': 'Manual',
  'toolbar.autoBackward.title': 'Auto-synthesise a loss when no Loss / BackwardOnce node exists',
  'toolbar.backward.trainingLoopHint': 'TrainingLoop already runs backward. Insert a BackwardOnce node for ad-hoc inspection.',
  'inspector.backward.empty': 'No gradients captured',
  'inspector.backward.disabled': 'Enable Backward and re-run to inspect gradients',
  'inspector.backward.weightSection': 'Weight gradients',
  'inspector.backward.portSection': 'Output gradients',
  'inspector.backward.health.vanishing': 'vanishing',
  'inspector.backward.health.exploding': 'exploding',
  'inspector.backward.health.healthy': 'healthy',

  // Settings popover (consolidates Rec / Verbose / Persist / Backward / etc. into one panel)
  'toolbar.settings': 'Settings',
  'toolbar.settings.title': 'Open settings',
  'toolbar.settings.search': 'Search settings…',
  'toolbar.settings.section.execution': 'Execution',
  'toolbar.settings.section.recording': 'Recording & Inspection',
  'toolbar.settings.section.training': 'Training Behavior',
  'toolbar.settings.section.editor': 'Editor',
  'toolbar.settings.section.llm': 'LLM Providers',
  'toolbar.settings.section.system': 'This Server',
  'settings.device.name': 'Compute device',
  'settings.device.desc': 'Run the graph on this device. Nodes set to "auto" follow it.',

  // Font-size menu
  'toolbar.fontSize.title': 'Font size',
  'toolbar.fontSize.small': 'Small',
  'toolbar.fontSize.default': 'Default',
  'toolbar.fontSize.large': 'Large',

  // Settings rows
  'settings.record.name': 'Record node outputs',
  'settings.record.desc': "Capture every node's output on each run so the Inspector can show input → output diffs.",
  'settings.verbose.name': 'Verbose internals',
  'settings.verbose.desc': 'Inspector also shows Q / K / V / attention scores and other algorithm internals (teaching mode).',
  'settings.compare.name': 'Compare segment',
  'settings.compare.desc': 'Select two nodes on the canvas, then click to compare the head-input with the tail-output.',
  'settings.compare.actionCreate': 'Create segment',
  'settings.compare.actionClear': 'Clear active',
  'settings.compare.actionDisabled': 'Select two nodes',
  'settings.persist.name': 'Persist weights between runs',
  'settings.persist.desc': 'When off, every run re-initialises Conv2d / Linear / Attention weights — the model never learns.',
  'settings.resetWeights.name': 'Reset all weights now',
  'settings.resetWeights.desc': 'Drop every cached weight; the next Run starts from fresh initialisation.',
  'settings.resetWeights.action': 'Reset',
  'settings.gradients.name': 'Capture gradients',
  'settings.gradients.desc': 'Run forward + .backward() and store each layer\'s gradient for the Inspector.',
  'settings.autoLoss.name': 'Auto-synthesize loss',
  'settings.autoLoss.desc': 'When the graph has no Loss / BackwardOnce node, synthesize one so .backward() can run.',
  'settings.seed.name': 'Random seed',
  'settings.seed.desc': 'Seed every node from one number so the run is reproducible. Seeded runs execute one node at a time. Blank = unseeded.',
  'settings.seed.placeholder': 'none',
  'settings.deterministic.name': 'Deterministic algorithms',
  'settings.deterministic.desc': 'Ask PyTorch for reproducible kernels. Operations with no deterministic implementation warn instead of failing the run.',
  'settings.gridSnap.name': 'Grid snap',
  'settings.gridSnap.desc': 'Snap dragged nodes to the canvas grid.',
  'settings.tooltips.name': 'Show node tooltips',
  'settings.tooltips.desc': 'Reveal the description card when hovering nodes on the canvas.',
  'settings.nodeMode.name': 'Node category mode',
  'settings.nodeMode.desc': 'Basic shows only the essential categories in the sidebar; All shows every category.',
  'settings.nodeMode.basic': 'Basic',
  'settings.nodeMode.all': 'All',
  'settings.edgeStyle.name': 'Connection style',
  'settings.edgeStyle.desc': 'How value connections are drawn: circuit-board traces or smooth curves.',
  'settings.edgeStyle.circuit': 'Circuit',
  'settings.edgeStyle.curve': 'Curve',
  'settings.codex.name': 'ChatGPT Codex account',
  'settings.codex.descLoggedOut': 'Sign in to use the Codex provider in LLMChat. This uses your ChatGPT account session.',
  'settings.codex.descPending': 'Sign-in is in progress. Complete it in the browser tab, then return here.',
  'settings.codex.descLoggedIn': 'Signed in as {email}. Codex nodes can use your ChatGPT session.',
  'settings.codex.actionSignIn': 'Sign in',
  'settings.codex.actionSignOut': 'Sign out',
  'settings.codex.actionRefresh': 'Refresh',
  'settings.codex.signInOpened': 'Opened ChatGPT sign-in in a new tab.',
  'settings.codex.signInFailed': 'Codex sign-in failed',
  'settings.codex.logoutFailed': 'Codex sign-out failed',
  'settings.codex.statusFailed': 'Codex status check failed',

  // "This Server" section (#193 item 2). /api/health has reported all of this
  // since #135; until now nothing in the editor showed it.
  'settings.health.name': 'What this server has loaded',
  'settings.health.desc': 'The version you are running, and how much memory its caches are holding right now.',
  'settings.health.refresh': 'Refresh',
  // The Codex row's button is also called "Refresh", and the two are only
  // distinguishable by which section they sit in — which a screen reader does
  // not read out. The accessible name says which one this is.
  'settings.health.refreshAria': 'Refresh server status',
  'settings.health.loading': 'Reading the server…',
  'settings.health.failed': 'Could not read the server status. Press Refresh to try again.',
  'settings.health.version': 'Version',
  'settings.health.nodes': 'Nodes',
  'settings.health.presets': 'Presets',
  'settings.health.unknown': 'unknown',
  'settings.health.caches': 'Caches',
  'settings.health.cachesEmpty': 'No caches are running yet.',
  'settings.health.cachesHint': 'These hold results the server already computed so a re-run can skip the work; none of it is your saved graphs or files. Clearing one costs recompute time — for the weight cache that means training time, unless you saved a checkpoint.',
  'settings.health.cache.execution_cache': 'Node outputs (per editor connection)',
  'settings.health.cache.run_output_store': 'Recorded run outputs',
  'settings.health.cache.node_state_store': 'Layer weights kept between runs',
  'settings.health.cacheOf': '{used} of {budget}',
  // LLM
  'tokenizer.tokenCount': '{count} tokens',
  'tokenizer.emptyOutput': 'No tokens — input text was empty.',
  'tokenizer.runHint': 'Run the graph to see tokens',
  'tokenizer.truncatedInline': 'showing first {shown} of {total} — see Inspector for full list',
  'scatter.runHint': 'Run the graph to see the projection',
  'scatter.tooLargeInline': 'Too many points to preview on the node',
  'scatter.openDetail': 'Open detailed view',
  'scatter.points': '{count} points',
  'scatter.nearestToCenter': 'Nearest to view centre',
  'scatter.searchPlaceholder': 'Filter labels…',
  'scatter.noMatches': 'No labels match',
  'scatter.showAll': 'Show all',
  'scatter.hiddenCount': '{count} hidden',
  'scatter.zoomIn': 'Zoom in',
  'scatter.zoomOut': 'Zoom out',
  'scatter.resetView': 'Reset view',
  'scatter.hidePoint': 'Hide',
  'scatter.showPoint': 'Show',
  'scatter.loading': 'Loading points…',
  'scatter.loadError': "Couldn't load points: {error}",
  'scatter.unavailable': 'Cannot load: this run is no longer available.',
  'scatter.noData': 'No points to display',
  'scatter.closeHint': 'click outside or press Esc to close',
  'scatter.recenterHint': 'click a label to centre · drag to pan · scroll to zoom',
  'scatter.close': 'Close',
  'attention.runHint': 'Run the graph to see attention weights',
  'attention.heads': '{count} heads',
  'attention.causalMasked': 'striped cells = causally masked',
  'attention.maskRunHint': 'Run the graph to see the mask',
  'attention.tooLargeInline': 'Tensor too large for inline preview',
  'attention.viewFull': 'View full',
  'textInput.placeholder': 'Type text here…',
  'textInput.charCount': '{count} chars',

  // Misc strings extracted to translate UI surfaces that previously had
  // hard-coded English (results panel collapse, empty-canvas card footer,
  // download failures, Start node label, toolbar aria, persistence quota).
  'results.expand': 'Expand panel',
  'results.collapse': 'Collapse panel',
  'empty.nodeCount': '{count} nodes',
  'node.start.label': 'Start',
  'download.failed': 'Download failed',
  'toolbar.layoutMode.aria': 'Layout mode',
  'toolbar.language.aria': 'Language',
  'persistence.quotaError': 'Could not save tabs — browser storage is full.',
  'persistence.storageUnavailable':
    'Browser storage is not working — the tabs shown may be out of date, and new changes may not be saved. Export anything you need before closing this tab.',
  // #164: the write-side counterpart to storageUnavailable above. A save
  // still succeeded here (on the smaller fallback tier), so this is a
  // warning about reduced headroom going forward, not a data-loss notice.
  'persistence.downgraded':
    'Browser storage dropped to a smaller fallback — large graphs may stop saving.',

  // Shared Confirm/Prompt dialog (#160): generic fallback button labels for
  // whichever call site does not override cancelText/confirmText. Every
  // real dialog in the app used to fall through to a hardcoded English
  // literal here regardless of locale.
  'dialog.cancel': 'Cancel',
  'dialog.ok': 'OK',
  'dialog.confirm': 'Confirm',

  // Per-project tab scoping (ID10): header badge + cross-project save refusal.
  'project.badge.title': 'Active project directory',
  'project.save.crossProjectRefused': 'This graph belongs to another project ({origin}) and cannot be saved into the open project.',

  // format_version read policy (ID8): newer-than-this-build graphs open
  // read-only, never blocked on load. The third string is the one case where
  // read-only is not an available answer (#200 item 10): MERGING a too-new
  // template into an editable graph has no document to mark read-only, so
  // that path refuses the merge and explains the two ways forward.
  'project.readOnly.loadNotice': 'Opened read-only: this graph uses a newer format (v{version}) than this CodefyUI build.',
  'project.readOnly.saveBlocked': 'Save is disabled: this graph was written by a newer CodefyUI. Update CodefyUI to edit it.',
  'project.formatTooNew.insertRefused': 'Nothing was inserted: this template uses a newer format (v{version}) than this CodefyUI build, so merging it into your graph could quietly drop parts of it. Open it instead to view it read-only, or update CodefyUI to use it.',

  // Plugin API v3 chrome (#132). Plugin panels and toolbar buttons supply
  // their own titles, so the host only labels the containers around them.
  'plugins.panels': 'Plugin panels',
  'plugins.moreActions': 'More plugin actions',

  // Runs panel (#124) — the Run Service made visible: every run the server
  // owns, its queue position, its live curves and its artifacts.
  'runs.tab': 'Runs',
  'runs.empty': 'No runs yet. Runs appear here as soon as one is submitted.',
  'runs.emptyFiltered': 'No runs with this status.',
  'runs.loading': 'Loading runs…',
  'runs.refresh': 'Refresh',
  'runs.showing': '{shown} of {total}',
  'runs.filter.all': 'All',
  'runs.col.name': 'Run',
  'runs.col.status': 'Status',
  'runs.col.device': 'Device',
  'runs.col.started': 'Started',
  'runs.col.duration': 'Duration',
  'runs.col.loss': 'Final loss',
  'runs.unnamed': '(unnamed)',
  'runs.queuePosition': 'Queue #{position}',
  'runs.status.queued': 'Queued',
  'runs.status.running': 'Running',
  'runs.status.succeeded': 'Succeeded',
  'runs.status.failed': 'Failed',
  'runs.status.cancelled': 'Cancelled',
  'runs.status.interrupted': 'Interrupted',
  'runs.action.cancel': 'Stop',
  'runs.action.reattach': 'Watch',
  'runs.action.delete': 'Delete',
  'runs.action.csv': 'CSV',
  'runs.action.cancelTitle': 'Ask this run to stop',
  'runs.action.reattachTitle': 'Stream this run into the Execution Log of the active tab',
  'runs.action.deleteTitle': 'Delete this run and its metrics, events and artifacts',
  'runs.action.csvTitle': 'Download the metrics of this run as CSV',
  'runs.delete.title': 'Delete run?',
  'runs.delete.message': 'The metrics, event log and artifact records for "{name}" are removed. Checkpoint files on disk are kept.',
  'runs.delete.confirm': 'Delete',
  'runs.reattach.title': 'Switch this tab to another run?',
  'runs.reattach.message': 'This tab is following run {current}. It will stop showing events from that run — the run itself keeps going.',
  'runs.reattach.confirm': 'Switch',
  'runs.reattach.offline': 'Cannot reach the execution server.',
  'runs.detail.metrics': 'Metrics',
  'runs.detail.downloadCsv': 'Download CSV',
  'runs.detail.noMetrics': 'No metrics recorded yet.',
  'runs.detail.log': 'Event log',
  'runs.detail.noLog': 'No events yet.',
  'runs.detail.artifacts': 'Artifacts',
  'runs.detail.noArtifacts': 'No artifacts recorded.',
  'runs.detail.copyPath': 'Copy path',
  'runs.detail.copied': 'Path copied to clipboard',
  'runs.detail.copyFailed': 'Could not copy the path',
  'runs.detail.error': 'Error',
  'runs.detail.seed': 'Seed',
  'runs.detail.deterministic': 'Deterministic',
  'runs.detail.close': 'Close detail',
  'runs.nodeStatus.running': 'running',
  'runs.nodeStatus.completed': 'completed',
  'runs.nodeStatus.cached': 'cached',
  'runs.nodeStatus.skipped': 'skipped',
  'runs.nodeStatus.error': 'failed',
  'runs.detail.step': 'step',
  'runs.log.started': 'Run started',
  'runs.log.node': 'Node {node} {status}',
  'runs.log.completed': 'Run completed',
  'runs.log.failed': 'Run failed: {error}',
  'runs.log.stopped': 'Run stopped ({reason})',
  'runs.log.warning': 'Warning: {detail}',
  'runs.error.gone': 'This run no longer exists on the server.',
  'runs.toast.inProgress': '{count} run(s) still in progress — open the Runs panel',
  'runs.toast.cancelling': 'Stop requested.',
  'runs.toast.alreadyDone': 'That run had already finished.',
  'runs.toast.cancelFailed': 'Could not stop the run',
  'runs.toast.deleted': 'Run deleted.',
  'runs.toast.deleteFailed': 'Could not delete the run',
  'runs.toast.exportFailed': 'Could not export metrics',

  // Node Detail Modal (#127)
  'contextMenu.openDetails': 'Open details',
  'nodeDetail.title': 'Node details',
  'nodeDetail.close': 'Close node details',
  'nodeDetail.prev': 'Previous node',
  'nodeDetail.next': 'Next node',
  'nodeDetail.position': '{index} / {total}',
  'nodeDetail.rename': 'Node name',
  'nodeDetail.renameHint': 'Enter to apply, Esc to cancel',
  'nodeDetail.parameters': 'Parameters',
  'nodeDetail.noParams': 'This node has no configurable parameters',
  'nodeDetail.tabs.code': 'Code',
  'nodeDetail.tabs.inputs': 'Inputs',
  'nodeDetail.tabs.outputs': 'Outputs',
  'nodeDetail.tabs.steps': 'Steps',
  'nodeDetail.tabs.backward': 'Backward',
  'nodeDetail.tabs.stats': 'Stats',
  'nodeDetail.tabs.docs': 'Docs',
  'nodeDetail.tabs.subgraph': 'Subgraph',
  'nodeDetail.code.title': 'Script',
  'nodeDetail.code.contract': 'def run(inputs, params) -> dict — inputs holds one key per connected port (in1, in2, ...); return {"out1": value}. A bare value becomes out1.',
  'nodeDetail.code.security': 'A guardrail, not a sandbox. The policy bounds which libraries a script can reach, not what those libraries can do, and the code runs in the CodefyUI process with your permissions. This check is the fast first pass: a script it accepts can still be refused while it runs. Only run scripts you trust.',
  'nodeDetail.code.inputs': 'Input ports',
  'nodeDetail.code.outputs': 'Output ports',
  'nodeDetail.code.inputCount': 'Number of input ports',
  'nodeDetail.code.outputCount': 'Number of output ports',
  'nodeDetail.code.unavailable': 'This node has no code parameter',
  'nodeDetail.tabError': 'This tab failed to render',
  'nodeDetail.inputs.title': 'Inputs ({count})',
  'nodeDetail.outputs.title': 'Outputs ({count})',
  'nodeDetail.inputs.empty': 'No inputs connected',
  'nodeDetail.outputs.empty': 'Run the graph to see outputs',
  'nodeDetail.captures.notRun': 'No captured data yet',
  'nodeDetail.captures.notRunHint': 'Run the graph with Rec on to capture this node’s values',
  'nodeDetail.captures.recordingOff': 'Record outputs is off — re-run with Rec on to capture values',
  'nodeDetail.captures.summaryTitle': 'Shapes from the last run',
  'nodeDetail.captures.noSummary': 'Nothing recorded for this node yet',
  'nodeDetail.stats.title': 'Node statistics',
  'nodeDetail.stats.loading': 'Computing…',
  'nodeDetail.stats.shape': 'Shape',
  'nodeDetail.stats.dtype': 'Dtype',
  'nodeDetail.stats.device': 'Device',
  'nodeDetail.stats.count': 'Count',
  'nodeDetail.stats.mean': 'Mean',
  'nodeDetail.stats.std': 'Std',
  'nodeDetail.stats.min': 'Min',
  'nodeDetail.stats.max': 'Max',
  'nodeDetail.stats.nan': 'NaN',
  'nodeDetail.stats.inf': 'Inf',
  'nodeDetail.stats.zeros': 'Zeros',
  'nodeDetail.stats.quantiles': 'Quantiles',
  'nodeDetail.stats.distribution': 'Distribution',
  'nodeDetail.stats.classBalance': 'Class balance',
  'nodeDetail.stats.sampled': 'sampled {size}',
  'nodeDetail.stats.sampledNote':
    'Mean, std, quantiles and the histogram come from a sample of this size. Count, min, max, NaN/Inf counts and class balance are exact.',
  'nodeDetail.stats.rows': 'Rows',
  'nodeDetail.stats.columns': 'Columns',
  'nodeDetail.stats.column': 'Column',
  'nodeDetail.stats.missing': 'Missing',
  'nodeDetail.stats.unique': 'Unique',
  'nodeDetail.stats.top': 'Most common',
  'nodeDetail.stats.columnsTruncated': 'Showing {shown} of {total} columns',
  'nodeDetail.stats.unsupported': 'No statistics for this value type ({type})',
  'nodeDetail.stats.notCaptured': 'Nothing captured for this port — re-run with Rec on',
  'nodeDetail.docs.description': 'Description',
  'nodeDetail.docs.noDescription': 'This node ships no description.',
  'nodeDetail.docs.params': 'Parameters',
  'nodeDetail.docs.noParams': 'This node has no parameters.',
  'nodeDetail.docs.inputs': 'Input ports',
  'nodeDetail.docs.outputs': 'Output ports',
  'nodeDetail.docs.noPorts': 'This node has no ports.',
  'nodeDetail.docs.default': 'default',
  'nodeDetail.docs.options': 'options',
  'nodeDetail.docs.range': 'range',

  // Edge data tooltip
  'edge.viewStats': 'View stats',

  // Shared plots
  'plot.noDistribution': 'no distribution',
  'plot.peak': 'peak {count}',

  // Chart output entries (#130)
  'chart.bar': 'Bar chart',
  'chart.series': 'series',
  'chart.unknownKind': 'This chart kind ({kind}) needs a newer editor',
  'chart.malformed': 'This {kind} chart arrived without its data',
  'nodeDetail.captures.charts': 'Charts ({count})',

  // Package Center — the keys the pack store toasts with (the panel's own
  // strings land with the panel).
  'packs.item.remove': 'Remove',
  // The accessible name, because a pack card carries one Remove button per
  // downloaded model and "Remove" three times over is one control repeated to
  // anyone navigating by name. The visible label stays the short word.
  'packs.item.removeNamed': 'Remove {item}',
  'packs.item.removeConfirm':
    'Remove "{item}"? It will be downloaded again the next time it is installed.',
  'packs.item.removed': 'Removed {item}.',
  'packs.item.removeFailed': 'Could not remove {item}',
  'packs.item.removeError': 'Could not remove {item}: {message}',
  'packs.restart.done': 'Server restarted. {pack} is ready.',
  'packs.restart.failed': 'The server restarted, but installing {pack} failed: {message}',
  // A second toast, and only when the record carried no message at all: the
  // helper that ran the install died with the old server, so its last output
  // is all that is left to go on.
  'packs.restart.failedLog': 'Last output from the installer: {log}',
  'packs.toast.installed': '{pack} installed.',
  'packs.toast.installFailed': 'Install failed: {message}',
  'packs.toast.cancelled': 'Install cancelled.',
  'packs.toast.cancelFailed': 'Could not cancel the install: {message}',
  'packs.toast.busy': 'Another install is already running.',
  'packs.toast.needsCli':
    'This pack cannot be installed from inside the app yet. Run: {command}',
  // What `needsCli` would have said on the one server it would be lying to:
  // a LIVE install the constraints file stopped, on a server that CAN
  // restart itself. The app can finish this one — the activity banner in the
  // Package Center carries the button — so the toast says where to go rather
  // than what to type, and brings the click with it.
  'packs.toast.restartRetry':
    'The install stopped at a package the server has loaded. Open the Package Center to restart the server and finish it.',
  'packs.toast.blocked': 'Install {pack} first.',
  // The two refusals a server that CAN restart still makes. Both are 409s
  // that carry a command, and `needsCli` — "cannot be installed from inside
  // the app" — would be a lie about either: the button works, just not this
  // second. So each says what is in the way and what ends it.
  'packs.toast.restartRefusedRunning': 'A graph is running. Stop it, then install.',
  'packs.toast.restartRefusedPending':
    'A restart is already pending. Wait for the server to come back.',
  'packs.toast.devRestart':
    'This pack needs a server restart, which cdui dev cannot do by itself. Use the command shown in the Package Center.',
  'packs.toast.inProgress': 'A pack is still installing. Open the Package Center to watch it.',
  'packs.toast.openCenter': 'Open Package Center',

  // Package Center — the panel itself: its chrome, the catalog copy, the
  // activity pane and the restart overlay. (The toast keys the pack store
  // fires are the block directly above.)
  'packs.title': 'Package Center',
  'packs.subtitle':
    'Install optional models and libraries so LLM nodes can use real implementations',
  'packs.close': 'Close Package Center',
  'packs.refresh': 'Refresh pack status',
  'packs.list': 'Pack list',
  'packs.activity': 'Install activity',
  'packs.loading': 'Loading packs...',
  'packs.loadFail': 'Failed to load packs: {error}',
  'packs.unsupported':
    'This server does not support the Package Center. Update CodefyUI and restart it.',
  'packs.empty': 'No optional packs are available',

  // Catalog copy, keyed by pack id. A pack this build has no string for is
  // not a bug: the panel falls back to the title and description the server
  // sent, so a newer backend still renders.
  'packs.catalog.sentence-embeddings.title': 'Sentence embeddings',
  'packs.catalog.sentence-embeddings.desc':
    'sentence-transformers plus four small embedding models (English, multilingual, Chinese) for TextEmbedding and WordVector',
  'packs.catalog.word-vectors.title': 'Word vectors (GloVe)',
  'packs.catalog.word-vectors.desc':
    'Real 400k-word GloVe-50d table for WordVector; no Python packages needed',
  'packs.catalog.rag.title': 'RAG stack',
  // Neither of these names what the row under it already says: the dependency
  // line carries "Requires: Sentence embeddings" with its live state, and the
  // button two rows below the GPU description reads "Install and restart".
  'packs.catalog.rag.desc':
    'Local generator model Qwen2.5-0.5B-Instruct for HFTextGenerate',
  'packs.catalog.gpu-torch.title': 'GPU PyTorch',
  'packs.catalog.gpu-torch.desc':
    'Switch PyTorch to the CUDA/ROCm build that matches this machine',

  // Pack and item state, keyed by the value the API sends.
  'packs.status.not_installed': 'Not installed',
  'packs.status.partial': 'Partly installed',
  'packs.status.installed': 'Installed',
  'packs.status.installing': 'Installing',
  'packs.status.needs_restart': 'Restart needed',
  'packs.status.failed': 'Failed',
  'packs.item.missing': 'Not downloaded',
  'packs.item.present': 'Downloaded',
  'packs.item.downloading': 'Downloading',
  'packs.item.license': 'License: {license}',

  // What a pack costs, and the buttons that spend it.
  'packs.pip': 'Python packages: {specs}',
  // Said next to the specs ONLY when it is the one thing left to do: every
  // model file is on disk and the libraries are not, which is the only reason
  // Install is alive on a card with nothing ticked. In every other state the
  // pack's own status pill already carries it.
  'packs.pipMissing': 'Python packages not installed',
  'packs.size': 'Download size: {size}',
  'packs.sizeSelected': '{size} selected',
  // The label only. The dependency's NAME is the link beside it, so the card
  // does not print it once as prose and again as a button.
  'packs.dependsOnLabel': 'Requires:',
  'packs.dependsOnMissing': 'Install {pack} first',
  // Why the Install button is dead, on its tooltip. A sentence rather than the
  // other button's label: a disabled control has to say what to DO next.
  'packs.selectSomething': 'Tick at least one item to install',
  'packs.installSelected': 'Install selected',
  'packs.cancel': 'Cancel install',
  'packs.cancelling': 'Cancelling...',
  'packs.remoteDisabled': 'Installing is only allowed from the computer that runs the server.',

  // Activity pane. `packs.activity.step.*` is keyed by the step id the job
  // sends, so an unknown step falls back to the server's own English label.
  'packs.activity.idle': 'Nothing is installing right now.',
  // Only the half the screen does not already show. "Pick a pack on the left"
  // described what the reader was looking at.
  'packs.activity.idleHint': 'Downloads keep going if you close this window.',
  'packs.activity.job': 'Installing {pack}',
  'packs.activity.step': 'Step {index}: {label}',
  'packs.activity.step.pip': 'Installing Python packages',
  'packs.activity.step.download': 'Downloading {item}',
  'packs.activity.step.convert': 'Preparing {item}',
  'packs.activity.step.verify': 'Verifying the installation',
  'packs.activity.overall': 'Overall progress',
  'packs.activity.progressAria': 'Install progress',
  'packs.activity.log': 'Install log',
  'packs.activity.logEmpty': 'Waiting for the first message...',
  'packs.activity.done': 'Installed {pack}.',
  'packs.activity.failed': 'Install failed: {message}',
  'packs.activity.cancelled': 'Install cancelled.',
  'packs.activity.needsRestart':
    'Installed. The server has to restart before {pack} can be used.',
  // The same status, the opposite story. A LIVE install that hit a resolver
  // conflict stopped BEFORE it changed anything, so "Installed." would be
  // false — and the restart is what would let it start, not what would let
  // it be used.
  'packs.activity.needsRestartConflict':
    'The install stopped: it would replace a package the server has loaded. Restart the server to finish it.',
  'packs.activity.lost': 'Lost contact with the server. Refresh to check the pack status.',
  'packs.activity.dismiss': 'Dismiss',
  // Offered only when the server said it can restart itself AND the job that
  // stopped said a restart is what would finish it.
  'packs.activity.restartAndInstall': 'Restart the server and install',
  // The confirm body, and the one thing that would otherwise surprise the
  // user: the helper that runs during the restart installs PACKAGES. It runs
  // from an interpreter with none of this app's downloader in it, so a pack's
  // models are still missing when the server comes back, and a second,
  // ordinary install is what fetches them.
  'packs.activity.restartAndInstallNote':
    'The server restarts to install the Python packages; download the models afterwards with a normal install.',

  // GPU PyTorch pack — the one install that swaps the wheel under the running
  // interpreter, so the user may have to run a command themselves.
  'packs.gpu.detected': 'Detected GPU: {label}',
  'packs.gpu.none': 'No GPU detected. The CPU build of PyTorch is already installed.',
  'packs.gpu.installed': 'Installed build: {variant}',
  'packs.gpu.recommended': 'Recommended build: {variant}',
  'packs.gpu.variant': 'PyTorch build',
  // The one condition under which the button declines — and only that. That
  // the server restarts is already the button's own label and the confirm
  // dialog's question; this is the half neither of them says.
  'packs.gpu.restartNote': 'It will not start while a graph is running.',
  'packs.gpu.restartConfirm': 'Install {variant} and restart the server?',
  'packs.gpu.installRestart': 'Install and restart',
  'packs.gpu.devMode':
    'You started CodefyUI with cdui dev, so the server cannot restart itself. Run this in the backend terminal, then start it again:',
  'packs.gpu.notYet':
    'Switching the PyTorch build from inside the app is not available yet. Run this in a terminal with the server stopped:',
  'packs.gpu.noCommand':
    'The server did not provide an install command. See the README for the GPU install steps.',
  // The disclosure label wherever a command block sits under a button that
  // does the same thing: the command is then a choice, not the only way
  // through, and it costs one folded line instead of three open ones.
  'packs.manualCommand': 'Manual install command',
  'packs.copy': 'Copy command',
  'packs.copied': 'Copied to clipboard.',
  'packs.copyFailed': 'Could not copy. Select the text and copy it by hand.',

  // Restart overlay (`packs.restart.done` / `.failed` are with the toasts above).
  'packs.restart.title': 'Server restarting',
  'packs.restart.body': 'Waiting for the server to come back. This page reloads by itself.',
  'packs.restart.elapsed': 'Waiting for {seconds} s',
  'packs.restart.timeout': 'The server has not come back after 10 minutes.',
  'packs.restart.notStarted': 'The server did not restart. Run this command, then reload:',
  'packs.restart.reload': 'Reload now',

  // The one pack toast fired from outside the pack store: a run stopped
  // because a node needs a pack that is not installed.
  'packs.toast.missingPack': 'This run needs the {pack} pack.',

  // Where the Package Center is opened from.
  'toolbar.settings.section.packs': 'Optional packs',
  'settings.packs.name': 'Package Center',
  'settings.packs.desc': 'Download models and libraries for the LLM nodes.',
  'settings.packs.summary': '{installed} of {total} packs installed',
  'settings.packs.summaryInstalling': 'Installing {pack}...',
  'settings.packs.unsupported': 'Not available on this server',
  'settings.packs.action': 'Open',
  'customTab.section.packs': 'Optional packs',
  'customTab.packs.open': 'Package Center...',
  'customTab.packs.empty': 'No optional packs available',
  'customTab.packs.hint':
    'Models and libraries for LLM nodes are installed from the Package Center',

  // Where the Plugin Center is opened from: the settings popover's own row,
  // beside the pack one it is modelled on. The sidebar's entry point is up
  // with the rest of the Custom & Plugins tab. `settings.packs.unsupported`
  // is reused as it is -- "Not available on this server" says the same thing
  // about either center, and a second translation of it is a second string to
  // keep in step.
  'toolbar.settings.section.plugins': 'Plugins',
  'settings.plugins.name': 'Plugin Center',
  'settings.plugins.desc': 'Install teaching node packs and plugins from GitHub.',
  'settings.plugins.summary': '{installed} installed, {available} available',
  'settings.plugins.summaryInstalling': 'Installing {plugin}...',
  // The button's visible word is `settings.packs.action` ("Open"), the same
  // as the pack row's. This is its accessible name: two buttons reading
  // "Open" say nothing in a list of controls, and speech input ("click Open")
  // has to reach one of them -- so the pack row keeps the bare word and this
  // one contains it, the way `paramField.installPackFor` does.
  'settings.plugins.action': 'Open Plugin Center',

  // "This needs a pack" — said on a select option, a node, a palette entry
  // and a refused run. Every one of them names the pack, because "needs a
  // pack" without a name is not something a user can act on.
  // Short on purpose: this goes inside an `<option>` in a full-width select,
  // which clips with no ellipsis when the current value is a gated one. The
  // sentence under the select still spells out "pack" and "model".
  'paramField.needsPack': 'needs {pack}',
  'paramField.needsModel': 'needs {item}',
  'paramField.packHint': '"{option}" needs the {pack} pack.',
  'paramField.modelHint': '"{option}" needs the model {item} from the {pack} pack.',
  'paramField.packHintOthers': 'Greyed-out options need an optional pack.',
  'paramField.installPack': 'Install pack',
  // The accessible name for the same button: one config panel can show two of
  // them, and "Install pack" twice is one entry repeated to anyone navigating
  // by control. A key rather than a hardcoded "label: pack", because the
  // separator is punctuation and punctuation is translated.
  'paramField.installPackFor': 'Install pack: {pack}',
  'config.needsPack': 'This node needs the {pack} pack.',
  'palette.needsPack': 'Needs pack',
  'palette.needsPack.title':
    'Needs the {pack} pack. You can drag it now and install the pack from the Package Center.',
  'node.needsPack': 'PACK',
  'node.needsPack.title': 'Needs the {pack} pack. Click to open the Package Center.',
  'node.paramNeedsPack': 'needs pack',
  'error.missingPack': 'This node needs the {pack} pack. Install it from the Package Center.',

  // Plugin Center. The host's own two toasts (a UI bundle that would not
  // load, the dev hot-reload) and everything the store says about an install,
  // an update, a removal or a switch. The panel's own copy arrives with the
  // panel.
  'pluginCenter.toast.frontendFailed': 'The UI of plugin "{plugin}" failed to load.',
  'pluginCenter.toast.frontendsReloaded': 'Plugin UI reloaded.',
  'pluginCenter.toast.installed': '{plugin} installed.',
  'pluginCenter.toast.updated': '{plugin} updated.',
  'pluginCenter.toast.upToDate': '{plugin} is up to date.',
  'pluginCenter.toast.removed': '{plugin} uninstalled.',
  'pluginCenter.toast.removeFailed': 'Could not remove {plugin}: {message}',
  'pluginCenter.toast.enabled': '{plugin} enabled.',
  'pluginCenter.toast.disabled': '{plugin} disabled.',
  'pluginCenter.toast.toggleFailed': 'Could not change {plugin}: {message}',
  // The change itself landed; re-reading the catalog, the node definitions
  // or the plugin UIs did not. Its own sentence, because saying "install
  // failed" about a successful uninstall is the one reading that is flatly
  // wrong.
  'pluginCenter.toast.refreshFailed':
    'Could not refresh the editor after the change: {message}',
  // NOTHING was installed. A plugin job reaches this status from one place
  // only -- the resolver refusing to replace a package the running server has
  // already loaded -- and that happens before a single file is written, so
  // the install has to be run again after the packages are in place.
  'pluginCenter.toast.needsRestart':
    'The install of {plugin} stopped: its Python packages need the server '
    + 'stopped first. The command is in the Plugin Center.',
  'pluginCenter.toast.inProgress':
    'A plugin is still installing. Open the Plugin Center to watch it.',
  'pluginCenter.toast.openCenter': 'Open Plugin Center',
  'pluginCenter.updateFailed': 'Update failed: {message}',
  'pluginCenter.uninstall': 'Uninstall',
  // Says what is lost AND what is not: nothing uninstalls a plugin's pip
  // packages, and a user who expects the disk space back should hear so here
  // rather than after saying yes.
  'pluginCenter.uninstallConfirm':
    'Uninstall {plugin}? Graphs that use its nodes will stop running. '
    + 'Its Python packages stay installed.',
  'pluginCenter.source.invalid':
    'Enter a catalog name, owner/repo[@ref] or a GitHub URL.',

  // The panel: its chrome, what the list can be saying instead of rows, and
  // the filter over them.
  'pluginCenter.title': 'Plugin Center',
  'pluginCenter.subtitle': 'Install packs of teaching nodes, and plugins from GitHub',
  'pluginCenter.close': 'Close Plugin Center',
  'pluginCenter.refresh': 'Refresh plugin status',
  'pluginCenter.list': 'Plugin list',
  'pluginCenter.loading': 'Loading plugins...',
  'pluginCenter.loadFail': 'Failed to load plugins: {error}',
  // A server older than the Plugin Center. `pluginCenter.error.unavailable`
  // says the same sentence about a server whose plugin service is not up:
  // the cause differs, what the user does about it does not.
  'pluginCenter.unsupported':
    'This server has no Plugin Center. Update CodefyUI and restart it.',
  'pluginCenter.empty': 'No plugins are available',
  'pluginCenter.filter.all': 'All',
  'pluginCenter.filter.installed': 'Installed',
  'pluginCenter.filter.available': 'Available',

  // Where a plugin came from. A plain third-party repository gets no chip --
  // the card prints owner/repo, and "GitHub" over a GitHub link says it twice.
  'pluginCenter.origin.builtin': 'Built-in',
  'pluginCenter.origin.official': 'Official',
  'pluginCenter.origin.local': 'Linked folder',
  'pluginCenter.homepage': 'Homepage',
  'pluginCenter.chapters': 'Lessons: {chapters}',

  // The two states a pack has no word for. The other four statuses reuse
  // `packs.status.*` and `customNodes.disabled`.
  'pluginCenter.status.removed': 'Removed',
  'pluginCenter.status.missingFiles': 'Files missing',

  // The row's buttons (`pluginCenter.uninstall` is up with its confirm).
  'pluginCenter.install': 'Install',
  'pluginCenter.enable': 'Enable',
  'pluginCenter.disable': 'Disable',
  'pluginCenter.update': 'Update',
  // The server refused an install because the plugin is already there. That
  // refusal is an offer rather than a failure, and this button accepts it.
  'pluginCenter.reinstall': 'Reinstall',

  // Installing something the catalog does not list.
  'pluginCenter.source.label': 'Install from GitHub',
  'pluginCenter.source.placeholder': 'owner/repo[@ref] or GitHub URL',
  'pluginCenter.source.review': 'Review',
  'pluginCenter.source.reviewing': 'Downloading...',
  'pluginCenter.source.fail': 'Could not fetch {source}: {message}',
  // A bare word that is not one of this build's packs. The refusal is a code
  // with no sentence in it, and no sentence written on the server could have
  // said the useful part: which names WOULD work. That list only exists in
  // the body, so the panel is what puts the two together.
  'pluginCenter.source.unknownName': 'No plugin is called "{source}".',
  'pluginCenter.source.knownNames': 'This server can install: {known}',

  // The consent screen. Nothing is installed until this has been read and
  // what it asks for has been ticked.
  'pluginCenter.review.title': 'Review before installing',
  'pluginCenter.review.author': 'Author: {author}',
  'pluginCenter.review.nodes': 'Nodes: {nodes}',
  'pluginCenter.review.capabilities': 'This plugin asks for:',
  // Says what granting is and is not: "capabilities" reads like a sandbox,
  // and this is a declaration -- nothing here is enforced at runtime.
  'pluginCenter.review.capNote':
    'Granting is a declaration, not a sandbox: the plugin may use these '
    + 'modules and will not be asked again.',
  'pluginCenter.review.grant': 'Grant these capabilities',
  'pluginCenter.review.trust': 'I trust this author. Allows: {modules}',
  'pluginCenter.review.frontend':
    'Ships JavaScript that runs in this editor with full access.',
  'pluginCenter.review.idConflict': 'The id "{id}" is reserved for a built-in pack.',
  // The 409 the store treats as an OFFER, said out loud: without it the only
  // sign is the Install button coming back as Reinstall.
  'pluginCenter.review.alreadyInstalled':
    '{plugin} is already installed. Reinstall replaces the installed copy '
    + 'with this one.',

  // One line per declared capability, each saying what granting it COSTS
  // rather than what it is called. An id this build has no line for is
  // printed as itself.
  'pluginCenter.cap.network':
    'network: reach any host, and write what it downloads to disk',
  'pluginCenter.cap.filesystem':
    'filesystem: use the file libraries (pathlib, shutil, zip/tar, sqlite3)',
  'pluginCenter.cap.processEnv':
    "process-env: the whole os module, including this process's environment "
    + 'and API keys, starting programs, deleting files',

  // The activity pane: what is running, and how it ended.
  'pluginCenter.activity.installing': 'Installing {plugin}',
  'pluginCenter.activity.updating': 'Updating {plugin}',
  'pluginCenter.activity.installed': 'Installed {plugin}.',
  'pluginCenter.activity.updated': 'Updated {plugin}.',
  'pluginCenter.activity.lost':
    'Lost contact with the server. Refresh to check the plugin status.',
  // Ends in a colon because a `CommandBlock` follows it, and the command is a
  // `uv pip install` line to run with the server STOPPED -- not a restart.
  'pluginCenter.activity.needsRestart':
    "The install stopped before changing anything: {plugin}'s Python packages "
    + 'would replace one the server has loaded. With the server stopped, run this, '
    + 'then install again:',
  'pluginCenter.activity.cliFallback': 'Or install from a terminal:',

  // The steps of an install, in the order they run. There is no `step.deps`
  // on purpose: pip is pip, and it reuses `packs.activity.step.pip`.
  'pluginCenter.step.resolve': 'Resolving the source',
  'pluginCenter.step.download': 'Downloading',
  'pluginCenter.step.extract': 'Unpacking',
  'pluginCenter.step.verify': 'Checking the code',
  'pluginCenter.step.stage': 'Copying files',
  'pluginCenter.step.lock': 'Recording the install',
  'pluginCenter.step.reload': 'Loading the nodes',

  // The refusals whose entire body is a code: these routes answer
  // `{detail: {code: ...}}` with no message at all, so without a sentence
  // here the panel shows the raw token -- "inspection_expired".
  'pluginCenter.error.unavailable':
    'This server has no Plugin Center. Update CodefyUI and restart it.',
  'pluginCenter.error.inspectionExpired': 'The review expired. Review the source again.',
  'pluginCenter.error.unknownJob': 'That install is no longer tracked. Refresh.',
  // A row for a plugin that has since been removed somewhere else: the button
  // was pressed against a catalog this tab read a while ago.
  'pluginCenter.error.notInstalled': 'This plugin is not installed any more. Refresh the list.',
  // One review at a time, server-side: this one is not about the source at
  // all, which is why it says "again" rather than anything about the repo.
  'pluginCenter.error.inspectBusy': 'Another review is still running. Try again in a moment.',
  // The three ways a source can turn out not to be a plugin this build
  // installs: the manifest, a repository or ref that is not there, and
  // GitHub itself.
  'pluginCenter.error.invalidManifest': "The plugin's manifest is invalid.",
  'pluginCenter.error.notFound': 'GitHub has no such repository or ref.',
  'pluginCenter.error.githubRateLimited':
    "GitHub's request limit was reached. Try again later, or set "
    + 'CODEFYUI_GITHUB_TOKEN on the server.',
  'pluginCenter.error.githubUnreachable': 'Could not reach GitHub.',

  // Source Control (the sidebar's fifth tab). Part 2 of the track: status,
  // stage / unstage / discard, commit, init and the commit identity. Branches,
  // remotes, history and the diff view arrive with their own strings later, so
  // nothing here mentions them.
  'git.action.more': 'More actions',
  'git.action.identity': 'Commit identity...',
  'git.action.docs': 'Setup guide',
  'git.menu.hideLayout': 'Hide layout files',
  // Appended to the menu item above, so the count of what the filter is
  // swallowing is visible without turning it off.
  'git.menu.hiddenCount': '({count} hidden)',
  'git.branch.label': 'Branch: {name}',
  'git.aheadBehind': '{ahead} to push, {behind} to pull',
  'git.noUpstream': 'Not published',
  'git.upstreamGone': 'Upstream deleted',
  'git.detached': 'Detached HEAD',
  'git.unborn': 'No commits yet',
  // {op} is a translated word from git.op.* below, never the wire opcode.
  'git.busy': 'Running {op}...',
  'git.op.status': 'status',
  'git.op.init': 'init',
  'git.op.stage': 'stage',
  'git.op.unstage': 'unstage',
  'git.op.discard': 'discard',
  'git.op.commit': 'commit',
  'git.op.identity': 'identity',
  'git.commit.placeholder': 'Message ({mod}+Enter to commit)',
  'git.commit.button': 'Commit',
  'git.commit.options': 'Commit options',
  'git.commit.all': 'Commit All (stages every change, including new files)',
  'git.commit.amend': 'Amend Last Commit',
  'git.commit.amending': 'Amending',
  'git.commit.amendPushed': 'Cannot amend: the last commit is already pushed',
  'git.commit.needMessage': 'Enter a message',
  'git.commit.nothingStaged': 'Nothing staged',
  'git.group.merge': 'Merge Changes',
  'git.group.staged': 'Staged Changes',
  'git.group.changes': 'Changes',
  'git.group.stageAll': 'Stage All',
  'git.group.unstageAll': 'Unstage All',
  'git.group.discardAll': 'Discard All Changes',
  // A whole-tree stage/unstage/discard silently leaves symlinks that resolve
  // outside the project alone; saying so is the difference between a filter
  // and a file that quietly never changes.
  'git.group.skipped': '{count} file(s) skipped (they point outside the project)',
  'git.file.stage': 'Stage',
  'git.file.unstage': 'Unstage',
  'git.file.discard': 'Discard Changes',
  'git.file.open': 'Open changes',
  'git.status.modified': 'Modified',
  'git.status.added': 'Added',
  'git.status.deleted': 'Deleted',
  'git.status.renamed': 'Renamed',
  'git.status.copied': 'Copied',
  'git.status.typechange': 'Type changed',
  'git.status.untracked': 'Untracked',
  'git.status.conflict': 'Conflict',
  'git.discard.confirm': 'Discard changes to {name}?',
  // An untracked file has no copy anywhere, so this one says "delete" and
  // says the loss is permanent.
  'git.discard.confirmUntracked':
    'Delete {name}? It is not tracked by git and cannot be recovered.',
  'git.discard.confirmAll':
    'Discard {changed} changed file(s) and delete {untracked} untracked file(s)? '
    + 'This cannot be undone.',
  'git.discard.action': 'Discard',
  'git.merge.banner': 'Merge in progress: resolve each file, then commit.',
  'git.empty.noProject': 'Source control needs a project directory.',
  'git.empty.noProjectHint': 'Create one and start the server on it:',
  'git.empty.notRepo': 'This project is not a git repository yet.',
  'git.empty.nested':
    'It sits inside another repository ({path}); initializing creates a separate one here.',
  'git.empty.init': 'Initialize Repository',
  'git.empty.gitMissing': 'git is not installed on the server computer.',
  'git.empty.gitMissingHint': 'Install it, then restart the server.',
  'git.empty.gitTooOld': 'git {version} is too old; 2.23 or newer is required.',
  'git.empty.clean': 'No changes',
  'git.identity.title': 'Commit identity',
  'git.identity.name': 'Name',
  'git.identity.email': 'Email',
  'git.identity.save': 'Save',
  'git.identity.scopeGlobal': 'from global git config',
  'git.identity.scopeLocal': 'for this project',
  'git.identity.missing': 'Not set',
  'git.toast.committed': 'Committed {sha}',
  'git.toast.initialized': 'Repository initialized',
  'git.toast.changedOnDisk': '{count} open graph(s) changed on disk.',
  'git.toast.reload': 'Reload',
  'git.toast.reloadConfirm':
    'Reload {count} graph(s) from disk? Unsaved edits in those tabs are lost.',
  'git.toast.missingOnDisk': '{name} no longer exists on this branch.',
  'git.error.busy': 'Another git action is still running.',
  'git.error.timeout': 'git did not finish within {seconds}s.',
  'git.error.nothingToCommit': 'Nothing to commit.',
  'git.error.identityMissing': 'Set your name and email before committing.',
  'git.error.detachedHead': 'HEAD is detached; create or switch to a branch first.',
  'git.error.mergeInProgress': 'Finish or abort the merge first.',
  'git.error.notFound': 'Not found: {what}',
  'git.error.invalid': 'Invalid value.',
  'git.error.notRepo': 'Not a git repository.',
  'git.error.generic': 'git failed: {message}',
  'git.error.loadFail': 'Could not read repository status: {error}',
  'git.error.dismiss': 'Dismiss',
  'git.error.details': 'Details',
} as const;

export type TranslationKey = keyof typeof en;
export default en;
