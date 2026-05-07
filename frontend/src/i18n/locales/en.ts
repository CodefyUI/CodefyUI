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
  'toolbar.load': 'Load',
  'toolbar.load.title': 'Load a saved graph',
  'toolbar.load.fail': 'Load failed: {error}',
  'toolbar.load.loading': 'Loading...',
  'toolbar.load.empty': 'No saved graphs',
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
  'toolbar.exportPython': 'Export as Python',
  'toolbar.exportPython.title': 'Download graph as a standalone Python script',
  'toolbar.exportPython.empty': 'Canvas is empty — add some nodes before exporting.',
  'toolbar.exportPython.fail': 'Python export failed: {error}',

  // Status
  'status.idle': 'Idle',
  'status.running': 'Running',
  'status.completed': 'Completed',
  'status.error': 'Error',
  'status.skipped': 'Skipped',
  'status.cached': 'Cached',

  // Connection (WebSocket reconnect surface)
  'connection.lost': 'Connection lost — reconnecting…',
  'connection.restored': 'Connection restored',
  'connection.failed': 'Could not reconnect to the execution server',

  // Node Palette
  'palette.title': 'Nodes',
  'palette.search': 'Search nodes...',
  'palette.loading': 'Loading nodes...',
  'palette.loadFail': 'Failed to load nodes: {error}',
  'palette.retry': 'Retry',
  'palette.noMatch': 'No matching nodes',
  'palette.empty': 'No nodes available',
  'palette.hint': 'Drag nodes onto the canvas',
  'palette.composite': 'Composite',
  'palette.basic': 'Basic',

  // Config Panel
  'config.title': 'Node Config',
  'config.selectNode': 'Select a node to configure',
  'config.parameters': 'Parameters',
  'config.noParams': 'No configurable parameters',
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
  'node.error': 'Error: {error}',

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
  'empty.section.trainable': 'Trainable workflows',
  'empty.section.architecture': 'Architecture gallery',

  // Context Menu
  'contextMenu.rename': 'Rename',
  'contextMenu.duplicate': 'Duplicate',
  'contextMenu.delete': 'Delete',
  'contextMenu.rename.prompt': 'Enter a new name for this node:',
  'contextMenu.addTextNote': 'Add Text Note',
  'contextMenu.addImageNote': 'Add Image Note',

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

  // Subgraph Editor (SequentialModel)
  'subgraph.title': 'Model Architecture Editor',
  'subgraph.palette': 'Layers',
  'subgraph.apply': 'Apply',
  'subgraph.cancel': 'Cancel',
  'subgraph.import': 'Import',
  'subgraph.export': 'Export',
  'subgraph.import.title': 'Import a saved model architecture',
  'subgraph.export.title': 'Export current architecture as JSON',
  'subgraph.empty': 'Drag layers from the left panel to build your model',
  'subgraph.layerCount': '{count} layers',
  'subgraph.params': 'Parameters',
  'subgraph.noParams': 'No parameters',
  'subgraph.deleteLayer': 'Delete',
  'subgraph.hint': 'Double-click to edit architecture',
  'subgraph.import.fail': 'Import failed: {error}',
  'subgraph.import.selectModel': 'Select SequentialModel to Import',
  'subgraph.import.noContent': 'No importable layers or SequentialModel nodes found in this file.',
  'subgraph.searchLayers': 'Search layers...',
  'subgraph.snapOn': 'Snap: ON',
  'subgraph.snapOff': 'Snap: OFF',
  'subgraph.snapTitle': 'Toggle grid snap',
  'subgraph.autoLayout': 'Auto Layout',
  'subgraph.autoLayoutTitle': 'Arrange nodes top-to-bottom by connection order',
  'subgraph.category.io': 'I/O',
  'subgraph.category.merge': 'Merge',
  'subgraph.validation.cycle': 'Graph contains a cycle',
  'subgraph.validation.noInput': 'Graph must have exactly one Input node',
  'subgraph.validation.noOutput': 'Graph must have exactly one Output node',
  'subgraph.port.add': '+ Add port',
  'subgraph.port.remove': 'Remove',
  'subgraph.port.namePlaceholder': 'port name',
  'subgraph.port.duplicate': 'Duplicate port name',
  'subgraph.port.list': 'Ports',
  'subgraph.layerNode.moreParams': '+{count} more',

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
  'paramField.download': 'Download selected file',
  'paramField.refresh': 'Refresh file list',
  'paramField.selectFile': '-- select file --',
  'paramField.uploadFailed': 'Upload failed',
  'paramField.downloadFailed': 'Download failed',

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
  'inspector.valueDiff.noValues': 'No values captured for this port',
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
  'toolbar.settings.section.recording': 'Recording & Inspection',
  'toolbar.settings.section.training': 'Training Behavior',
  'toolbar.settings.section.editor': 'Editor',

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
  'settings.gridSnap.name': 'Grid snap',
  'settings.gridSnap.desc': 'Snap dragged nodes to the canvas grid.',
  'settings.tooltips.name': 'Show node tooltips',
  'settings.tooltips.desc': 'Reveal the description card when hovering nodes on the canvas.',
  'settings.nodeMode.name': 'Node category mode',
  'settings.nodeMode.desc': 'Basic shows only the essential categories in the sidebar; All shows every category.',
  'settings.nodeMode.basic': 'Basic',
  'settings.nodeMode.all': 'All',

  // LLM
  'tokenizer.tokenCount': '{count} tokens',
  'tokenizer.emptyOutput': 'No tokens — input text was empty.',
  'tokenizer.runHint': 'Run the graph to see tokens',
  'tokenizer.truncatedInline': 'showing first {shown} of {total} — see Inspector for full list',
  'scatter.runHint': 'Run the graph to see the projection',
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
} as const;

export type TranslationKey = keyof typeof en;
export default en;
