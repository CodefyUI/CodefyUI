import { useCallback } from 'react';
import { useReactFlow } from '@xyflow/react';
import { useTabStore } from '../store/tabStore';
import { useNodeDefStore } from '../store/nodeDefStore';
import { insertExample } from '../utils/openExample';

export function useDragAndDrop() {
  const { screenToFlowPosition } = useReactFlow();
  const addNode = useTabStore((s) => s.addNode);
  const addPresetNode = useTabStore((s) => s.addPresetNode);
  const definitions = useNodeDefStore((s) => s.definitions);
  const presets = useNodeDefStore((s) => s.presets);

  const onDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
  }, []);

  const onDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();
      const position = screenToFlowPosition({
        x: event.clientX,
        y: event.clientY,
      });

      // Check for preset drop
      const presetName = event.dataTransfer.getData('application/codefyui-preset');
      if (presetName) {
        const preset = presets.find((p) => p.preset_name === presetName);
        if (preset) addPresetNode(preset, position);
        return;
      }

      // Check for node drop
      const nodeType = event.dataTransfer.getData('application/codefyui-node');
      if (nodeType) {
        const definition = definitions.find((d) => d.node_name === nodeType);
        if (definition) addNode(definition, position);
        return;
      }

      // Check for example drop (#348). The odd one out: the graph being
      // dropped is not in memory yet, so this branch is asynchronous where
      // the two above are not.
      //
      // `path` and `position` are both read off the event BEFORE that await
      // and closed over, which is not a style choice. A DataTransfer is
      // readable only while the event carrying it is being dispatched, and
      // `event.clientX/Y` describe a pointer that has moved on by the time
      // the fetch resolves — so reaching for either afterwards gets a drop
      // that lands nowhere, or lands wrong.
      //
      // Deliberately not awaited: `insertExample` reports failure with its
      // own toast and never throws, and an event handler that returns a
      // promise is a promise nobody is watching.
      const examplePath = event.dataTransfer.getData('application/codefyui-example');
      if (examplePath) void insertExample(examplePath, position);
    },
    [definitions, presets, screenToFlowPosition, addNode, addPresetNode]
  );

  return { onDragOver, onDrop };
}
