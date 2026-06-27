/**
 * xyflow node component for plugin-provided nodes.
 *
 * Namespaced plugin nodes (e.g. `my-plugin:Foo`) that have no first-party viz
 * renderer route here. When the plugin has registered a custom renderer for the
 * node's type, we render the standard node card (`BaseNodeBody`) with the
 * plugin's UI injected into the body slot; otherwise this is identical to the
 * default `baseNode`. The renderer registry is observable, so a renderer
 * registered after the node mounts shows up without a reload.
 */
import { memo, useEffect, useRef, useSyncExternalStore } from 'react';
import type { NodeProps } from '@xyflow/react';
import type { AppNode } from '../../types';
import { BaseNodeBody } from './BaseNode';
import {
  getNodeRenderer,
  subscribeNodeRenderers,
  type PluginNodeContext,
  type PluginNodeRenderer,
} from '../../plugins/nodeRenderers';

function nodeContext(props: NodeProps<AppNode>): PluginNodeContext {
  return {
    node: {
      id: props.id,
      type: props.data.type,
      params: props.data.params,
      definition: props.data.definition,
    },
  };
}

/** Hosts the plugin's own (React) root inside the node card body slot. */
function PluginNodeBody({
  nodeProps, renderer,
}: { nodeProps: NodeProps<AppNode>; renderer: PluginNodeRenderer }) {
  const ref = useRef<HTMLDivElement>(null);

  // Mount once per renderer; tear down on unmount or renderer change.
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    try {
      renderer.mount(el, nodeContext(nodeProps));
    } catch (err) {
      console.warn('[plugins] node renderer mount failed:', err);
    }
    return () => {
      try { renderer.unmount?.(el); } catch { /* ignore plugin teardown errors */ }
    };
    // nodeProps intentionally omitted — updates flow through the effect below.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [renderer]);

  // Push fresh node state to the renderer whenever params/type change.
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    try {
      renderer.update?.(el, nodeContext(nodeProps));
    } catch (err) {
      console.warn('[plugins] node renderer update failed:', err);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [renderer, nodeProps.data.params, nodeProps.data.type]);

  return <div className="cdui-plugin-node-body" ref={ref} />;
}

function PluginNodeBridge(props: NodeProps<AppNode>) {
  const renderer = useSyncExternalStore(
    subscribeNodeRenderers,
    () => getNodeRenderer(props.data.type),
  );
  if (!renderer) return <BaseNodeBody {...props} />;
  return (
    <BaseNodeBody
      {...props}
      bodyExtra={<PluginNodeBody nodeProps={props} renderer={renderer} />}
    />
  );
}

export default memo(PluginNodeBridge);
