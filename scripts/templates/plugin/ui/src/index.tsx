import './styles.css';
import {
  mountTool,
  defineNodeRenderer,
  useGraph,
  useToast,
  type CodefyUIPluginAPI,
  type NodeRenderContext,
} from './sdk';

// A floating tool panel. The hooks (useGraph / useToast / ...) work anywhere
// inside a component mounted by mountTool or defineTool.
function Panel() {
  const graph = useGraph();
  const toast = useToast();
  return (
    <div className="cdui-panel">
      <div className="cdui-panel__title">{{plugin_name}}</div>
      <div className="cdui-panel__row">
        Nodes on canvas: <b>{graph.nodes.length}</b>
      </div>
      <button
        className="cdui-panel__btn"
        onClick={() => toast(`${graph.nodes.length} node(s) on the canvas`)}
      >
        Toast node count
      </button>
    </div>
  );
}

// A custom React body for a node's card. It receives the live node context
// (id, type, params) and re-renders whenever the node's params change.
function ExampleNodeBody({ node }: { node: NodeRenderContext['node'] }) {
  return (
    <div className="cdui-node-body">
      <div className="cdui-node-body__title">{{plugin_name}}</div>
      <div className="cdui-node-body__row">
        <span>params</span>
        <b>{Object.keys(node.params).length}</b>
      </div>
    </div>
  );
}

// The editor imports this bundle and calls the default export once at startup,
// passing the plugin API. You can mount tool panels and register node renderers.
export default function activate(api: CodefyUIPluginAPI) {
  mountTool(api, { id: '{{plugin_id}}-panel', title: '{{plugin_name}}' }, Panel);

  // Draw the Example node's card body with React. Plugin node types use the
  // snake_case namespace — id "{{plugin_id}}" exposes "{{plugin_snake}}:Example".
  api.nodes.registerRenderer(
    '{{plugin_snake}}:Example',
    defineNodeRenderer(ExampleNodeBody),
  );
}
