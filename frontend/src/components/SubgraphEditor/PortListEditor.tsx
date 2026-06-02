// frontend/src/components/SubgraphEditor/PortListEditor.tsx
import { useI18n } from '../../i18n';
import { generateId } from '../../utils';
import type { Node, Edge } from '@xyflow/react';
import type { LayerNodeData, PortDef } from './graphSerialization';

interface Props {
  node: Node<LayerNodeData>;
  edges: Edge[];
  onUpdatePorts: (nodeId: string, ports: PortDef[]) => void;
  onRemoveEdges: (edgeIds: string[]) => void;
}

export function PortListEditor({ node, edges, onUpdatePorts, onRemoveEdges }: Props) {
  const { t } = useI18n();
  const ports = node.data.ports ?? [];
  const isInput = node.data.layerType === 'Input';

  const setName = (portId: string, name: string) => {
    const next = ports.map((p) => (p.id === portId ? { ...p, name } : p));
    onUpdatePorts(node.id, next);
  };

  const addPort = () => {
    const next: PortDef[] = [...ports, { id: generateId(), name: `port${ports.length + 1}` }];
    onUpdatePorts(node.id, next);
  };

  const removePort = (portId: string) => {
    const next = ports.filter((p) => p.id !== portId);
    onUpdatePorts(node.id, next);
    // Remove any edges referencing this port
    const orphaned = edges
      .filter((e) =>
        isInput
          ? e.source === node.id && e.sourceHandle === portId
          : e.target === node.id && e.targetHandle === portId
      )
      .map((e) => e.id);
    if (orphaned.length > 0) onRemoveEdges(orphaned);
  };

  const names = ports.map((p) => p.name);
  const hasDuplicate = (name: string) => names.filter((n) => n === name).length > 1;

  return (
    <div style={{ padding: '12px 10px' }}>
      <div
        style={{
          fontSize: '0.9375rem',
          fontWeight: 700,
          color: '#eee',
          marginBottom: 12,
          paddingBottom: 8,
          borderBottom: '1px solid #333',
        }}
      >
        {node.data.layerType} — {t('subgraph.port.list')}
      </div>
      {ports.map((p) => (
        <div key={p.id} style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 6 }}>
          <input
            type="text"
            value={p.name}
            placeholder={t('subgraph.port.namePlaceholder')}
            onChange={(e) => setName(p.id, e.target.value)}
            title={hasDuplicate(p.name) ? t('subgraph.port.duplicate') : undefined}
            style={{
              flex: 1,
              padding: '5px 8px',
              background: '#222',
              border: hasDuplicate(p.name) ? '1px solid #F44336' : '1px solid #444',
              borderRadius: 4,
              color: '#ddd',
              fontSize: '0.8125rem',
              outline: 'none',
            }}
          />
          <button type="button"
            onClick={() => removePort(p.id)}
            disabled={ports.length === 1}
            style={{
              padding: '3px 8px',
              background: '#3a1515',
              border: '1px solid #F44336',
              borderRadius: 4,
              color: '#F44336',
              fontSize: '0.6875rem',
              cursor: ports.length === 1 ? 'not-allowed' : 'pointer',
              opacity: ports.length === 1 ? 0.4 : 1,
            }}
          >
            {t('subgraph.port.remove')}
          </button>
        </div>
      ))}
      <button type="button"
        onClick={addPort}
        style={{
          marginTop: 8,
          padding: '5px 10px',
          background: '#2a2a2a',
          border: '1px solid #444',
          borderRadius: 4,
          color: '#aaa',
          fontSize: '0.75rem',
          cursor: 'pointer',
          width: '100%',
        }}
      >
        {t('subgraph.port.add')}
      </button>
    </div>
  );
}
