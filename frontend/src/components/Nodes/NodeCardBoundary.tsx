import { Component, type ComponentType, type ReactNode } from 'react';
import type { NodeProps, NodeTypes } from '@xyflow/react';
import { useI18n } from '../../i18n';
import styles from './NodeCardBoundary.module.css';

/** One line for the box's tooltip: what the card threw, for a bug report. */
function errorLine(error: unknown): string {
  try {
    if (error instanceof Error) return error.message ? `${error.name}: ${error.message}` : error.name;
    return String(error);
  } catch {
    // A thrown value String() cannot convert: a throwing toString, or no prototype.
    return 'Error';
  }
}

function CardFailed({ id, data, error }: { id: string; data: unknown; error: unknown }) {
  const { t } = useI18n();
  const fields = (typeof data === 'object' && data !== null ? data : {}) as {
    label?: unknown;
    type?: unknown;
    layerType?: unknown;
  };
  // A layers-editor card has no label or type, only its layer type.
  const type =
    typeof fields.type === 'string'
      ? fields.type
      : typeof fields.layerType === 'string'
        ? fields.layerType
        : '';
  const label = typeof fields.label === 'string' && fields.label !== '' ? fields.label : type || id;
  return (
    <div className={styles.box} title={errorLine(error)}>
      <div className={styles.name}>{label}</div>
      {/* The type only when it adds something: a renamed node, not "Note" over "note". */}
      {type !== '' && type.toLowerCase() !== label.toLowerCase() && (
        <div className={styles.type}>{type}</div>
      )}
      <div className={styles.message}>{t('node.cardFailed')}</div>
    </div>
  );
}

interface Props {
  id: string;
  /** The node's data. A new object -- an edit, a run, an undo -- draws the card again. */
  data: unknown;
  children: ReactNode;
}

interface State {
  failed: boolean;
  /** What the card threw. Any value can be thrown, null included, hence `failed`. */
  error: unknown;
  /** The data the current state belongs to. */
  data: unknown;
}

/**
 * One node card that throws while it draws costs that card, not the page.
 *
 * React unmounts the whole tree on a render error nobody catches, so before
 * this a single card -- a viz card handed a tensor it did not expect, a
 * plugin's card -- replaced the editor with AppErrorBoundary's recovery
 * screen, which in an exam is the worst outcome there is. Now that card
 * becomes a small box naming the node, and the rest of the canvas, the wires
 * and Run keep working. The node itself is untouched: it can still be
 * selected, moved, deleted or undone.
 *
 * The card is drawn again when the node's data changes, since whatever broke
 * it may be gone; the same data is not retried, so a card that keeps throwing
 * costs one failed render per change, not a loop.
 */
export class NodeCardBoundary extends Component<Props, State> {
  state: State = { failed: false, error: null, data: this.props.data };

  static getDerivedStateFromProps(props: Props, state: State): Partial<State> | null {
    if (props.data === state.data) return null;
    return { failed: false, error: null, data: props.data };
  }

  static getDerivedStateFromError(error: unknown): Partial<State> {
    return { failed: true, error };
  }

  render() {
    if (this.state.failed) {
      return <CardFailed id={this.props.id} data={this.props.data} error={this.state.error} />;
    }
    return this.props.children;
  }
}

/** `Card`, drawn inside its own NodeCardBoundary. */
export function withNodeCardBoundary<P extends NodeProps>(Card: ComponentType<P>): ComponentType<P> {
  function InNodeCardBoundary(props: P) {
    return (
      <NodeCardBoundary id={props.id} data={props.data}>
        <Card {...props} />
      </NodeCardBoundary>
    );
  }
  InNodeCardBoundary.displayName = `NodeCardBoundary(${Card.displayName || Card.name || 'Card'})`;
  return InNodeCardBoundary;
}

/**
 * A React Flow `nodeTypes` map with every card in its own NodeCardBoundary.
 * Call it once, at module level: React Flow needs the same map every render.
 */
export function withNodeCardBoundaries(types: NodeTypes): NodeTypes {
  return Object.fromEntries(
    Object.entries(types).map(([type, Card]) => [type, withNodeCardBoundary(Card)]),
  );
}
