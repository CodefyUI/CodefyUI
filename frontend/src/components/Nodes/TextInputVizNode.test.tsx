import { describe, it, expect, beforeEach } from 'vitest';
import { screen, fireEvent } from '@testing-library/react';
import { renderWithFlow } from '../../test/utils';
import { useI18n } from '../../i18n';
import { useTabStore } from '../../store/tabStore';
import type { NodeData, NodeDefinition } from '../../types';
import TextInputVizNode from './TextInputVizNode';

const flowProps = {
  zIndex: 0,
  isConnectable: true,
  positionAbsoluteX: 0,
  positionAbsoluteY: 0,
  dragging: false,
  draggable: false,
  selectable: true,
  deletable: true,
} as const;

function def(): NodeDefinition {
  return {
    node_name: 'TextInput',
    category: 'IO',
    description: '',
    inputs: [],
    outputs: [{ name: 'text', data_type: 'STRING', description: '', optional: false }],
    params: [],
  };
}

function data(overrides: Partial<NodeData> = {}): NodeData {
  return {
    label: 'Text Input',
    type: 'TextInput',
    params: {},
    definition: def(),
    executionStatus: 'idle',
    ...overrides,
  };
}

function renderNode(d: NodeData, id = 't1') {
  return renderWithFlow(
    <TextInputVizNode id={id} type="textInputNode" data={d} selected={false} {...flowProps} />,
  );
}

let captured: { id: string; params: Record<string, unknown> } | null = null;

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
  captured = null;
  const id = 'tab-ti';
  useTabStore.setState((s) => ({
    activeTabId: id,
    tabs: [{ ...s.tabs[0], id, name: 'Tab', nodes: [], edges: [], outputSummaries: {} }],
    updateNodeParams: (nodeId: string, params: Record<string, unknown>) => {
      captured = { id: nodeId, params };
    },
  }));
});

describe('TextInputVizNode', () => {
  it('renders the textarea with the current value and char count', () => {
    renderNode(data({ params: { value: 'hello' } }));
    const ta = document.querySelector('textarea') as HTMLTextAreaElement;
    expect(ta.value).toBe('hello');
    expect(screen.getByText(useI18n.getState().t('textInput.charCount', { count: '5' }))).toBeTruthy();
  });

  it('defaults to an empty value (char count 0) when params.value is absent', () => {
    renderNode(data({ params: {} }));
    const ta = document.querySelector('textarea') as HTMLTextAreaElement;
    expect(ta.value).toBe('');
    expect(screen.getByText(useI18n.getState().t('textInput.charCount', { count: '0' }))).toBeTruthy();
  });

  it('handles a missing params object (optional chaining on data.params)', () => {
    // params is required on NodeData, but the source uses `data.params?.value`.
    renderNode(data({ params: undefined as unknown as Record<string, unknown> }));
    const ta = document.querySelector('textarea') as HTMLTextAreaElement;
    expect(ta.value).toBe('');
  });

  it('coerces a non-string value to a string for display and count', () => {
    renderNode(data({ params: { value: 123 } }));
    const ta = document.querySelector('textarea') as HTMLTextAreaElement;
    expect(ta.value).toBe('123');
    expect(screen.getByText(useI18n.getState().t('textInput.charCount', { count: '3' }))).toBeTruthy();
  });

  it('typing in the textarea calls updateNodeParams with the new value', () => {
    renderNode(data({ params: { value: '' } }));
    const ta = document.querySelector('textarea') as HTMLTextAreaElement;
    fireEvent.change(ta, { target: { value: 'new text' } });
    expect(captured).toEqual({ id: 't1', params: { value: 'new text' } });
  });

  it('renders the placeholder text', () => {
    renderNode(data({ params: {} }));
    const ta = document.querySelector('textarea') as HTMLTextAreaElement;
    expect(ta.getAttribute('placeholder')).toBe(useI18n.getState().t('textInput.placeholder'));
  });
});
