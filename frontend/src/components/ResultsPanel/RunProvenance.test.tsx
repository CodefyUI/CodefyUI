import { beforeEach, describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/react';
import type { NodeProvenance } from '../../api/rest';
import { useI18n } from '../../i18n';
import { RunProvenance } from './RunProvenance';

function record(overrides: Partial<NodeProvenance> = {}): NodeProvenance {
  return {
    schema_version: 1,
    node_id: 'csv',
    node_type: 'CSVReader',
    kind: 'dataset',
    mode: 'content',
    source: { type: 'file', name: null, location: 'files', path: 'iris.csv', subset: null },
    split: null,
    revision: { requested: null, resolved: null, kind: null },
    content: {
      method: 'sha256',
      digest: 'a'.repeat(64),
      exists: true,
      files: 1,
      bytes: 10,
      latest_mtime: null,
      truncated: false,
      unreadable: 0,
    },
    rows: 150,
    notes: ['rows counts the rows the node kept'],
    ...overrides,
  };
}

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
});

describe('RunProvenance', () => {
  it('renders nothing for a run without records', () => {
    const { container, rerender } = render(<RunProvenance records={undefined} />);
    expect(container).toBeEmptyDOMElement();
    rerender(<RunProvenance records={[]} />);
    expect(container).toBeEmptyDOMElement();
  });

  it('labels a content hash and a stat identity apart', () => {
    render(
      <RunProvenance
        records={[
          record(),
          record({
            node_id: 'hf',
            node_type: 'HuggingFaceDataset',
            mode: 'identity',
            source: { type: 'huggingface', name: 'ylecun/mnist', location: null, path: null, subset: 'mnist' },
            split: 'train',
            revision: { requested: null, resolved: 'b'.repeat(40), kind: 'hf-commit' },
            content: { ...record().content!, method: 'stat', digest: 'c'.repeat(64) },
            rows: null,
          }),
        ]}
      />,
    );
    expect(screen.getByTestId('run-provenance')).toBeInTheDocument();
    expect(screen.getByText('Data and models')).toBeInTheDocument();
    expect(screen.getByText('files:iris.csv')).toBeInTheDocument();
    expect(screen.getByText('sha256 aaaaaaaaaaaa')).toHaveAttribute('title', `sha256 of the content: ${'a'.repeat(64)}`);
    expect(screen.getByText('150 rows')).toBeInTheDocument();
    expect(screen.getByText('no revision')).toBeInTheDocument();
    expect(screen.getByText('ylecun/mnist/mnist [train]')).toBeInTheDocument();
    expect(screen.getByText('revision bbbbbbbbbbbb')).toHaveAttribute('title', 'b'.repeat(40));
    expect(screen.getByText('identity cccccccccccc').getAttribute('title')).toMatch(/not a content hash/);
    expect(screen.getByText('rows unknown')).toBeInTheDocument();
  });

  it('says when the content was not identified and falls back to the node id', () => {
    render(
      <RunProvenance
        records={[
          record({
            node_id: 'loader',
            node_type: 'ModelLoader',
            kind: 'model',
            source: { type: 'file', name: null, location: null, path: null, subset: null },
            content: null,
            rows: null,
            notes: ['no file selected'],
          }),
          record({
            node_id: 'missing',
            source: { type: 'file', name: null, location: null, path: 'x.csv', subset: null },
            content: { ...record().content!, digest: null },
          }),
        ]}
      />,
    );
    expect(screen.getByText('loader')).toHaveAttribute('title', '\nno file selected');
    expect(screen.getAllByText('content not identified')).toHaveLength(2);
    expect(screen.getByText('?:x.csv')).toBeInTheDocument();
  });
});
