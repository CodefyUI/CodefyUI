import { afterEach, describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/react';
import type { StaticMemoryEstimate } from '../../api/rest';
import { useI18n } from '../../i18n';
import { MemoryEstimateHint } from './MemoryEstimateHint';

const GIB = 1024 ** 3;

const STATIC: StaticMemoryEstimate = {
  bytes: 2 * GIB, confidence: 'medium', parameters: 1000, trainable_parameters: 1000, training: false,
  optimizer: null, multiplier: 1, batch_size: 0, eval_batch_size: 16, precision: 'fp32',
  weights_bytes: 4000, activation_bytes: GIB, workspace_bytes: 256 * 1024 ** 2,
  models: [{ type: 'SequentialModel', parameters: 1000, activations_known: true, probe_input: [[8]] }],
  uncounted: [], notes: [], batch_assumed: false, activation_phase: 'evaluation',
};

const empty = { suggested_gib: null, source: null, peak_bytes: null, from_run: null };

afterEach(() => {
  useI18n.setState({ locale: 'en' });
});

describe('MemoryEstimateHint (#794)', () => {
  it('reads an inference estimate at its evaluation batch, 4 bytes per parameter', () => {
    render(<MemoryEstimateHint estimate={{ ...empty, suggested_gib: 2, source: 'estimate', static: STATIC }} error={null} />);
    expect(screen.getByText('Estimate')).toBeInTheDocument();
    expect(screen.getByText(/^Not a measured peak/)).toHaveTextContent(
      '1,000 parameters x 4 bytes (weights) = 3.9 KB, activations of an evaluation pass at batch 16 about 1.0 GB, '
      + 'and 256.0 MB of library workspace.',
    );
  });

  it('leaves the activations out when the estimate has none', () => {
    render(<MemoryEstimateHint
      estimate={{ ...empty, suggested_gib: 1, source: 'estimate', static: { ...STATIC, activation_phase: 'none' } }}
      error={null}
    />);
    expect(screen.getByText(/^Not a measured peak/)).toHaveTextContent(
      'Estimated from the model: 1,000 parameters x 4 bytes (weights) = 3.9 KB and 256.0 MB of library workspace.',
    );
  });

  it('says there is nothing to suggest for no estimate, and for a history entry without a peak', () => {
    const { rerender } = render(<MemoryEstimateHint estimate={null} error={null} />);
    expect(screen.getByText(/No earlier run of this graph reported its peak memory/)).toBeInTheDocument();
    rerender(<MemoryEstimateHint estimate={{ ...empty, source: 'history' }} error={null} />);
    expect(screen.getByText(/No earlier run of this graph reported its peak memory/)).toBeInTheDocument();
    expect(screen.queryByText('Measured')).toBeNull();
  });

  it('gives one reason for a rough estimate whose model was not counted', () => {
    render(<MemoryEstimateHint
      estimate={{ ...empty, static: { ...STATIC, confidence: 'low', uncounted: ['MyPluginNet'], models: [] } }}
      error={null}
    />);
    expect(screen.getByText(/^About 2.0 GiB/)).toHaveTextContent(
      'About 2.0 GiB, too rough to use: could not count MyPluginNet. Declare a budget.',
    );
  });

  it('renders in zh-TW', () => {
    useI18n.setState({ locale: 'zh-TW' });
    const { rerender } = render(
      <MemoryEstimateHint estimate={{ ...empty, suggested_gib: 2, source: 'estimate', static: STATIC }} error={null} />,
    );
    expect(screen.getByText('估計值')).toBeInTheDocument();
    expect(screen.getByText(/^這不是實測峰值/)).toBeInTheDocument();
    rerender(<MemoryEstimateHint
      estimate={{ ...empty, static: { ...STATIC, confidence: 'low', batch_assumed: true, batch_size: 32 } }}
      error={null}
    />);
    expect(screen.getByText('粗估')).toBeInTheDocument();
    expect(screen.getByText(/^約 2.0 GiB/)).toHaveTextContent('圖中沒有 batch size（假設為 32）');
    rerender(<MemoryEstimateHint
      estimate={{ ...empty, source: 'history', peak_bytes: GIB, suggested_gib: 1.2, from_run: 'r' }}
      error={null}
    />);
    expect(screen.getByText('實測')).toBeInTheDocument();
  });
});
