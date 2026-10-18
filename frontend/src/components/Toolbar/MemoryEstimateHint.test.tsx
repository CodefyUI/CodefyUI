import { afterEach, describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/react';
import type { StaticMemoryEstimate } from '../../api/rest';
import { useI18n } from '../../i18n';
import { MemoryEstimateHint } from './MemoryEstimateHint';

const GIB = 1024 ** 3;
const WORKSPACE = 256 * 1024 ** 2;

/** The server's total: the tensors plus 20% reserve, plus the workspace. */
function total(weights: number, activations: number): number {
  return Math.ceil((weights + activations) * 1.2 + WORKSPACE);
}

const STATIC: StaticMemoryEstimate = {
  bytes: total(4000, GIB), confidence: 'medium', parameters: 1000, trainable_parameters: 1000, training: false,
  optimizer: null, multiplier: 1, batch_size: 0, eval_batch_size: 16, precision: 'fp32',
  weights_bytes: 4000, activation_bytes: GIB, workspace_bytes: WORKSPACE,
  models: [{ type: 'SequentialModel', parameters: 1000, activations_known: true, probe_input: [[8]] }],
  uncounted: [], notes: [], batch_assumed: false, activation_phase: 'evaluation',
};

const empty = { suggested_gib: null, source: null, peak_bytes: null, from_run: null };

afterEach(() => {
  useI18n.setState({ locale: 'en' });
});

describe('MemoryEstimateHint (#794)', () => {
  it('reads an inference estimate at its evaluation batch, its parts adding up to the total', () => {
    render(<MemoryEstimateHint estimate={{ ...empty, estimate_gib: 1.5, static: STATIC }} error={null} />);
    expect(screen.getByText('Estimate')).toBeInTheDocument();
    expect(screen.getByText(/^Not a measured peak/)).toHaveTextContent(
      '1,000 parameters x 4 bytes (weights) = 3.9 KB, activations of an evaluation pass at batch 16 about 1.0 GB, '
      + '256.0 MB of library workspace, and 204.8 MB of allocator reserve (20%). Queued as it is, the run is capped '
      + 'to this estimate (marked est. in Runs)',
    );
  });

  it('leaves the activations out when the estimate has none', () => {
    const statics = { ...STATIC, activation_phase: 'none', activation_bytes: 0, bytes: total(4000, 0) };
    render(<MemoryEstimateHint estimate={{ ...empty, estimate_gib: 0.26, static: statics }} error={null} />);
    expect(screen.getByText(/^Not a measured peak/)).toHaveTextContent(
      'Estimated from the model: 1,000 parameters x 4 bytes (weights) = 3.9 KB, 256.0 MB of library workspace, '
      + 'and 800 B of allocator reserve (20%).',
    );
  });

  it('counts frozen parameters once and autocast copies apart, so the weights add up', () => {
    // 1,000 parameters, 600 trainable, Adam (4x), bf16: 4,000 + 600 x 12 +
    // 1,000 x 2 bytes of weights.
    const weights = 1000 * 4 + 600 * 12 + 1000 * 2;
    const statics: StaticMemoryEstimate = {
      ...STATIC, training: true, optimizer: 'Adam', multiplier: 4, trainable_parameters: 600, precision: 'bf16',
      batch_size: 64, eval_batch_size: 64, weights_bytes: weights, activation_bytes: 2 * 1024 ** 2,
      bytes: total(weights, 2 * 1024 ** 2), activation_phase: 'training',
    };
    render(<MemoryEstimateHint estimate={{ ...empty, estimate_gib: 0.26, static: statics }} error={null} />);
    expect(screen.getByText(/^Not a measured peak/)).toHaveTextContent(
      'Estimated from the model: 1,000 parameters x 4 bytes (weights) = 3.9 KB, 600 trainable parameters x 12 bytes '
      + '(gradients, optimizer state) = 7.0 KB, 1,000 parameters x 2 bytes (bf16 copies for autocast) = 2.0 KB, '
      + 'activations of a training step at batch 64 about 2.0 MB, 256.0 MB of library workspace, and 412.2 KB of '
      + 'allocator reserve (20%).',
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
      estimate={{ ...empty, static: { ...STATIC, bytes: 2 * GIB, confidence: 'low', uncounted: ['MyPluginNet'], models: [] } }}
      error={null}
    />);
    expect(screen.getByText(/^About 2.0 GiB/)).toHaveTextContent(
      'About 2.0 GiB, too rough to offer: could not count MyPluginNet. Declare a budget.',
    );
  });

  it('says when the input size or the batch was guessed', () => {
    render(<MemoryEstimateHint
      estimate={{
        ...empty,
        static: {
          ...STATIC, bytes: 2 * GIB, confidence: 'low', batch_assumed: true, eval_batch_size: 1,
          models: [{ ...STATIC.models[0], type: 'SequentialModel', shape_guessed: true }],
        },
      }}
      error={null}
    />);
    expect(screen.getByText(/^About 2.0 GiB/)).toHaveTextContent(
      'About 2.0 GiB, too rough to offer: the graph does not give the input size of SequentialModel and no batch '
      + 'size in the graph (assumed 1). Declare a budget.',
    );
  });

  it('renders in zh-TW', () => {
    useI18n.setState({ locale: 'zh-TW' });
    const { rerender } = render(
      <MemoryEstimateHint estimate={{ ...empty, estimate_gib: 1.5, static: STATIC }} error={null} />,
    );
    expect(screen.getByText('估計值')).toBeInTheDocument();
    expect(screen.getByText(/^這不是實測峰值/)).toHaveTextContent('配置器保留 204.8 MB（20%）');
    rerender(<MemoryEstimateHint
      estimate={{ ...empty, static: { ...STATIC, confidence: 'low', training: true, batch_assumed: true, batch_size: 32 } }}
      error={null}
    />);
    expect(screen.getByText('粗估')).toBeInTheDocument();
    expect(screen.getByText(/^約 /)).toHaveTextContent('圖中沒有 batch size（假設為 32）');
    rerender(<MemoryEstimateHint
      estimate={{ ...empty, source: 'history', peak_bytes: GIB, suggested_gib: 1.2, from_run: 'r' }}
      error={null}
    />);
    expect(screen.getByText('實測')).toBeInTheDocument();
  });
});
