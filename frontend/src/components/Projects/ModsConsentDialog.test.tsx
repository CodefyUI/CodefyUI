import { describe, it, expect, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import { useI18n } from '../../i18n';
import type { ModsFile, ModsFinding } from '../../api/projects';
import { ModsFileRow, findingText, findingTitle } from './ModsConsentDialog';

// Two wording fixes in the mods prompt (phase 5): the capability an import
// needs is named in the page's language, and one line is "1 line".

const t = (key: Parameters<ReturnType<typeof useI18n.getState>['t']>[0], vars?: Record<string, string | number>) =>
  useI18n.getState().t(key, vars);

const importFinding: ModsFinding = {
  line: 3, kind: 'import', module: 'socket', capability: 'network', construct: null,
} as ModsFinding;

const file = (over: Partial<ModsFile>): ModsFile => ({
  path: 'nodes/a.py', size: 10, role: 'node', sha256: '0'.repeat(64), finding: null, ...over,
});

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
});

describe('the import finding names its capability', () => {
  it('in English', () => {
    expect(findingText(t, importFinding)).toBe('Line 3 imports socket: needs network');
  });

  it('in Chinese, with no English id in the sentence', () => {
    useI18n.setState({ locale: 'zh-TW' });
    expect(findingText(t, importFinding)).toBe('第 3 行匯入 socket：需要網路');
    expect(findingText(t, { ...importFinding, capability: 'process-env' })).toBe('第 3 行匯入 socket：需要行程與環境變數存取');
    expect(findingText(t, { ...importFinding, capability: 'filesystem' })).toBe('第 3 行匯入 socket：需要檔案存取');
  });

  it('prints an id this build has no name for as it is', () => {
    expect(findingText(t, { ...importFinding, capability: 'gpu-direct' })).toBe('Line 3 imports socket: needs gpu-direct');
  });

  it('keeps what granting it costs in the tooltip', () => {
    expect(findingTitle(t, importFinding)).toBe('network: reach any host, and write what it downloads to disk');
    expect(findingTitle(t, { ...importFinding, kind: 'syntax' })).toBeUndefined();
  });
});

describe('line count', () => {
  it('says "1 line" for one and "{n} lines" for more', () => {
    const { unmount } = render(<ul><ModsFileRow file={file({ lines: 1 })} mark={null} /></ul>);
    expect(screen.getByTestId('mods-file').textContent).toContain('1 line');
    expect(screen.getByTestId('mods-file').textContent).not.toContain('1 lines');
    unmount();
    render(<ul><ModsFileRow file={file({ lines: 2 })} mark={null} /></ul>);
    expect(screen.getByTestId('mods-file').textContent).toContain('2 lines');
  });

  it('in Chinese', () => {
    useI18n.setState({ locale: 'zh-TW' });
    render(<ul><ModsFileRow file={file({ lines: 1 })} mark={null} /></ul>);
    expect(screen.getByTestId('mods-file').textContent).toContain('1 行');
  });
});
