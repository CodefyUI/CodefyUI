import { describe, it, expect, beforeEach } from 'vitest';
import { useProjectStore, isProjectMode } from './projectStore';

beforeEach(() => {
  useProjectStore.setState({ projectDir: null, projectName: null, loaded: false });
});

describe('projectStore', () => {
  it('defaults to non-project mode', () => {
    expect(isProjectMode()).toBe(false);
    expect(useProjectStore.getState().projectDir).toBeNull();
  });

  it('setProject records dir + derives basename', () => {
    useProjectStore.getState().setProject('/home/me/my-service');
    expect(useProjectStore.getState().projectName).toBe('my-service');
    expect(isProjectMode()).toBe(true);
  });

  it('setProject(null) returns to non-project mode', () => {
    useProjectStore.getState().setProject('/x/y');
    useProjectStore.getState().setProject(null);
    expect(isProjectMode()).toBe(false);
    expect(useProjectStore.getState().projectName).toBeNull();
  });

  it('handles trailing separators and windows paths', () => {
    useProjectStore.getState().setProject('C:\\svc\\my-app\\');
    expect(useProjectStore.getState().projectName).toBe('my-app');
  });
});
