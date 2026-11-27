import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import {
  PROJECT_CHANGED_EVENT,
  _setSessionTokenForTesting,
  apiFetch,
  setProjectId,
  wsUrlWithToken,
} from './_auth';
import { createProject, formatBytes, ProjectError } from './projects';

const g = globalThis as unknown as { fetch: typeof fetch };
let originalFetch: typeof fetch;

function response(status: number, body: unknown): Response {
  const text = JSON.stringify(body);
  return new Response(text, {
    status,
    headers: { 'Content-Type': 'application/json' },
  });
}

beforeEach(() => {
  originalFetch = g.fetch;
  _setSessionTokenForTesting('tok');
  setProjectId(null);
});

afterEach(() => {
  g.fetch = originalFetch;
  setProjectId(null);
});

describe('the project header (R3)', () => {
  it('rides on every write once the project is known, never on a read', async () => {
    const seen: Headers[] = [];
    g.fetch = vi.fn(async (_url: RequestInfo | URL, init?: RequestInit) => {
      seen.push(new Headers(init?.headers));
      return response(200, {});
    }) as typeof fetch;

    await apiFetch('/api/graph/save', { method: 'POST', body: '{}' });
    expect(seen[0].get('X-CodefyUI-Project')).toBeNull();

    setProjectId('0123456789abcdef');
    await apiFetch('/api/graph/save', { method: 'POST', body: '{}' });
    expect(seen[1].get('X-CodefyUI-Project')).toBe('0123456789abcdef');
    expect(seen[1].get('X-CodefyUI-Token')).toBe('tok');

    await apiFetch('/api/graph/list');
    // A read goes to fetch untouched: no token, no project.
    expect(seen[2].get('X-CodefyUI-Project')).toBeNull();
    expect(seen[2].get('X-CodefyUI-Token')).toBeNull();
  });

  it('puts the project on the execution socket URL', async () => {
    setProjectId('feedfacecafebeef');
    const url = new URL(await wsUrlWithToken('/ws/execution'));
    expect(url.searchParams.get('project')).toBe('feedfacecafebeef');
    expect(url.searchParams.get('token')).toBe('tok');
  });

  it('announces a project_changed refusal and still hands back the response', async () => {
    setProjectId('0123456789abcdef');
    g.fetch = vi.fn(async () => response(409, {
      detail: { code: 'project_changed', message: 'reload', details: { current: 'aaaa' } },
    })) as typeof fetch;
    const heard = new Promise<string | null>((resolve) => {
      window.addEventListener(PROJECT_CHANGED_EVENT, (event) => {
        resolve((event as CustomEvent).detail.projectId);
      }, { once: true });
    });
    const res = await apiFetch('/api/graph/save', { method: 'POST', body: '{}' });
    expect(res.status).toBe(409);
    expect((await res.json()).detail.code).toBe('project_changed');
    expect(await heard).toBe('aaaa');
  });
});

describe('api/projects', () => {
  it('turns a refusal into a ProjectError with the server sentence', async () => {
    g.fetch = vi.fn(async () => response(409, {
      detail: { code: 'runs_active', message: 'Cannot switch projects: 1 run is running', details: [] },
    })) as typeof fetch;
    const error = await createProject('x').catch((err: unknown) => err);
    expect(error).toBeInstanceOf(ProjectError);
    expect((error as ProjectError).code).toBe('runs_active');
    expect((error as ProjectError).message).toBe('Cannot switch projects: 1 run is running');
  });

  it('keeps a clone hint', async () => {
    g.fetch = vi.fn(async () => response(400, {
      detail: { code: 'not_a_project', message: 'no manifest', details: 'Add the manifest' },
    })) as typeof fetch;
    const error = await createProject('x').catch((err: unknown) => err) as ProjectError;
    expect(error.hint).toBe('Add the manifest');
  });

  it('formats sizes', () => {
    expect(formatBytes(512)).toBe('512 B');
    expect(formatBytes(1536)).toBe('1.5 KB');
    expect(formatBytes(50 * 1024 * 1024)).toBe('50 MB');
  });
});
