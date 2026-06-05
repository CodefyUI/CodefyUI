import type { NodeDefinition, GraphSaveData, PresetDefinition } from '../types';
import { apiFetch } from './_auth';

const BASE_URL = '/api';

export async function fetchNodeDefinitions(): Promise<NodeDefinition[]> {
  const res = await fetch(`${BASE_URL}/nodes`);
  if (!res.ok) throw new Error(`Failed to fetch node definitions: ${res.statusText}`);
  return res.json();
}

export async function fetchPresetDefinitions(): Promise<PresetDefinition[]> {
  const res = await fetch(`${BASE_URL}/presets`);
  if (!res.ok) throw new Error(`Failed to fetch presets: ${res.statusText}`);
  return res.json();
}

export interface DeviceInfo {
  value: string;
  label: string;
  detail: string;
  available: boolean;
}

export interface DevicesResponse {
  default: string;
  devices: DeviceInfo[];
}

/** Compute devices available for graph execution (CPU + any GPU backend,
 * with NVIDIA-CUDA / AMD-ROCm / Apple-MPS labels). Powers the global device
 * selector. */
export async function fetchDevices(): Promise<DevicesResponse> {
  const res = await fetch(`${BASE_URL}/system/devices`);
  if (!res.ok) throw new Error(`Failed to fetch devices: ${res.statusText}`);
  return res.json();
}

export async function validateGraph(nodes: any[], edges: any[]) {
  const res = await apiFetch(`${BASE_URL}/graph/validate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ nodes, edges }),
  });
  if (!res.ok) throw new Error(`Validation failed: ${res.statusText}`);
  return res.json();
}

export async function saveGraph(data: GraphSaveData) {
  const res = await apiFetch(`${BASE_URL}/graph/save`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
  if (!res.ok) throw new Error(`Save failed: ${res.statusText}`);
  return res.json();
}

export async function loadGraph(name: string) {
  const res = await fetch(`${BASE_URL}/graph/load/${encodeURIComponent(name)}`);
  if (!res.ok) throw new Error(`Load failed: ${res.statusText}`);
  return res.json();
}

export async function listGraphs() {
  const res = await fetch(`${BASE_URL}/graph/list`);
  if (!res.ok) throw new Error(`List failed: ${res.statusText}`);
  return res.json();
}

export async function exportGraph(nodes: any[], edges: any[], name?: string) {
  const body: { nodes: any[]; edges: any[]; name?: string } = { nodes, edges };
  if (name) body.name = name;
  const res = await apiFetch(`${BASE_URL}/graph/export`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`Export failed: ${res.statusText}`);
  return res.json() as Promise<{ script: string }>;
}

/** A2: clear persisted layer weights kept by the backend NodeStateStore.
 * Pass `node_ids` to scope the reset to specific nodes, omit to reset
 * the entire graph. Returns `{ graph_id, scope, evicted }`.
 */
export async function resetWeights(graphId: string, nodeIds?: string[]) {
  const res = await apiFetch(`${BASE_URL}/execution/state/reset`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      graph_id: graphId,
      ...(nodeIds && nodeIds.length > 0 ? { node_ids: nodeIds } : {}),
    }),
  });
  if (!res.ok) throw new Error(`Reset weights failed: ${res.statusText}`);
  return res.json() as Promise<{ graph_id: string; scope: string; evicted: number }>;
}

export async function createPreset(data: {
  name: string;
  description?: string;
  category?: string;
  tags?: string[];
  nodes: any[];
  edges: any[];
}): Promise<PresetDefinition> {
  const res = await apiFetch(`${BASE_URL}/presets/create`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? `Export failed: ${res.statusText}`);
  }
  return res.json();
}

// ── Examples ──

export interface ExampleSummary {
  name: string;
  description: string;
  category: string;
  path: string;
  node_count: number;
  edge_count: number;
}

export async function listExamples(): Promise<ExampleSummary[]> {
  const res = await fetch(`${BASE_URL}/examples/list`);
  if (!res.ok) throw new Error(`Failed to list examples: ${res.statusText}`);
  return res.json();
}

export async function loadExample(path: string) {
  const res = await fetch(`${BASE_URL}/examples/load?path=${encodeURIComponent(path)}`);
  if (!res.ok) throw new Error(`Failed to load example: ${res.statusText}`);
  return res.json();
}

export async function reloadNodes() {
  const res = await apiFetch(`${BASE_URL}/nodes/reload`, { method: 'POST' });
  if (!res.ok) throw new Error(`Reload failed: ${res.statusText}`);
  return res.json();
}

// ── Custom Node Manager ──

export interface CustomNodeInfo {
  filename: string;
  enabled: boolean;
  nodes: string[];
}

export async function listCustomNodes(): Promise<CustomNodeInfo[]> {
  const res = await fetch(`${BASE_URL}/custom-nodes`);
  if (!res.ok) throw new Error(`Failed to list custom nodes: ${res.statusText}`);
  return res.json();
}

export async function toggleCustomNode(filename: string) {
  const res = await apiFetch(`${BASE_URL}/custom-nodes/toggle`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ filename }),
  });
  if (!res.ok) throw new Error(`Toggle failed: ${res.statusText}`);
  return res.json();
}

export async function uploadCustomNode(file: File) {
  const form = new FormData();
  form.append('file', file);
  const res = await apiFetch(`${BASE_URL}/custom-nodes/upload`, {
    method: 'POST',
    body: form,
  });
  if (!res.ok) throw new Error(`Upload failed: ${res.statusText}`);
  return res.json();
}

export async function deleteCustomNode(filename: string) {
  const res = await apiFetch(`${BASE_URL}/custom-nodes/${encodeURIComponent(filename)}`, {
    method: 'DELETE',
  });
  if (!res.ok) throw new Error(`Delete failed: ${res.statusText}`);
  return res.json();
}

// ── Model Files ──

export interface ModelFileInfo {
  filename: string;
  size: number;
}

export async function listModelFiles(): Promise<ModelFileInfo[]> {
  const res = await fetch(`${BASE_URL}/models`);
  if (!res.ok) throw new Error(`Failed to list model files: ${res.statusText}`);
  return res.json();
}

export async function uploadModelFile(file: File): Promise<ModelFileInfo> {
  const form = new FormData();
  form.append('file', file);
  const res = await apiFetch(`${BASE_URL}/models/upload`, {
    method: 'POST',
    body: form,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? `Upload failed: ${res.statusText}`);
  }
  return res.json();
}

export async function deleteModelFile(filename: string) {
  const res = await apiFetch(`${BASE_URL}/models/${encodeURIComponent(filename)}`, {
    method: 'DELETE',
  });
  if (!res.ok) throw new Error(`Delete failed: ${res.statusText}`);
  return res.json();
}

export async function downloadModelFile(filename: string) {
  // Preserve slashes for nested paths (e.g. runs/exp1/model.pt)
  const urlPath = filename.split('/').map(encodeURIComponent).join('/');
  const res = await fetch(`${BASE_URL}/models/download/${urlPath}`);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? `Download failed: ${res.statusText}`);
  }
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  // Only the basename — the browser shouldn't recreate sub-directories locally
  // .pop() on a split result is always a string (possibly ''), never undefined
  /* v8 ignore start */
  a.download = filename.split('/').pop() ?? filename;
  /* v8 ignore stop */
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

// ── Image Files ──

export interface ImageFileInfo {
  filename: string;
  size: number;
}

export async function listImageFiles(): Promise<ImageFileInfo[]> {
  const res = await fetch(`${BASE_URL}/images`);
  if (!res.ok) throw new Error(`Failed to list image files: ${res.statusText}`);
  return res.json();
}

export async function uploadImageFile(file: File): Promise<ImageFileInfo> {
  const form = new FormData();
  form.append('file', file);
  const res = await apiFetch(`${BASE_URL}/images/upload`, {
    method: 'POST',
    body: form,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? `Upload failed: ${res.statusText}`);
  }
  return res.json();
}

export async function deleteImageFile(filename: string) {
  const res = await apiFetch(`${BASE_URL}/images/${encodeURIComponent(filename)}`, {
    method: 'DELETE',
  });
  if (!res.ok) throw new Error(`Delete failed: ${res.statusText}`);
  return res.json();
}

export async function downloadImageFile(filename: string) {
  const urlPath = filename.split('/').map(encodeURIComponent).join('/');
  const res = await fetch(`${BASE_URL}/images/download/${urlPath}`);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? `Download failed: ${res.statusText}`);
  }
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  // .pop() on a split result is always a string (possibly ''), never undefined
  /* v8 ignore start */
  a.download = filename.split('/').pop() ?? filename;
  /* v8 ignore stop */
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}
