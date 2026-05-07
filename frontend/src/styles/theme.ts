/** Shared design tokens — single source of truth for colors used across components. */

export const CATEGORY_COLORS: Record<string, string> = {
  CNN: '#4CAF50',
  RNN: '#2196F3',
  Transformer: '#9C27B0',
  LLM: '#A78BFA',
  Diffusion: '#EC4899',
  RL: '#FF9800',
  Data: '#00BCD4',
  Training: '#F44336',
  IO: '#795548',
  'Data Flow': '#FF6F00',
  Utility: '#607D8B',
  Normalization: '#26A69A',
  'Tensor Operations': '#5C6BC0',
};

/**
 * Token chip palette — soft pastels at ~60% saturation, tuned for the dark IDE
 * surface. Cycled by token index in the tokenizer visualization. Adjacent chips
 * remain distinguishable without jarring against #121212.
 */
export const TOKEN_COLORS: readonly string[] = [
  '#7DD3FC', // sky
  '#FCA5A5', // rose
  '#86EFAC', // mint
  '#FCD34D', // amber
  '#C4B5FD', // lavender
  '#FDA4AF', // pink
  '#6EE7B7', // emerald
  '#FDBA74', // peach
  '#A5B4FC', // indigo
  '#67E8F9', // cyan
  '#F0ABFC', // fuchsia
  '#BEF264', // lime
] as const;

export function getTokenColor(index: number): string {
  return TOKEN_COLORS[index % TOKEN_COLORS.length];
}

export const DIFFICULTY_COLORS: Record<string, string> = {
  beginner: '#4CAF50',
  intermediate: '#FF9800',
  advanced: '#F44336',
};

export const STATUS_COLORS: Record<string, string> = {
  running: '#FFC107',
  completed: '#4CAF50',
  error: '#F44336',
  cached: '#2196F3',
  skipped: '#9E9E9E',
  idle: '#444',
};

export const SURFACE = {
  bg: '#121212',
  panel: '#161616',
  card: '#1e1e1e',
  toolbar: '#1a1a1a',
  input: '#222',
  hover: '#2a2a2a',
  border: '#2a2a2a',
  borderLight: '#333',
  borderMedium: '#444',
  borderHeavy: '#555',
} as const;

export const TEXT = {
  primary: '#eee',
  secondary: '#ccc',
  tertiary: '#888',
  muted: '#666',
  dim: '#555',
  dimmer: '#444',
} as const;

export const BRAND = {
  primary: '#06b6d4',
  primaryHover: '#22d3ee',
  primaryDim: '#0e7490',
  primaryGlow: 'rgba(6, 182, 212, 0.5)',
  primaryBg: 'rgba(6, 182, 212, 0.09)',
  preset: '#D4A017',
  success: '#4CAF50',
  error: '#F44336',
  warning: '#FFC107',
} as const;

/**
 * Crafted toolbar tokens — surface gradients and micro-shadows used by the
 * Toolbar and Settings popover. Keep these synchronised with the design memory.
 */
export const TOOLBAR = {
  bgGradient: 'linear-gradient(180deg, #0e1520 0%, #0a0f17 100%)',
  border: '#1a2230',
  innerHighlight: 'inset 0 1px 0 rgba(255,255,255,0.04)',
  separator: '#1f2937',
  ctrlBorder: '#334155',
  popoverBg: '#0a0f17',
  popoverBorder: '#1f2937',
  popoverShadow: '0 24px 60px -12px rgba(0,0,0,0.8), 0 0 0 1px rgba(6,182,212,0.05)',
  // Run button accent reflection
  runShadow: 'inset 0 1px 0 rgba(255,255,255,0.22), 0 6px 14px -6px rgba(6,182,212,0.5)',
} as const;
