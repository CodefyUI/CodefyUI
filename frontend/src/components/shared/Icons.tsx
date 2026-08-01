import type { ReactNode } from 'react';

/**
 * Minimal inline-SVG icon set (Feather-style, monochrome, themeable via
 * `currentColor`). Used in place of emoji so glyphs inherit the button's text
 * colour and hover state and render consistently across platforms.
 *
 * Each icon is decorative (`aria-hidden`); the surrounding button carries the
 * accessible label.
 */
function Icon({ children, size = 14 }: { children: ReactNode; size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={2}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
    >
      {children}
    </svg>
  );
}

export function EyeIcon({ size }: { size?: number }) {
  return (
    <Icon size={size}>
      <path d="M1 12s4-7 11-7 11 7 11 7-4 7-11 7-11-7-11-7z" />
      <circle cx={12} cy={12} r={3} />
    </Icon>
  );
}

export function EyeOffIcon({ size }: { size?: number }) {
  return (
    <Icon size={size}>
      <path d="M17.94 17.94A10.07 10.07 0 0 1 12 20C5 20 1 12 1 12a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24" />
      <line x1={1} y1={1} x2={23} y2={23} />
    </Icon>
  );
}

export function ZoomInIcon({ size }: { size?: number }) {
  return (
    <Icon size={size}>
      <circle cx={11} cy={11} r={7} />
      <line x1={21} y1={21} x2={16.65} y2={16.65} />
      <line x1={11} y1={8} x2={11} y2={14} />
      <line x1={8} y1={11} x2={14} y2={11} />
    </Icon>
  );
}

export function ZoomOutIcon({ size }: { size?: number }) {
  return (
    <Icon size={size}>
      <circle cx={11} cy={11} r={7} />
      <line x1={21} y1={21} x2={16.65} y2={16.65} />
      <line x1={8} y1={11} x2={14} y2={11} />
    </Icon>
  );
}

/** Fit / reset-to-frame (four corner brackets). */
export function FitIcon({ size }: { size?: number }) {
  return (
    <Icon size={size}>
      <path d="M8 3H5a2 2 0 0 0-2 2v3m18 0V5a2 2 0 0 0-2-2h-3m0 18h3a2 2 0 0 0 2-2v-3M3 16v3a2 2 0 0 0 2 2h3" />
    </Icon>
  );
}

/** Expand / open-larger (diagonal arrows pointing outward). */
export function ExpandIcon({ size }: { size?: number }) {
  return (
    <Icon size={size}>
      <path d="M15 3h6v6" />
      <path d="M9 21H3v-6" />
      <path d="M21 3l-7 7" />
      <path d="M3 21l7-7" />
    </Icon>
  );
}

export function CloseIcon({ size }: { size?: number }) {
  return (
    <Icon size={size}>
      <line x1={18} y1={6} x2={6} y2={18} />
      <line x1={6} y1={6} x2={18} y2={18} />
    </Icon>
  );
}

// ── Sidebar rail (#126) ──────────────────────────────────────────────────────

/** Node library (a grid of blocks). */
export function LibraryIcon({ size }: { size?: number }) {
  return (
    <Icon size={size}>
      <rect x={3} y={3} width={7} height={7} rx={1} />
      <rect x={14} y={3} width={7} height={7} rx={1} />
      <rect x={3} y={14} width={7} height={7} rx={1} />
      <rect x={14} y={14} width={7} height={7} rx={1} />
    </Icon>
  );
}

/** Presets — several nodes stacked into one reusable block. */
export function LayersIcon({ size }: { size?: number }) {
  return (
    <Icon size={size}>
      <polygon points="12 2 2 7 12 12 22 7 12 2" />
      <polyline points="2 17 12 22 22 17" />
      <polyline points="2 12 12 17 22 12" />
    </Icon>
  );
}

/** Templates / examples. */
export function BookIcon({ size }: { size?: number }) {
  return (
    <Icon size={size}>
      <path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z" />
      <path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z" />
    </Icon>
  );
}

/** Custom nodes and installed plugin packs. */
export function PackageIcon({ size }: { size?: number }) {
  return (
    <Icon size={size}>
      <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z" />
      <polyline points="3.27 6.96 12 12.01 20.73 6.96" />
      <line x1={12} y1={22.08} x2={12} y2={12} />
    </Icon>
  );
}

/** Collapse the sidebar into its rail. */
export function PanelLeftCloseIcon({ size }: { size?: number }) {
  return (
    <Icon size={size}>
      <rect x={3} y={3} width={18} height={18} rx={2} />
      <line x1={9} y1={3} x2={9} y2={21} />
      <polyline points="16 15 13 12 16 9" />
    </Icon>
  );
}

/** Restore the sidebar from its rail. */
export function PanelLeftOpenIcon({ size }: { size?: number }) {
  return (
    <Icon size={size}>
      <rect x={3} y={3} width={18} height={18} rx={2} />
      <line x1={9} y1={3} x2={9} y2={21} />
      <polyline points="14 9 17 12 14 15" />
    </Icon>
  );
}

/** Expand every category accordion. */
export function ExpandAllIcon({ size }: { size?: number }) {
  return (
    <Icon size={size}>
      <polyline points="7 6 12 11 17 6" />
      <polyline points="7 13 12 18 17 13" />
    </Icon>
  );
}

/** Collapse every category accordion. */
export function CollapseAllIcon({ size }: { size?: number }) {
  return (
    <Icon size={size}>
      <polyline points="17 11 12 6 7 11" />
      <polyline points="17 18 12 13 7 18" />
    </Icon>
  );
}

/** Re-fetch a list that came from the backend. */
export function RefreshIcon({ size }: { size?: number }) {
  return (
    <Icon size={size}>
      <polyline points="23 4 23 10 17 10" />
      <polyline points="1 20 1 14 7 14" />
      <path d="M20.49 9A9 9 0 0 0 5.64 5.64L1 10m22 4l-4.64 4.36A9 9 0 0 1 3.51 15" />
    </Icon>
  );
}
