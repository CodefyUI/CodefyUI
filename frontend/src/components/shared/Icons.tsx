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
