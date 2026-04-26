declare module 'react-katex' {
  import type { ReactNode } from 'react';

  export interface KatexComponentProps {
    math: string;
    errorColor?: string;
    renderError?: (error: Error) => ReactNode;
    settings?: Record<string, unknown>;
    as?: string;
  }

  export const InlineMath: (props: KatexComponentProps) => ReactNode;
  export const BlockMath: (props: KatexComponentProps) => ReactNode;
}
