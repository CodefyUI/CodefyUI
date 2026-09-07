/**
 * The KaTeX half of {@link MathText}.
 *
 * Deliberately a SEPARATE module: `MathText` reaches it through a dynamic
 * `import()`, and that import is the chunk boundary. Nothing here may be
 * imported statically from anywhere else, or KaTeX (about 260 kB) lands in
 * an eager chunk and the lazy loading is undone silently. vite.config.ts
 * fails the build if a static import ever reaches the katex package, and
 * lazyBoundaries.test.ts fails `pnpm test` if one names this module.
 *
 * The stylesheet travels with the chunk for the same reason. Vite appends it
 * to <head> AFTER index.css when the chunk loads, so the overrides are
 * imported after katex.min.css here: that is the order that lets them win
 * the cascade at equal specificity.
 */
import 'katex/dist/katex.min.css';
import './katexOverrides.css';
export { InlineMath, BlockMath } from 'react-katex';
