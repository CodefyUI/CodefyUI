/**
 * CodefyUI plugin SDK — types + React bindings.
 *
 * This `sdk/` folder is vendored into your plugin (clone-and-own). It mirrors
 * the host's plugin API; keep it in sync with the CodefyUI release you target.
 * Import everything from here:
 *
 * ```tsx
 * import { defineTool, useGraph, type CodefyUIPluginAPI } from './sdk';
 * ```
 */
export * from './types';
export * from './react';
