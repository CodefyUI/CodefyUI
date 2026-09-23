// Types for act-warnings.mjs, which vitest.config.ts imports.
import type { Plugin } from 'vite';

/** Absolute path of act-warnings.baseline.json. */
export declare const BASELINE_PATH: string;

/** The variable act-warnings-update.mjs sets: "lower", or "accept" for --allow-increase. */
export declare const MODE_ENV: string;

/** The act() warnings gate, for `plugins` in vitest.config.ts. */
export declare function actWarningsGate(): Plugin;
