/**
 * Modifier-key label for anything that shows a keyboard shortcut to the user
 * ("Cmd+B" on a Mac, "Ctrl+B" everywhere else).
 *
 * Read once at module load: the platform cannot change during a session, and
 * a single evaluation keeps the shortcuts list and the sidebar tooltips from
 * ever disagreeing.
 */
export const MOD_LABEL = navigator.platform.toUpperCase().includes('MAC') ? 'Cmd' : 'Ctrl';
