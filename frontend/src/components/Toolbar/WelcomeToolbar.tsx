import { ProjectBadge } from './ProjectBadge';
import { ToolbarGlobalActions } from './ToolbarGlobalActions';
import styles from './Toolbar.module.css';

/**
 * The toolbar while no tab is open.
 *
 * The full `Toolbar` cannot stand in for it: it reads the active tab from the
 * first lines of its body -- status, device assignment, read-only flag -- and
 * there is no active tab to read. Nor would its controls mean anything if it
 * could: Run, Save, Export, Layout and the status light all name a graph.
 *
 * So what is left is what belongs to the app rather than to a graph: the
 * logo, the project badge (it names the directory the server was started in,
 * which is true with or without a tab) and the global actions -- Settings,
 * Help, font size and language. Language especially: the welcome screen is
 * the first thing a new install shows, and a user who cannot read it needs
 * that button more here than anywhere else in the app.
 */
export function WelcomeToolbar() {
  return (
    <div className={styles.root}>
      <div className={styles.logo}>
        <span className={styles.logoBrand}>Codefy</span>
        <span className={styles.logoSuffix}>UI</span>
      </div>
      <ProjectBadge />
      {/* No plugin buttons: a plugin's toolbar button acts on the graph in
          front of the user, and there is none. See the prop's own note. */}
      <ToolbarGlobalActions plugins={false} />
    </div>
  );
}
