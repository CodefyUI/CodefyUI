import { Component, type ReactNode } from 'react';
import styles from './NodeDetailModal.module.css';

interface Props {
  /** Changing this remounts the boundary, clearing a previous tab's error. */
  resetKey: string;
  fallback: (error: Error) => ReactNode;
  children: ReactNode;
}

interface State {
  error: Error | null;
}

/**
 * Catches a throwing detail tab so it takes down the panel, not the app.
 *
 * The tab registry is an extension point (#129 fills Stats, #131 adds a code
 * editor, and plugins may register their own): third-party render code runs
 * inside the modal, and React unmounts the whole tree on an uncaught render
 * error. Without this boundary, one bad tab blanks the entire editor and
 * takes the user's unsaved graph off screen with it.
 *
 * Deliberately narrow — it wraps only the tab body, so the header, the
 * parameter form and the tab strip stay usable and the user can switch away
 * from the broken tab.
 */
export class TabErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidUpdate(prevProps: Props) {
    // A different tab (or node) is being shown — give it a clean slate rather
    // than leaving the previous failure on screen forever.
    if (prevProps.resetKey !== this.props.resetKey && this.state.error !== null) {
      this.setState({ error: null });
    }
  }

  render() {
    if (this.state.error !== null) {
      return <div className={styles.tabBody}>{this.props.fallback(this.state.error)}</div>;
    }
    return this.props.children;
  }
}
