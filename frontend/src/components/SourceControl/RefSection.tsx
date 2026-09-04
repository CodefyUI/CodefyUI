import type { ReactNode } from 'react';
import type { GitRefKind } from '../../store/gitStore';
import { ChevronDownIcon } from '../shared/Icons';
import styles from './SourceControl.module.css';

/**
 * The DOM ids one reference section uses.
 *
 * Fixed strings rather than `useId`, because the header's branch button
 * expands the Branches section from OUTSIDE it: `aria-controls` has to name a
 * list drawn by a component the button cannot see, and the same button scrolls
 * that section's heading into view. The panel is a singleton -- the sidebar
 * mounts exactly one open tab, and each kind appears once in it -- so a fixed
 * id per kind cannot collide with itself.
 */
export function refSectionIds(kind: GitRefKind): {
  sectionId: string;
  headingId: string;
  listId: string;
} {
  return {
    sectionId: `scm-section-${kind}`,
    headingId: `scm-section-${kind}-heading`,
    listId: `scm-section-${kind}-list`,
  };
}

export interface RefSectionProps {
  kind: GitRefKind;
  /** The section's name, already translated. */
  title: string;
  /**
   * How many rows it holds -- shown whether it is open or not, and `null`
   * while the list has not been read.
   *
   * The three sections are closed on a fresh profile and nothing reads a list
   * until one is opened, so `?? 0` here printed "Branches 0" beside a
   * repository with five of them and kept printing it until somebody expanded
   * the section it was wrong about. No number is what "not read yet" looks
   * like; the first read fills it in.
   */
  count: number | null;
  open: boolean;
  /** The open state lives in the store, which persists it; see `setSectionOpen`. */
  onOpenChange: (open: boolean) => void;
  /** The section-level buttons, revealed by a hover or a focus in the header. */
  actions?: ReactNode;
  /** The `<li>` rows. */
  children?: ReactNode;
  sectionId?: string;
  listId?: string;
}

/**
 * One collapsible list of git references, with the actions that apply to all
 * of them.
 *
 * The same disclosure shape as `ChangeGroup` -- a labelled `<section>`, a
 * header button carrying `aria-expanded`, a `<ul role="list">` it controls --
 * so the five sections in the tab read as one kind of thing. The difference is
 * where the open state lives: a change group opens and closes for the life of
 * the panel, while these three are remembered across reloads, so this one is
 * controlled and the store is what it asks.
 *
 * The list is `hidden` rather than unmounted, which is what lets a control
 * outside the section point `aria-controls` at an element that is always
 * there.
 */
export function RefSection({
  kind,
  title,
  count,
  open,
  onOpenChange,
  actions,
  children,
  sectionId,
  listId,
}: RefSectionProps) {
  const ids = refSectionIds(kind);
  const section = sectionId ?? ids.sectionId;
  // The heading id is a function of the KIND and nothing else. The header's
  // branch button scrolls to `refSectionIds('branches').headingId` from
  // outside this component, so a heading that moved with an overridden
  // `sectionId` would take that scroll target with it, silently.
  const headingId = ids.headingId;
  const list = listId ?? ids.listId;

  return (
    <section
      id={section}
      className={styles.group}
      data-section={kind}
      aria-labelledby={headingId}
    >
      <div className={styles.groupHeader}>
        <button
          type="button"
          id={headingId}
          className={styles.groupToggle}
          aria-expanded={open}
          aria-controls={list}
          // The heading is the first thing to lose room at a 180px panel width,
          // and it loses more of it the moment a hover opens the actions beside
          // it -- so the name in full stays in a `title`.
          title={title}
          onClick={() => onOpenChange(!open)}
        >
          <span className={`${styles.chevron} ${open ? '' : styles.chevronCollapsed}`}>
            <ChevronDownIcon size={12} />
          </span>
          <span className={styles.groupTitle}>{title}</span>
        </button>
        {/* Title, then the actions, then the count -- the count LAST, pinned to
            the same edge in every section however many actions that one has.
            See the same note in `ChangeGroup`. */}
        <div className={styles.groupActions}>{actions}</div>
        {count !== null && <span className={styles.groupCount}>{count}</span>}
      </div>
      {/* `role="list"` is spelled out because `list-style: none` takes the list
          semantics away from a `<ul>` in Safari. */}
      <ul id={list} className={styles.list} role="list" hidden={!open}>
        {children}
      </ul>
    </section>
  );
}
