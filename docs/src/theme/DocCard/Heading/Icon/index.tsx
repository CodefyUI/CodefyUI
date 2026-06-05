/**
 * Swizzled override of @theme/DocCard/Heading/Icon.
 *
 * The stock component renders an emoji (📄 / 🗃 / 🔗) as the card icon. This
 * project uses plain SVG icons instead of emoji, so we render a small inline
 * Lucide-style icon picked from the sidebar item type. Kept self-contained
 * (no @docusaurus/theme-common import) so it resolves under pnpm's strict
 * node_modules from src/theme/.
 */
import React, {type ReactNode} from 'react';
import clsx from 'clsx';
import styles from './styles.module.css';

interface IconItem {
  readonly type: string;
  readonly href?: string;
}

interface Props {
  readonly item: IconItem;
  readonly icon?: ReactNode;
}

const svgProps = {
  viewBox: '0 0 24 24',
  fill: 'none',
  stroke: 'currentColor',
  strokeWidth: 2,
  strokeLinecap: 'round' as const,
  strokeLinejoin: 'round' as const,
  'aria-hidden': true,
};

function FileIcon(): ReactNode {
  return (
    <svg {...svgProps}>
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
      <polyline points="14 2 14 8 20 8" />
      <line x1="16" y1="13" x2="8" y2="13" />
      <line x1="16" y1="17" x2="8" y2="17" />
      <line x1="10" y1="9" x2="8" y2="9" />
    </svg>
  );
}

function FolderIcon(): ReactNode {
  return (
    <svg {...svgProps}>
      <path d="M4 20h16a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.93a2 2 0 0 1-1.66-.9l-.82-1.2A2 2 0 0 0 7.93 3H4a2 2 0 0 0-2 2v13c0 1.1.9 2 2 2Z" />
    </svg>
  );
}

function ExternalLinkIcon(): ReactNode {
  return (
    <svg {...svgProps}>
      <path d="M15 3h6v6" />
      <path d="M10 14 21 3" />
      <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6" />
    </svg>
  );
}

function isExternal(href?: string): boolean {
  return !!href && /^(https?:)?\/\//.test(href);
}

function pickIcon(item: IconItem): ReactNode {
  if (item.type === 'category') {
    return <FolderIcon />;
  }
  if (isExternal(item.href)) {
    return <ExternalLinkIcon />;
  }
  return <FileIcon />;
}

export default function DocCardHeadingIcon({item}: Props): ReactNode {
  return <span className={clsx(styles.cardTitleIcon)}>{pickIcon(item)}</span>;
}
