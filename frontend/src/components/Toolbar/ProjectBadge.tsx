import { useProjectStore } from '../../store/projectStore';
import { useI18n } from '../../i18n';

/** Editor-header badge naming the active project (nothing in non-project mode). */
export function ProjectBadge() {
  const projectName = useProjectStore((s) => s.projectName);
  const { t } = useI18n();
  if (!projectName) return null;
  return (
    <span
      title={t('project.badge.title')}
      style={{
        marginLeft: '0.5rem', padding: '0.1rem 0.5rem', fontSize: '0.75rem',
        borderRadius: '4px', background: 'rgba(6,182,212,0.15)',
        color: '#06b6d4', fontWeight: 600, whiteSpace: 'nowrap',
      }}
    >
      {projectName}
    </span>
  );
}
