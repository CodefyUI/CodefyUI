import { useEffect, useMemo, useRef, useState, type FormEvent, type KeyboardEvent } from 'react';
import { createPortal } from 'react-dom';
import type { SweepMethod, SweepParam } from '../../api/rest';
import { useI18n } from '../../i18n';
import { useNodeDefStore } from '../../store/nodeDefStore';
import { useSweepStore } from '../../store/sweepStore';
import { useTabStore } from '../../store/tabStore';
import { useUIStore } from '../../store/uiStore';
import {
  buildRangeDomain,
  buildValuesDomain,
  DEFAULT_SWEEP_RUN_LIMIT,
  eligibleSweepParams,
  previewSweepVariants,
  type EligibleSweepParam,
} from './sweepParams';
import styles from './NewSweepDialog.module.css';

type DomainMode = 'values' | 'range';

interface DomainEditor {
  id: number;
  address: string;
  mode: DomainMode;
  values: string;
  min: string;
  max: string;
  count: string;
  scale: 'linear' | 'log';
}

const FOCUSABLE = 'button:not(:disabled), input:not(:disabled), select:not(:disabled), textarea:not(:disabled), [tabindex]:not([tabindex="-1"])';
let nextEditorId = 1;

interface NewSweepDialogProps {
  onClose: () => void;
}

function newEditor(address = ''): DomainEditor {
  return { id: nextEditorId++, address, mode: 'values', values: '', min: '0', max: '1', count: '2', scale: 'linear' };
}

/** Compact, keyboard-contained editor for one sweep request. */
export function NewSweepDialog({ onClose }: NewSweepDialogProps) {
  const { t } = useI18n();
  const definitions = useNodeDefStore((state) => state.definitions);
  const creating = useSweepStore((state) => state.createState === 'creating');
  const serverError = useSweepStore((state) => state.error);
  const createSweep = useSweepStore((state) => state.createSweep);
  const dialogRef = useRef<HTMLDivElement>(null);
  const firstRef = useRef<HTMLInputElement>(null);
  const [name, setName] = useState('');
  const [method, setMethod] = useState<SweepMethod>('grid');
  const [objective, setObjective] = useState('loss');
  const [direction, setDirection] = useState<'minimize' | 'maximize'>('minimize');
  const [plannerSeed, setPlannerSeed] = useState('0');
  const [samples, setSamples] = useState('1');
  const [seedVariants, setSeedVariants] = useState(false);
  const [editors, setEditors] = useState<DomainEditor[]>(() => [newEditor()]);
  const [localError, setLocalError] = useState<string | null>(null);

  const tab = useTabStore((state) =>
    state.tabs.find((candidate) => candidate.id === state.activeTabId) ?? null,
  );
  const eligible = useMemo(
    () => eligibleSweepParams(definitions, tab?.nodes ?? []),
    [definitions, tab?.nodes],
  );

  useEffect(() => {
    if (!eligible[0]) return;
    const firstAddress = addressOf(eligible[0]);
    setEditors((current) => current.some((editor) => editor.address === '')
      ? current.map((editor) => editor.address === '' ? { ...editor, address: firstAddress } : editor)
      : current);
  }, [eligible]);

  useEffect(() => {
    const previous = document.activeElement as HTMLElement | null;
    firstRef.current?.focus();
    return () => {
      if (previous?.isConnected) previous.focus();
    };
  }, []);

  const eligibleByAddress = useMemo(
    () => new Map(eligible.map((entry) => [addressOf(entry), entry])),
    [eligible],
  );

  const updateEditor = (id: number, patch: Partial<DomainEditor>) => {
    setEditors((current) => current.map((editor) => editor.id === id ? { ...editor, ...patch } : editor));
  };

  const buildDomains = (): SweepParam[] => editors.map((editor) => {
    const selected = eligibleByAddress.get(editor.address);
    if (!selected) throw new Error(t('sweeps.new.noParams'));
    if (editor.mode === 'range') {
      return buildRangeDomain(selected, {
        min: Number(editor.min), max: Number(editor.max), count: Number(editor.count), scale: editor.scale,
      });
    }
    return buildValuesDomain(selected, editor.values);
  });

  const preview = useMemo(() => {
    try {
      return previewSweepVariants(method, buildDomains(), method === 'random' ? Number(samples) : null);
    } catch {
      return null;
    }
    // The builder is local; editor state and lookup identity are its inputs.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [editors, eligibleByAddress, method, samples]);

  const handleKeys = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key === 'Escape') {
      event.preventDefault();
      onClose();
      return;
    }
    if (event.key !== 'Tab') return;
    const focusable = Array.from(dialogRef.current?.querySelectorAll<HTMLElement>(FOCUSABLE) ?? []);
    if (focusable.length === 0) return;
    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    }
  };

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    setLocalError(null);
    try {
      const params = buildDomains();
      const submittedPreview = previewSweepVariants(
        method,
        params,
        method === 'random' ? Number(samples) : null,
      );
      if (submittedPreview.exceedsCap) {
        throw new Error(t('sweeps.new.capExceeded', {
          count: submittedPreview.variantCount,
          limit: DEFAULT_SWEEP_RUN_LIMIT,
        }));
      }
      if (!objective.trim()) throw new Error(t('sweeps.new.objectiveRequired'));
      const store = useTabStore.getState();
      // The child runs need the session key in memory. RunService removes it
      // before persisting each child snapshot; this dialog/store keep no copy.
      const graph = store.getSerializedGraph({ keepSecrets: true });
      const active = store.getActiveTab();
      const options: Record<string, unknown> = {
        device: graph.settings?.device ?? useUIStore.getState().globalDevice,
        record_outputs: active.recordOutputs,
        verbose: active.verboseMode,
        graph_id: active.graphId,
        weights_persistent: active.weightsPersistent,
        backward_mode: active.backwardMode,
        auto_backward: active.autoBackward,
        ...(active.deterministic ? { deterministic: true } : {}),
      };
      await createSweep({
        base_graph: graph,
        sweep_spec: {
          method,
          seed: method === 'random' || seedVariants ? Number(plannerSeed) : null,
          samples: method === 'random' ? Number(samples) : null,
          params,
        },
        objective: { metric: objective.trim(), direction },
        options,
        name: name.trim() || null,
        seed_variants: seedVariants,
      });
      if (useSweepStore.getState().error === null) onClose();
    } catch (error) {
      setLocalError(error instanceof Error ? error.message : String(error));
    }
  };

  const usedAddresses = new Set(editors.map((editor) => editor.address));
  const error = localError ?? serverError;

  return createPortal(
    <div className={styles.backdrop} onMouseDown={(event) => {
      if (event.target === event.currentTarget) onClose();
    }}>
      <div ref={dialogRef} className={styles.dialog} role="dialog" aria-modal="true" aria-labelledby="new-sweep-title" onKeyDown={handleKeys}>
        <form onSubmit={(event) => void submit(event)}>
          <header className={styles.header}>
            <h2 id="new-sweep-title">{t('sweeps.new.title')}</h2>
            <button type="button" className={styles.close} onClick={onClose} aria-label={t('sweeps.new.close')}>&#215;</button>
          </header>

          <div className={styles.body}>
            <div className={styles.pair}>
              <label>{t('sweeps.new.name')}<input ref={firstRef} value={name} onChange={(event) => setName(event.target.value)} /></label>
              <label>{t('sweeps.new.method')}<select value={method} onChange={(event) => setMethod(event.target.value as SweepMethod)}><option value="grid">{t('sweeps.method.grid')}</option><option value="random">{t('sweeps.method.random')}</option></select></label>
            </div>
            <div className={styles.pair}>
              <label>{t('sweeps.new.objective')}<input value={objective} onChange={(event) => setObjective(event.target.value)} required /></label>
              <label>{t('sweeps.new.direction')}<select value={direction} onChange={(event) => setDirection(event.target.value as 'minimize' | 'maximize')}><option value="minimize">{t('sweeps.direction.minimize')}</option><option value="maximize">{t('sweeps.direction.maximize')}</option></select></label>
            </div>

            <div className={styles.domains}>
              {eligible.length === 0 ? <p className={styles.muted}>{t('sweeps.new.noParams')}</p> : editors.map((editor, editorIndex) => {
                const selected = eligibleByAddress.get(editor.address) ?? null;
                const numeric = selected?.param.param_type === 'int' || selected?.param.param_type === 'float';
                const suffix = editors.length > 1 ? ` ${editorIndex + 1}` : '';
                return <fieldset className={styles.domain} key={editor.id}>
                  <legend>{t('sweeps.new.parameter')}{suffix}</legend>
                  <div className={styles.domainHead}>
                    <label>{t('sweeps.new.parameter')}{suffix}<select value={editor.address} onChange={(event) => updateEditor(editor.id, { address: event.target.value })}>{eligible.map((entry) => {
                      const address = addressOf(entry);
                      return <option key={address} value={address} disabled={usedAddresses.has(address) && address !== editor.address}>{entry.nodeLabel} · {entry.param.name}</option>;
                    })}</select></label>
                    {numeric && <label>{t('sweeps.new.domain')}<select value={editor.mode} onChange={(event) => updateEditor(editor.id, { mode: event.target.value as DomainMode })}><option value="values">{t('sweeps.new.values')}</option><option value="range">{t('sweeps.new.range')}</option></select></label>}
                    {editors.length > 1 && <button type="button" className={styles.remove} onClick={() => setEditors((current) => current.filter((candidate) => candidate.id !== editor.id))} aria-label={t('sweeps.new.removeParameter', { index: editorIndex + 1 })}>{t('sweeps.new.remove')}</button>}
                  </div>
                  {editor.mode === 'range' && numeric ? <div className={styles.rangeGrid}>
                    <label>{t('sweeps.new.minimum')}<input type="number" value={editor.min} onChange={(event) => updateEditor(editor.id, { min: event.target.value })} /></label>
                    <label>{t('sweeps.new.maximum')}<input type="number" value={editor.max} onChange={(event) => updateEditor(editor.id, { max: event.target.value })} /></label>
                    <label>{t('sweeps.new.count')}<input type="number" min="1" step="1" value={editor.count} onChange={(event) => updateEditor(editor.id, { count: event.target.value })} /></label>
                    <label>{t('sweeps.new.scale')}<select value={editor.scale} onChange={(event) => updateEditor(editor.id, { scale: event.target.value as 'linear' | 'log' })}><option value="linear">{t('sweeps.scale.linear')}</option><option value="log">{t('sweeps.scale.log')}</option></select></label>
                  </div> : <label>{t('sweeps.new.values')}{suffix}<textarea value={editor.values} onChange={(event) => updateEditor(editor.id, { values: event.target.value })} rows={2} placeholder={selected?.param.param_type === 'bool' ? 'true, false' : '1, 2, 3'} /></label>}
                </fieldset>;
              })}
              {editors.length < eligible.length && <button type="button" className={styles.add} onClick={() => {
                const next = eligible.find((entry) => !usedAddresses.has(addressOf(entry)));
                if (next) setEditors((current) => [...current, newEditor(addressOf(next))]);
              }}>{t('sweeps.new.addParameter')}</button>}
            </div>

            {method === 'random' && <div className={styles.pair}>
              <label>{t('sweeps.new.samples')}<input type="number" min="1" step="1" value={samples} onChange={(event) => setSamples(event.target.value)} /></label>
              <label>{t('sweeps.new.plannerSeed')}<input type="number" min="0" step="1" value={plannerSeed} onChange={(event) => setPlannerSeed(event.target.value)} /></label>
            </div>}
            <label className={styles.check}><input type="checkbox" checked={seedVariants} onChange={(event) => setSeedVariants(event.target.checked)} />{t('sweeps.new.seedVariants')}</label>
            {seedVariants && <p className={styles.warning}>{t('sweeps.new.seedWarning')}</p>}
            {preview && (preview.exceedsCap
              ? <p className={styles.error} role="alert">{t('sweeps.new.capExceeded', { count: preview.variantCount, limit: DEFAULT_SWEEP_RUN_LIMIT })}</p>
              : <p className={styles.preview}>{t('sweeps.new.preview', { count: preview.variantCount })}</p>)}
            {error && <div className={styles.error} role="alert">{error}</div>}
          </div>

          <footer className={styles.footer}>
            <button type="button" onClick={onClose}>{t('dialog.cancel')}</button>
            <button type="submit" className={styles.primary} disabled={creating || eligible.length === 0 || preview?.exceedsCap === true}>{creating ? t('sweeps.new.creating') : t('sweeps.new.start')}</button>
          </footer>
        </form>
      </div>
    </div>,
    document.body,
  );
}

function addressOf(entry: EligibleSweepParam): string {
  return `${entry.nodeId}.${entry.param.name}`;
}
