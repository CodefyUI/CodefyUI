import { useEffect, useId, useMemo, useRef, useState, type FormEvent, type KeyboardEvent } from 'react';
import { createPortal } from 'react-dom';
import type { SweepMethod, SweepParam } from '../../api/rest';
import { useI18n } from '../../i18n';
import { useNodeDefStore } from '../../store/nodeDefStore';
import { useRunStore } from '../../store/runStore';
import { useSweepStore } from '../../store/sweepStore';
import { flushSubgraphEditing, useTabStore } from '../../store/tabStore';
import { useUIStore } from '../../store/uiStore';
import {
  buildRangeDomain,
  buildValuesDomain,
  eligibleSweepParams,
  previewSweepVariants,
  SWEEP_LIMIT_DEFAULTS,
  SweepInputError,
  type EligibleSweepParam,
  type SweepLimitWarning,
} from './sweepParams';
import { metricProducers, producerLabel, recordedMetricNames } from './sweepObjective';
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
/** The planner seed range the server accepts (`sweep_compiler.MAX_PLANNER_SEED`). */
const MAX_PLANNER_SEED = 0xFFFFFFFF;
/**
 * Objectives offered before any listed run has recorded a series: what the
 * built-in TrainingLoop logs every epoch, and with validation data.
 */
const BUILT_IN_OBJECTIVES = ['train_loss', 'val_loss', 'val_accuracy'];
let nextEditorId = 1;

interface NewSweepDialogProps {
  onClose: () => void;
}

function newEditor(address = ''): DomainEditor {
  return { id: nextEditorId++, address, mode: 'values', values: '', min: '0', max: '1', count: '2', scale: 'linear' };
}

function isNumeric(entry: EligibleSweepParam | null | undefined): boolean {
  return entry?.param.param_type === 'int' || entry?.param.param_type === 'float';
}

function parseSeed(raw: string): number {
  const value = Number(raw);
  if (raw.trim() === '' || !Number.isInteger(value) || value < 0 || value > MAX_PLANNER_SEED) {
    throw new SweepInputError('seed');
  }
  return value;
}

/**
 * The server's run-name limit (`run_service.MAX_NAME_LENGTH`), in code points.
 * Also the Name field's `maxLength`, which counts UTF-16 units, at least one
 * per code point, so typing stops at the limit or before it. An input method
 * can compose past `maxLength`, so a typed name is cut when sent as well.
 */
const MAX_RUN_NAME_LENGTH = 64;

/**
 * A run name the server stores rather than refuses (#623). The sweep route
 * REFUSES a name over the limit instead of cutting it, so a sweep's name is
 * made storable here the way the server makes a canvas run's: trimmed, cut to
 * the limit in code points (`Array.from` counts them, as the server does), and
 * half of a surrogate pair, which the database cannot store, sent as "?".
 */
function storableRunName(raw: string): string {
  return Array.from(raw.trim(), (char) => {
    const code = char.codePointAt(0) ?? 0;
    return code >= 0xd800 && code <= 0xdfff ? '?' : char;
  }).slice(0, MAX_RUN_NAME_LENGTH).join('').trimEnd();
}

/**
 * What a blank Name sends: the tab's name, as a canvas run is named (#623).
 * Checked to be text, since a tab restored from a damaged record may not
 * carry one.
 */
function tabDefaultName(tab: { name?: unknown } | null): string {
  return tab && typeof tab.name === 'string' ? storableRunName(tab.name) : '';
}

/** Compact, keyboard-contained editor for one sweep request. */
export function NewSweepDialog({ onClose }: NewSweepDialogProps) {
  const { t } = useI18n();
  const definitions = useNodeDefStore((state) => state.definitions);
  const runs = useRunStore((state) => state.runs);
  const creating = useSweepStore((state) => state.createState === 'creating');
  const serverError = useSweepStore((state) => state.createError);
  const createSweep = useSweepStore((state) => state.createSweep);
  const baseId = useId();
  const dialogRef = useRef<HTMLDivElement>(null);
  const firstRef = useRef<HTMLInputElement>(null);
  const paramSelects = useRef(new Map<number, HTMLSelectElement>());
  const focusEditor = useRef<number | null>(null);
  const [name, setName] = useState('');
  const [method, setMethod] = useState<SweepMethod>('grid');
  const [objective, setObjective] = useState('train_loss');
  // The producer node; '' selects by the name alone (#641).
  const [objectiveNode, setObjectiveNode] = useState('');
  const [direction, setDirection] = useState<'minimize' | 'maximize'>('minimize');
  const [seed, setSeed] = useState('0');
  const [samples, setSamples] = useState('1');
  const [seedVariants, setSeedVariants] = useState(false);
  const [editors, setEditors] = useState<DomainEditor[]>(() => [newEditor()]);
  const [localError, setLocalError] = useState<unknown>(null);

  const tab = useTabStore((state) =>
    state.tabs.find((candidate) => candidate.id === state.activeTabId) ?? null,
  );
  // What a blank Name sends, shown as the field's placeholder.
  const defaultName = tabDefaultName(tab);
  // The ROOT graph, flushed exactly as Start serializes it: inside a block the
  // canvas holds the block's nodes, which the server cannot address.
  const rootNodes = useMemo(() => (tab ? flushSubgraphEditing(tab).nodes : []), [tab]);
  const eligible = useMemo(
    () => eligibleSweepParams(definitions, rootNodes),
    [definitions, rootNodes],
  );
  const objectiveNames = useMemo(
    () => [...new Set([...BUILT_IN_OBJECTIVES, ...recordedMetricNames(runs)])].sort(),
    [runs],
  );
  // The nodes the listed runs saw log this metric. A name several of them
  // log ranks nothing unless one is chosen, so the dialog says so up front.
  const producers = useMemo(() => metricProducers(runs, objective.trim()), [runs, objective]);
  const producerNodes = producers.filter((producer): producer is string => producer !== null);
  const ambiguous = objectiveNode === '' && producers.length > 1;

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

  // Registered for as long as it is mounted, so the canvas shortcuts and the
  // Delete key stand down behind it however the Runs panel goes away.
  useEffect(() => {
    const { setNewSweepOpen } = useSweepStore.getState();
    setNewSweepOpen(true);
    return () => setNewSweepOpen(false);
  }, []);

  // A parameter added or removed takes the focus, so it never falls to the
  // page, where Escape and the Tab trap cannot hear it.
  useEffect(() => {
    if (focusEditor.current === null) return;
    paramSelects.current.get(focusEditor.current)?.focus();
    focusEditor.current = null;
  });

  const eligibleByAddress = useMemo(
    () => new Map(eligible.map((entry) => [addressOf(entry), entry])),
    [eligible],
  );

  const updateEditor = (id: number, patch: Partial<DomainEditor>) => {
    setEditors((current) => current.map((editor) => editor.id === id ? { ...editor, ...patch } : editor));
  };

  const addEditor = () => {
    const next = eligible.find((entry) => !usedAddresses.has(addressOf(entry)));
    if (!next) return;
    const added = newEditor(addressOf(next));
    focusEditor.current = added.id;
    setEditors((current) => [...current, added]);
  };

  const removeEditor = (id: number) => {
    const index = editors.findIndex((editor) => editor.id === id);
    const remaining = editors.filter((editor) => editor.id !== id);
    focusEditor.current = remaining[Math.min(index, remaining.length - 1)]?.id ?? null;
    setEditors(remaining);
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

  const explain = (error: unknown): string => {
    if (error instanceof SweepInputError) return t(`sweeps.error.${error.code}`, error.vars);
    return error instanceof Error ? error.message : String(error);
  };

  const describeWarning = (warning: SweepLimitWarning): string => {
    switch (warning.limit) {
      case 'runs':
        return t('sweeps.new.limitRuns', { count: warning.count, limit: SWEEP_LIMIT_DEFAULTS.runs });
      case 'params':
        return t('sweeps.new.limitParams', { count: warning.count, limit: SWEEP_LIMIT_DEFAULTS.params });
      case 'values': {
        const entry = eligibleByAddress.get(editors[warning.index]?.address ?? '');
        return t('sweeps.new.limitValues', {
          name: entry ? `${entry.nodeLabel} · ${entry.param.name}` : String(warning.index + 1),
          count: warning.count,
          limit: SWEEP_LIMIT_DEFAULTS.values,
        });
      }
    }
  };

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
    // The dialog itself holds the focus after a click between controls.
    const atStart = document.activeElement === first || document.activeElement === dialogRef.current;
    if (event.shiftKey && atStart) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    }
  };

  const seedUsed = method === 'random' || seedVariants;

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    setLocalError(null);
    try {
      const params = buildDomains();
      // Throws on input no server would take. A cap warning does not stop
      // Start: the server knows its configured caps, and says which one.
      previewSweepVariants(method, params, method === 'random' ? Number(samples) : null);
      if (!objective.trim()) throw new Error(t('sweeps.new.objectiveRequired'));
      const sweepSeed = seedUsed ? parseSeed(seed) : null;
      const store = useTabStore.getState();
      // The child runs need the session key in memory. RunService removes it
      // before persisting each child snapshot; this dialog/store keep no copy.
      const graph = store.getSerializedGraph({ keepSecrets: true });
      const active = store.getActiveTab();
      // Neither the tab's seed (the sweep owns seeding, and the server refuses
      // options.seed) nor its Record outputs: captured outputs are kept for
      // only the newest 20 runs, shared with the canvas, and the server
      // refuses record_outputs on a sweep larger than that.
      const options: Record<string, unknown> = {
        device: graph.settings?.device ?? useUIStore.getState().globalDevice,
        verbose: active.verboseMode,
        graph_id: active.graphId,
        weights_persistent: active.weightsPersistent,
        backward_mode: active.backwardMode,
        auto_backward: active.autoBackward,
        ...(active.deterministic ? { deterministic: true } : {}),
      };
      const created = await createSweep({
        base_graph: graph,
        sweep_spec: {
          method,
          seed: sweepSeed,
          samples: method === 'random' ? Number(samples) : null,
          params,
        },
        objective: {
          metric: objective.trim(),
          direction,
          ...(objectiveNode ? { node_id: objectiveNode } : {}),
        },
        options,
        name: storableRunName(name) || tabDefaultName(active) || null,
        seed_variants: seedVariants,
      }, active.id);
      if (created) onClose();
    } catch (error) {
      setLocalError(error);
    }
  };

  const usedAddresses = new Set(editors.map((editor) => editor.address));
  const runsWarned = preview?.warnings.some((warning) => warning.limit === 'runs') ?? false;
  const errorText = localError !== null ? explain(localError) : serverError;

  return createPortal(
    <div className={styles.backdrop} onMouseDown={(event) => {
      if (event.target === event.currentTarget) onClose();
    }}>
      {/* tabIndex -1: a click between controls focuses the dialog, out of the
          Tab order, rather than the page, where Escape cannot reach it. */}
      <div ref={dialogRef} className={styles.dialog} role="dialog" aria-modal="true" aria-labelledby={`${baseId}-title`} tabIndex={-1} onKeyDown={handleKeys}>
        <form onSubmit={(event) => void submit(event)}>
          <header className={styles.header}>
            <h2 id={`${baseId}-title`}>{t('sweeps.new.title')}</h2>
            <button type="button" className={styles.close} onClick={onClose} aria-label={t('sweeps.new.close')}>&#215;</button>
          </header>

          <div className={styles.body}>
            <div className={styles.pair}>
              <label>{t('sweeps.new.name')}<input ref={firstRef} value={name} placeholder={defaultName} maxLength={MAX_RUN_NAME_LENGTH} onChange={(event) => setName(event.target.value)} /></label>
              <label>{t('sweeps.new.method')}<select value={method} onChange={(event) => setMethod(event.target.value as SweepMethod)}><option value="grid">{t('sweeps.method.grid')}</option><option value="random">{t('sweeps.method.random')}</option></select></label>
            </div>
            <div className={styles.pair}>
              <label>{t('sweeps.new.objective')}<input value={objective} list={`${baseId}-objectives`} onChange={(event) => {
                const metric = event.target.value;
                setObjective(metric);
                // A node chosen for another metric is not a producer of this one.
                if (objectiveNode && !metricProducers(runs, metric.trim()).includes(objectiveNode)) setObjectiveNode('');
              }} required /></label>
              <label>{t('sweeps.new.direction')}<select value={direction} onChange={(event) => setDirection(event.target.value as 'minimize' | 'maximize')}><option value="minimize">{t('sweeps.direction.minimize')}</option><option value="maximize">{t('sweeps.direction.maximize')}</option></select></label>
            </div>
            <label>{t('sweeps.new.objectiveNode')}<select value={objectiveNode} onChange={(event) => setObjectiveNode(event.target.value)}>
              <option value="">{t('sweeps.new.objectiveAnyNode')}</option>
              {producerNodes.map((node) => <option key={node} value={node}>{producerLabel(node, rootNodes)}</option>)}
            </select></label>
            {ambiguous && <p className={styles.warning}>{t('sweeps.new.objectiveAmbiguous', { metric: objective.trim(), count: producers.length })}</p>}
            <datalist id={`${baseId}-objectives`}>{objectiveNames.map((series) => <option key={series} value={series} />)}</datalist>

            <div className={styles.domains}>
              {eligible.length === 0 ? <p className={styles.muted}>{t('sweeps.new.noParams')}</p> : editors.map((editor, editorIndex) => {
                const selected = eligibleByAddress.get(editor.address) ?? null;
                const numeric = isNumeric(selected);
                const legendId = `${baseId}-param-${editor.id}`;
                return <fieldset className={styles.domain} key={editor.id}>
                  <legend id={legendId}>{editors.length > 1 ? t('sweeps.new.parameterN', { index: editorIndex + 1 }) : t('sweeps.new.parameter')}</legend>
                  <div className={styles.domainHead}>
                    <select
                      ref={(element) => {
                        if (element) paramSelects.current.set(editor.id, element);
                        else paramSelects.current.delete(editor.id);
                      }}
                      className={styles.paramSelect}
                      aria-labelledby={legendId}
                      value={editor.address}
                      onChange={(event) => {
                        const address = event.target.value;
                        // A range is numeric only; a bool or select starts from a list.
                        updateEditor(editor.id, isNumeric(eligibleByAddress.get(address)) ? { address } : { address, mode: 'values' });
                      }}
                    >{eligible.map((entry) => {
                      const address = addressOf(entry);
                      return <option key={address} value={address} disabled={usedAddresses.has(address) && address !== editor.address}>{entry.nodeLabel} · {entry.param.name}</option>;
                    })}</select>
                    {numeric && <label className={styles.mode}>{t('sweeps.new.domain')}<select value={editor.mode} onChange={(event) => updateEditor(editor.id, { mode: event.target.value as DomainMode })}><option value="values">{t('sweeps.new.values')}</option><option value="range">{t('sweeps.new.range')}</option></select></label>}
                    {editors.length > 1 && <button type="button" className={styles.remove} onClick={() => removeEditor(editor.id)} aria-label={t('sweeps.new.removeParameter', { index: editorIndex + 1 })}>{t('sweeps.new.remove')}</button>}
                  </div>
                  {editor.mode === 'range' && numeric ? <div className={styles.rangeGrid}>
                    <label>{t('sweeps.new.minimum')}<input type="number" value={editor.min} onChange={(event) => updateEditor(editor.id, { min: event.target.value })} /></label>
                    <label>{t('sweeps.new.maximum')}<input type="number" value={editor.max} onChange={(event) => updateEditor(editor.id, { max: event.target.value })} /></label>
                    <label>{t('sweeps.new.count')}<input type="number" min="1" step="1" value={editor.count} onChange={(event) => updateEditor(editor.id, { count: event.target.value })} /></label>
                    <label>{t('sweeps.new.scale')}<select value={editor.scale} onChange={(event) => updateEditor(editor.id, { scale: event.target.value as 'linear' | 'log' })}><option value="linear">{t('sweeps.scale.linear')}</option><option value="log">{t('sweeps.scale.log')}</option></select></label>
                  </div> : <label>{t('sweeps.new.values')}<textarea value={editor.values} onChange={(event) => updateEditor(editor.id, { values: event.target.value })} rows={2} placeholder={selected?.param.param_type === 'bool' ? 'true, false' : '1, 2, 3'} /></label>}
                </fieldset>;
              })}
              {editors.length < eligible.length && <button type="button" className={styles.add} onClick={addEditor}>{t('sweeps.new.addParameter')}</button>}
            </div>

            {method === 'random' && <div className={styles.pair}>
              <label>{t('sweeps.new.samples')}<input type="number" min="1" step="1" value={samples} onChange={(event) => setSamples(event.target.value)} /></label>
            </div>}
            <label className={styles.check}><input type="checkbox" checked={seedVariants} onChange={(event) => setSeedVariants(event.target.checked)} />{t('sweeps.new.seedVariants')}</label>
            {/* Below the checkbox that can reveal it, so the checkbox does not move. */}
            {seedUsed && <div className={styles.pair}>
              <label>{t('sweeps.new.seed')}<input type="number" min="0" step="1" value={seed} onChange={(event) => setSeed(event.target.value)} /></label>
            </div>}
            {seedVariants && <p className={styles.warning}>{t('sweeps.new.seedWarning')}</p>}
            {preview?.warnings.map((warning) => <p key={`${warning.limit}-${'index' in warning ? warning.index : ''}`} className={styles.warning}>{describeWarning(warning)}</p>)}
            {preview && preview.variantCount !== null && !runsWarned && <p className={styles.preview}>{preview.variantCount === 1
              ? t('sweeps.new.previewOne')
              : t('sweeps.new.preview', { count: preview.variantCount })}</p>}
            {errorText && <div className={styles.error} role="alert">{errorText}</div>}
          </div>

          <footer className={styles.footer}>
            <button type="button" onClick={onClose}>{t('dialog.cancel')}</button>
            <button type="submit" className={styles.primary} disabled={creating || eligible.length === 0}>{creating ? t('sweeps.new.creating') : t('sweeps.new.start')}</button>
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
