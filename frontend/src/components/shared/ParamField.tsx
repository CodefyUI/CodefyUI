import { useCallback, useEffect, useId, useRef, useState } from 'react';
import type { ParamDefinition } from '../../types';
import {
  downloadDataFile,
  downloadImageFile,
  downloadModelFile,
  listDataFiles,
  listImageFiles,
  listModelFiles,
  uploadDataFile,
  uploadImageFile,
  uploadModelFile,
} from '../../api/rest';
import { useI18n, type TranslationKey } from '../../i18n';
import { useToastStore } from '../../store/toastStore';
import { useUIStore } from '../../store/uiStore';
import {
  itemTitle,
  missingRequirementForOption,
  packTitle,
  usePackAvailability,
  type PackIndex,
  type PackRequirement,
} from '../../utils/packAvailability';
import { TensorGridEditor } from '../ConfigPanel/TensorGridEditor';
import { ScriptCodeField } from './ScriptCodeField';
import styles from './ParamField.module.css';

interface ParamFieldProps {
  param: ParamDefinition;
  value: any;
  onChange: (name: string, value: any) => void;
  label?: string;
  /**
   * Other params on the same node — only consumed by the tensor_grid editor,
   * which needs the sibling `shape` and `value_mode` to know what to render.
   */
  siblingParams?: Record<string, any>;
}

interface FileFieldBackend {
  list: () => Promise<{ filename: string }[]>;
  upload: (file: File) => Promise<{ filename: string }>;
  download: (filename: string) => Promise<void>;
  accept: string;
  uploadTitleKey: TranslationKey;
}

const MODEL_FILE_BACKEND: FileFieldBackend = {
  list: listModelFiles,
  upload: uploadModelFile,
  download: downloadModelFile,
  accept: '.pt,.pth,.safetensors,.ckpt,.bin',
  uploadTitleKey: 'paramField.upload.model',
};

const IMAGE_FILE_BACKEND: FileFieldBackend = {
  list: listImageFiles,
  upload: uploadImageFile,
  download: downloadImageFile,
  accept: '.png,.jpg,.jpeg,.bmp,.webp,.gif,.tiff',
  uploadTitleKey: 'paramField.upload.image',
};

const DATA_FILE_BACKEND: FileFieldBackend = {
  list: listDataFiles,
  upload: uploadDataFile,
  download: downloadDataFile,
  accept: '.csv,.tsv,.txt,.json',
  uploadTitleKey: 'paramField.upload.data',
};

function FileField({
  param,
  value,
  onChange,
  displayLabel,
  backend,
}: {
  param: ParamDefinition;
  value: any;
  onChange: (name: string, value: any) => void;
  displayLabel: string;
  backend: FileFieldBackend;
}) {
  const { t } = useI18n();
  const [files, setFiles] = useState<string[]>([]);
  const [uploading, setUploading] = useState(false);
  const [downloading, setDownloading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const refresh = useCallback(() => {
    backend.list().then((list) => setFiles(list.map((f) => f.filename)));
  }, [backend]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploading(true);
    try {
      const result = await backend.upload(file);
      refresh();
      onChange(param.name, result.filename);
    } catch (err: any) {
      useToastStore.getState().addToast(err.message ?? t('paramField.uploadFailed'), 'error');
    } finally {
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  const handleDownload = async () => {
    // The download button is disabled whenever !value, so this early-return
    // guard is never reached through the UI.
    /* v8 ignore start */
    if (!value) return;
    /* v8 ignore stop */
    setDownloading(true);
    try {
      await backend.download(String(value));
    } catch (err: any) {
      useToastStore.getState().addToast(err.message ?? t('paramField.downloadFailed'), 'error');
    } finally {
      setDownloading(false);
    }
  };

  return (
    <div>
      <label className={styles.label}>{displayLabel}</label>
      <div className={styles.modelFileRow}>
        <select
          value={value ?? ''}
          onChange={(e) => onChange(param.name, e.target.value)}
          className={`${styles.input} ${styles.select} ${styles.modelFileSelect}`}
        >
          <option value="" style={{ background: 'var(--surface-input)' }}>
            {t('paramField.selectFile')}
          </option>
          {files.map((f) => (
            <option key={f} value={f} style={{ background: 'var(--surface-input)' }}>
              {f}
            </option>
          ))}
        </select>
        <button
          type="button"
          className={styles.modelFileBtn}
          onClick={() => fileInputRef.current?.click()}
          disabled={uploading}
          title={t(backend.uploadTitleKey)}
        >
          {uploading ? '...' : '↑'}
        </button>
        <button
          type="button"
          className={styles.modelFileBtn}
          onClick={handleDownload}
          disabled={!value || downloading}
          title={t('paramField.download')}
        >
          {downloading ? '...' : '↓'}
        </button>
        <button
          type="button"
          className={styles.modelFileBtn}
          onClick={refresh}
          title={t('paramField.refresh')}
        >
          ↻
        </button>
        <input
          ref={fileInputRef}
          type="file"
          accept={backend.accept}
          style={{ display: 'none' }}
          onChange={handleUpload}
        />
      </div>
    </div>
  );
}

export function ParamField({ param, value, onChange, label, siblingParams }: ParamFieldProps) {
  const displayLabel = label ?? param.name;

  if (param.param_type === 'tensor_grid') {
    return (
      <TensorGridEditor
        param={param}
        value={value}
        onChange={onChange}
        displayLabel={displayLabel}
        siblingParams={siblingParams}
      />
    );
  }

  if (param.param_type === 'model_file') {
    return (
      <FileField
        param={param}
        value={value}
        onChange={onChange}
        displayLabel={displayLabel}
        backend={MODEL_FILE_BACKEND}
      />
    );
  }

  if (param.param_type === 'image_file') {
    return (
      <FileField
        param={param}
        value={value}
        onChange={onChange}
        displayLabel={displayLabel}
        backend={IMAGE_FILE_BACKEND}
      />
    );
  }

  if (param.param_type === 'data_file') {
    return (
      <FileField
        param={param}
        value={value}
        onChange={onChange}
        displayLabel={displayLabel}
        backend={DATA_FILE_BACKEND}
      />
    );
  }

  if (param.param_type === 'bool') {
    return (
      <div className={styles.checkboxRow}>
        <input
          type="checkbox"
          id={`param-${param.name}`}
          checked={Boolean(value)}
          onChange={(e) => onChange(param.name, e.target.checked)}
          className={styles.checkbox}
        />
        <label htmlFor={`param-${param.name}`} className={styles.boolLabel}>
          {displayLabel}
        </label>
      </div>
    );
  }

  if (param.param_type === 'select') {
    return (
      <SelectField param={param} value={value} onChange={onChange} displayLabel={displayLabel} />
    );
  }

  if (param.param_type === 'int' || param.param_type === 'float') {
    const numVal = Number(value ?? param.default ?? 0);
    const hasMin = param.min_value != null;
    const hasMax = param.max_value != null;
    const outOfRange =
      !isNaN(numVal) &&
      ((hasMin && numVal < param.min_value!) || (hasMax && numVal > param.max_value!));
    const isInt = param.param_type === 'int';

    return (
      <div>
        <label className={styles.label}>{displayLabel}</label>
        <input
          type="number"
          value={value ?? param.default ?? 0}
          min={param.min_value ?? undefined}
          max={param.max_value ?? undefined}
          step={isInt ? 1 : 'any'}
          onChange={(e) =>
            onChange(param.name, isInt ? parseInt(e.target.value, 10) : parseFloat(e.target.value))
          }
          className={`${styles.input} ${outOfRange ? styles.inputError : ''}`}
        />
        {outOfRange && (hasMin || hasMax) && (
          <span className={styles.errorHint}>
            {hasMin && hasMax
              ? `Range: ${param.min_value} — ${param.max_value}`
              : hasMin
                ? `Min: ${param.min_value}`
                : `Max: ${param.max_value}`}
          </span>
        )}
      </div>
    );
  }

  if (param.param_type === 'secret') {
    return <SecretField param={param} value={value} onChange={onChange} displayLabel={displayLabel} />;
  }

  if (param.param_type === 'code') {
    return (
      <ScriptCodeField
        param={param}
        value={value}
        onChange={onChange}
        displayLabel={displayLabel}
      />
    );
  }

  // Default: string
  return (
    <div>
      <label className={styles.label}>{displayLabel}</label>
      <input
        type="text"
        value={value ?? param.default ?? ''}
        onChange={(e) => onChange(param.name, e.target.value)}
        className={styles.input}
      />
    </div>
  );
}

/**
 * SELECT param, with the optional-pack rules folded in.
 *
 * A dropdown is where a missing pack is most likely to be met: `demo-16d` and
 * `glove-50d` sit next to each other in the same list, and only one of them
 * loads on a base install. The rules, in order of how much they matter:
 *
 *  - The CURRENT value is never disabled, even when its pack is missing. A
 *    saved graph may hold it, and a `<select>` whose selected `<option>` is
 *    disabled does not merely look wrong — the browser drops the selection,
 *    which would rewrite the graph to something the user never chose the
 *    moment the config panel opened. So it stays selectable and gets a
 *    warning underneath instead.
 *  - Every OTHER unavailable option is `disabled` and carries a suffix
 *    naming what it needs, so the reason travels with the option.
 *  - An unloaded catalog or a server with no Package Center greys out
 *    nothing at all: `missingRequirementForOption` answers null for every
 *    option, and this renders exactly what it rendered before packs existed.
 */
function SelectField({
  param,
  value,
  onChange,
  displayLabel,
}: {
  param: ParamDefinition;
  value: any;
  onChange: (name: string, value: any) => void;
  displayLabel: string;
}) {
  const { t } = useI18n();
  const { byId, loaded, unsupported } = usePackAvailability();
  // Per instance, because both the config panel and the Node Detail Modal can
  // be showing the same param at once and `aria-describedby` must not point
  // at the other one's hint.
  const hintId = useId();

  const selected = value ?? param.default;
  const currentValue = selected == null ? '' : String(selected);

  const options = param.options.map((option) => ({
    option,
    missing: missingRequirementForOption(param, option, byId, loaded, unsupported),
  }));

  // Asked of the raw value rather than looked up in `options`: a graph can
  // hold a value this build no longer lists, and it still deserves its hint.
  const currentMissing = missingRequirementForOption(
    param,
    currentValue,
    byId,
    loaded,
    unsupported,
  );
  const otherMissing =
    options.find((entry) => entry.missing !== null && entry.option !== currentValue)?.missing ??
    null;

  // What the current value needs comes first: it is the thing standing
  // between this node and a run. The others are a footnote.
  const focus = currentMissing ?? otherMissing;
  const hint = currentMissing
    ? packSentence(t, byId, currentValue, currentMissing)
    : otherMissing
      ? t('paramField.packHintOthers')
      : null;

  return (
    <div>
      <label className={styles.label}>{displayLabel}</label>
      <select
        value={selected}
        onChange={(e) => onChange(param.name, e.target.value)}
        className={`${styles.input} ${styles.select}`}
        aria-describedby={hint === null ? undefined : hintId}
      >
        {options.map(({ option, missing }) => (
          <option
            key={option}
            value={option}
            disabled={missing !== null && option !== currentValue}
            style={{ background: 'var(--surface-input)' }}
          >
            {missing === null ? option : `${option} — ${optionSuffix(t, byId, missing)}`}
          </option>
        ))}
      </select>
      {hint !== null && (
        <span
          id={hintId}
          className={currentMissing ? `${styles.hint} ${styles.hintWarning}` : styles.hint}
        >
          {hint}{' '}
          <button
            type="button"
            className={styles.linkBtn}
            // `getState()` rather than a subscription: every select on the
            // canvas holds this component, and none of them re-render when
            // the Package Center opens.
            onClick={() => useUIStore.getState().openPackCenter(focus?.packId)}
          >
            {t('paramField.installPack')}
          </button>
        </span>
      )}
    </div>
  );
}

type Translate = (key: TranslationKey, vars?: Record<string, string | number>) => string;

/** The short "needs ..." tag appended to an option that cannot be chosen. */
function optionSuffix(t: Translate, byId: PackIndex, req: PackRequirement): string {
  return req.itemId === null
    ? t('paramField.needsPack', { pack: packTitle(byId, req.packId) })
    : t('paramField.needsModel', { item: itemTitle(byId, req) });
}

/** The full sentence under the select, when the CURRENT value is the problem. */
function packSentence(
  t: Translate,
  byId: PackIndex,
  option: string,
  req: PackRequirement,
): string {
  const pack = packTitle(byId, req.packId);
  return req.itemId === null
    ? t('paramField.packHint', { option, pack })
    : t('paramField.modelHint', { option, pack, item: itemTitle(byId, req) });
}

/**
 * SECRET param: a masked (password) input. The value lives only in canvas /
 * runtime state — the editor strips it from the serialized graph on save and
 * export, so it never reaches disk. The hint steers users to the environment
 * variable for anything they want to keep.
 */
function SecretField({
  param,
  value,
  onChange,
  displayLabel,
}: {
  param: ParamDefinition;
  value: any;
  onChange: (name: string, value: any) => void;
  displayLabel: string;
}) {
  const { t } = useI18n();
  return (
    <div>
      <label className={styles.label}>{displayLabel}</label>
      <input
        type="password"
        autoComplete="off"
        value={value ?? param.default ?? ''}
        onChange={(e) => onChange(param.name, e.target.value)}
        className={styles.input}
      />
      <span className={styles.hint}>{t('paramField.secretHint')}</span>
    </div>
  );
}
