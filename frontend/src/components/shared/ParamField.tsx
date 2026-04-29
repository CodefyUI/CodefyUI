import { useCallback, useEffect, useRef, useState } from 'react';
import type { ParamDefinition } from '../../types';
import {
  downloadImageFile,
  downloadModelFile,
  listImageFiles,
  listModelFiles,
  uploadImageFile,
  uploadModelFile,
} from '../../api/rest';
import { useI18n, type TranslationKey } from '../../i18n';
import { useToastStore } from '../../store/toastStore';
import { TensorGridEditor } from '../ConfigPanel/TensorGridEditor';
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
    if (!value) return;
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
          <option value="" style={{ background: '#222' }}>
            {t('paramField.selectFile')}
          </option>
          {files.map((f) => (
            <option key={f} value={f} style={{ background: '#222' }}>
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
      <div>
        <label className={styles.label}>{displayLabel}</label>
        <select
          value={value ?? param.default}
          onChange={(e) => onChange(param.name, e.target.value)}
          className={`${styles.input} ${styles.select}`}
        >
          {param.options.map((opt) => (
            <option key={opt} value={opt} style={{ background: '#222' }}>
              {opt}
            </option>
          ))}
        </select>
      </div>
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
