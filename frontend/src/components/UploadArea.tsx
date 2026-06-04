import { useRef, useState } from 'react';
import type { ChangeEvent, DragEvent, KeyboardEvent } from 'react';
import type { UploadResponse } from '../api';
import { Spinner } from './Spinner';
import './UploadArea.css';

interface UploadAreaProps {
  loading: boolean;
  result: UploadResponse | null;
  error: string | null;
  onUpload: (files: File[]) => void;
}

const ALLOWED_EXTENSIONS = ['.md', '.txt', '.pdf'] as const;
const ALLOWED_LABEL = '.md, .txt e .pdf';

function filterAllowed(files: FileList | null): File[] {
  if (!files) return [];
  const allowed: File[] = [];
  for (const file of Array.from(files)) {
    const lowered = file.name.toLowerCase();
    if (ALLOWED_EXTENSIONS.some((ext) => lowered.endsWith(ext))) {
      allowed.push(file);
    }
  }
  return allowed;
}

export function UploadArea({
  loading,
  result,
  error,
  onUpload,
}: UploadAreaProps) {
  const [dragging, setDragging] = useState<boolean>(false);
  const [localError, setLocalError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const submitFiles = (incoming: FileList | null): void => {
    const allowed = filterAllowed(incoming);
    if (allowed.length === 0) {
      setLocalError(
        `Nenhum arquivo ${ALLOWED_LABEL} encontrado. Apenas esses tipos são aceitos.`,
      );
      return;
    }
    setLocalError(null);
    onUpload(allowed);
  };

  const handleDragOver = (e: DragEvent<HTMLDivElement>): void => {
    e.preventDefault();
    e.stopPropagation();
    if (!loading) setDragging(true);
  };

  const handleDragLeave = (e: DragEvent<HTMLDivElement>): void => {
    e.preventDefault();
    e.stopPropagation();
    setDragging(false);
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>): void => {
    e.preventDefault();
    e.stopPropagation();
    setDragging(false);
    if (loading) return;
    submitFiles(e.dataTransfer.files);
  };

  const handleClick = (): void => {
    if (!loading) fileInputRef.current?.click();
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLDivElement>): void => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      handleClick();
    }
  };

  const handleFileInput = (e: ChangeEvent<HTMLInputElement>): void => {
    submitFiles(e.target.files);
    // Reset so re-selecting the same file re-fires onChange.
    e.target.value = '';
  };

  const errorMessage = localError ?? error;

  return (
    <section className="upload-area-section">
      <header className="upload-area-header">
        <h2 className="upload-area-title">Documentos</h2>
        <p className="upload-area-subtitle">
          Solte arquivos para o agente indexar. Aceita {ALLOWED_LABEL}.
        </p>
      </header>

      <div
        className={[
          'upload-zone',
          dragging ? 'is-dragging' : '',
          loading ? 'is-loading' : '',
        ]
          .filter(Boolean)
          .join(' ')}
        onDragOver={handleDragOver}
        onDragEnter={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={handleClick}
        onKeyDown={handleKeyDown}
        role="button"
        tabIndex={loading ? -1 : 0}
        aria-disabled={loading}
        aria-label="Área de upload de documentos. Solte ou clique para selecionar."
      >
        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept=".md,.txt,.pdf"
          onChange={handleFileInput}
          className="upload-input-hidden"
          tabIndex={-1}
        />
        {loading ? (
          <Spinner message="Subindo e ingerindo arquivos..." />
        ) : (
          <>
            <div className="upload-icon" aria-hidden="true">
              ↑
            </div>
            <p className="upload-primary">
              Solte arquivos aqui ou{' '}
              <span className="upload-link">clique para selecionar</span>
            </p>
            <p className="upload-hint">
              Apenas {ALLOWED_LABEL} · múltiplos arquivos permitidos
            </p>
          </>
        )}
      </div>

      {result && (
        <div className="upload-result">
          <div className="upload-result-header">
            <span className="upload-result-check" aria-hidden="true">
              ✓
            </span>
            Ingestão concluída ·{' '}
            <strong>{result.total_chunks_in_collection}</strong> chunks na
            coleção
          </div>
          {result.files.length > 0 && (
            <ul className="upload-result-list">
              {result.files.map((f) => (
                <li key={f.filename} className="upload-result-item">
                  <span className="upload-file-name">{f.filename}</span>
                  <span className="upload-file-chunks">
                    +{f.chunks} chunks
                  </span>
                </li>
              ))}
            </ul>
          )}
          {result.skipped.length > 0 && (
            <div className="upload-skipped-block">
              <div className="upload-skipped-header">
                <span className="upload-skipped-icon" aria-hidden="true">
                  !
                </span>
                {result.skipped.length === 1
                  ? '1 arquivo pulado'
                  : `${result.skipped.length} arquivos pulados`}
              </div>
              <ul className="upload-skipped-list">
                {result.skipped.map((f) => (
                  <li key={f.filename} className="upload-skipped-item">
                    <span className="upload-file-name">{f.filename}</span>
                    <span className="upload-skipped-reason">{f.reason}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}

      {errorMessage !== null && (
        <div className="upload-error" role="alert">
          {errorMessage}
        </div>
      )}
    </section>
  );
}
