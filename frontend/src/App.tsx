import { useState } from 'react';
import {
  askQuestion,
  evaluateAnswer,
  EvaluatorUnavailableError,
  uploadDocuments,
  type AskResponse,
  type UploadResponse,
} from './api';
import { AnswerBox } from './components/AnswerBox';
import {
  MetricsPanel,
  type MetricsState,
} from './components/MetricsPanel';
import { QuestionForm } from './components/QuestionForm';
import { SourceList } from './components/SourceList';
import { UploadArea } from './components/UploadArea';
import './App.css';

function errorMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

function App() {
  const [question, setQuestion] = useState<string>('');

  // Upload lifecycle
  const [uploading, setUploading] = useState<boolean>(false);
  const [uploadResult, setUploadResult] = useState<UploadResponse | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);

  // Ask lifecycle
  const [asking, setAsking] = useState<boolean>(false);
  const [askResult, setAskResult] = useState<AskResponse | null>(null);
  const [askError, setAskError] = useState<string | null>(null);

  // Evaluate lifecycle — a single discriminated state instead of three
  // related-but-independent booleans / nullables. The MetricsPanel
  // branches on `kind` and cannot get into an inconsistent shape.
  const [metricsState, setMetricsState] = useState<MetricsState>({
    kind: 'idle',
  });

  const handleUpload = async (files: File[]): Promise<void> => {
    setUploading(true);
    setUploadError(null);
    try {
      const result = await uploadDocuments(files);
      setUploadResult(result);
    } catch (err) {
      setUploadError(errorMessage(err));
      setUploadResult(null);
    } finally {
      setUploading(false);
    }
  };

  const handleAsk = async (): Promise<void> => {
    setAsking(true);
    setAskError(null);
    setAskResult(null);
    // Any prior evaluation is about a different answer; reset so the
    // panel doesn't show last round's gauges next to a brand-new answer.
    setMetricsState({ kind: 'idle' });
    try {
      const result = await askQuestion({ question: question.trim() });
      setAskResult(result);
    } catch (err) {
      setAskError(errorMessage(err));
    } finally {
      setAsking(false);
    }
  };

  const handleEvaluate = async (): Promise<void> => {
    if (askResult === null) return;
    setMetricsState({ kind: 'loading' });
    try {
      const result = await evaluateAnswer({
        question: question.trim(),
        answer: askResult.answer,
        contexts: askResult.sources.map((s) => s.text),
      });
      setMetricsState({ kind: 'success', result });
    } catch (err) {
      // 503 from the backend = optional dependency missing. Distinct
      // user-facing treatment from a real error.
      if (err instanceof EvaluatorUnavailableError) {
        setMetricsState({ kind: 'unavailable' });
        return;
      }
      setMetricsState({ kind: 'error', message: errorMessage(err) });
    }
  };

  return (
    <div className="app-shell">
      <header className="app-header">
        <h1 className="app-title">RAG System</h1>
        <p className="app-tagline">
          Ingerir documentos, perguntar com fontes citadas, e medir a
          qualidade do que o agente devolveu.
        </p>
      </header>

      <main className="app-main">
        <UploadArea
          loading={uploading}
          result={uploadResult}
          error={uploadError}
          onUpload={handleUpload}
        />

        <QuestionForm
          question={question}
          onQuestionChange={setQuestion}
          disabled={asking}
          onSubmit={handleAsk}
        />

        <AnswerBox
          answer={askResult?.answer ?? null}
          error={askError}
          loading={asking}
        />

        {askResult !== null && askResult.sources.length > 0 && (
          <SourceList sources={askResult.sources} />
        )}

        {askResult !== null && (
          <MetricsPanel state={metricsState} onEvaluate={handleEvaluate} />
        )}
      </main>

      <footer className="app-footer">
        <a
          href="https://github.com/raulmn00/rag-system"
          target="_blank"
          rel="noreferrer"
        >
          github.com/raulmn00/rag-system
        </a>
      </footer>
    </div>
  );
}

export default App;
