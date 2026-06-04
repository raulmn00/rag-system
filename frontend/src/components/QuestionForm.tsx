import type { ChangeEvent, FormEvent, KeyboardEvent } from 'react';
import './QuestionForm.css';

interface QuestionFormProps {
  question: string;
  onQuestionChange: (value: string) => void;
  disabled: boolean;
  onSubmit: () => void;
}

export function QuestionForm({
  question,
  onQuestionChange,
  disabled,
  onSubmit,
}: QuestionFormProps) {
  const canSubmit = !disabled && question.trim().length > 0;

  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>): void => {
    if (event.key === 'Enter' && (event.metaKey || event.ctrlKey)) {
      event.preventDefault();
      if (canSubmit) onSubmit();
    }
  };

  const handleSubmit = (event: FormEvent<HTMLFormElement>): void => {
    event.preventDefault();
    if (canSubmit) onSubmit();
  };

  return (
    <section className="question-form-section">
      <header className="question-form-header">
        <h2 className="question-form-title">Pergunta</h2>
        <p className="question-form-subtitle">
          O agente busca trechos nos documentos ingeridos e responde com
          citações.
        </p>
      </header>

      <form className="question-form" onSubmit={handleSubmit}>
        <textarea
          id="question-input"
          className="question-textarea"
          value={question}
          onChange={(e: ChangeEvent<HTMLTextAreaElement>) =>
            onQuestionChange(e.target.value)
          }
          onKeyDown={handleKeyDown}
          placeholder="Ex: O que é multi-head attention e por que usar várias cabeças?"
          rows={4}
          disabled={disabled}
        />
        <div className="question-actions">
          <button
            type="submit"
            className="ask-button"
            disabled={!canSubmit}
          >
            <span>Perguntar</span>
            <kbd className="kbd-hint" aria-label="Atalho: comando enter">
              ⌘↵
            </kbd>
          </button>
        </div>
      </form>
    </section>
  );
}
