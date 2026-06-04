import { Spinner } from './Spinner';
import './AnswerBox.css';

interface AnswerBoxProps {
  answer: string | null;
  error: string | null;
  loading: boolean;
}

export function AnswerBox({ answer, error, loading }: AnswerBoxProps) {
  if (loading) {
    return (
      <Spinner message="Buscando trechos e gerando resposta..." />
    );
  }

  if (error !== null) {
    return (
      <div className="answer-box answer-box-error" role="alert">
        <div className="answer-meta answer-meta-error">
          <span className="error-dot" aria-hidden="true" />
          A requisição falhou
        </div>
        <p className="answer-error-text">{error}</p>
      </div>
    );
  }

  if (answer === null) return null;

  return (
    <div className="answer-box">
      <div className="answer-meta">Resposta</div>
      <p className="answer-text">{answer}</p>
    </div>
  );
}
