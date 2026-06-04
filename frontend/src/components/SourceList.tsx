import type { Source } from '../api';
import './SourceList.css';

interface SourceListProps {
  sources: Source[];
}

export function SourceList({ sources }: SourceListProps) {
  if (sources.length === 0) return null;

  // Cross-encoder rerank scores aren't bounded to [0, 1], so a fixed
  // 0-100% mapping wouldn't tell the user anything useful. Normalize
  // bar widths relative to the max in this result set — that conveys
  // "this is the most relevant" visually while we still show the raw
  // numeric score from the API beside each row.
  const maxScore = sources.reduce((m, s) => Math.max(m, s.score), 0);
  const denom = maxScore > 0 ? maxScore : 1;

  return (
    <section className="source-list">
      <h3 className="source-list-title">
        Trechos recuperados
        <span className="source-list-count">{sources.length} fontes</span>
      </h3>
      <ul className="source-items">
        {sources.map((source) => {
          const widthPct = Math.max(2, (source.score / denom) * 100);
          return (
            <li key={source.chunk_id} className="source-item">
              <div className="source-row">
                <span className="source-filename" title={source.source}>
                  {source.source}
                </span>
                <span className="source-score" aria-label="score">
                  {source.score.toFixed(3)}
                </span>
              </div>
              <div
                className="source-bar"
                aria-label="relevância do trecho recuperado"
                role="progressbar"
                aria-valuenow={source.score}
              >
                <span
                  className="source-bar-fill"
                  style={{ width: `${widthPct}%` }}
                />
              </div>
              <p className="source-text">{source.text}</p>
            </li>
          );
        })}
      </ul>
      <p className="source-list-footnote">
        A barra mostra a relevância de cada trecho relativa ao mais
        bem-pontuado do conjunto. O valor numérico é o score retornado
        pela API (cosine-similarity ou logit do re-ranker).
      </p>
    </section>
  );
}
