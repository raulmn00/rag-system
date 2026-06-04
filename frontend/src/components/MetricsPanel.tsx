import { useEffect, useState } from 'react';
import type { CSSProperties } from 'react';
import type { EvaluateResponse } from '../api';
import { Spinner } from './Spinner';
import './MetricsPanel.css';

/**
 * Discriminated union over the panel's lifecycle. The parent builds the
 * variant that matches its state; the panel branches once and renders
 * the corresponding view. Forces every callsite to handle all cases.
 */
export type MetricsState =
  | { kind: 'idle' }
  | { kind: 'loading' }
  | { kind: 'unavailable' }
  | { kind: 'error'; message: string }
  | { kind: 'success'; result: EvaluateResponse };

interface MetricsPanelProps {
  state: MetricsState;
  onEvaluate: () => void;
}

const FAITHFULNESS_HINT =
  'Mede o quanto da resposta é diretamente suportado pelos trechos recuperados — é o indicador direto de alucinação.';

const ANSWER_RELEVANCY_HINT =
  'Mede o quanto a resposta efetivamente aborda a pergunta feita, independente de estar correta.';

export function MetricsPanel({ state, onEvaluate }: MetricsPanelProps) {
  const showActionButton =
    state.kind === 'idle' ||
    state.kind === 'success' ||
    state.kind === 'error';

  return (
    <section className="metrics-panel">
      <header className="metrics-panel-header">
        <h2 className="metrics-panel-title">Qualidade desta resposta</h2>
        <p className="metrics-panel-subtitle">
          Avaliada por <strong>Ragas</strong> (modelo externo julga
          fidelidade e relevância). Faz chamadas de LLM, portanto leva
          alguns segundos.
        </p>
      </header>

      <div className="metrics-body">
        {state.kind === 'idle' && (
          <p className="metrics-empty">
            Clique no botão abaixo para avaliar a resposta atual.
          </p>
        )}

        {state.kind === 'loading' && (
          <Spinner message="Avaliando qualidade... (pode levar alguns segundos)" />
        )}

        {state.kind === 'success' && (
          <div className="gauges">
            <Gauge
              value={clamp01(state.result.faithfulness)}
              label="Fidelidade"
              hint={FAITHFULNESS_HINT}
            />
            <Gauge
              value={clamp01(state.result.answer_relevancy)}
              label="Relevância da resposta"
              hint={ANSWER_RELEVANCY_HINT}
            />
          </div>
        )}

        {state.kind === 'unavailable' && (
          <div className="metrics-unavailable" role="status">
            <span className="metrics-info-icon" aria-hidden="true">
              i
            </span>
            <div>
              <p className="metrics-unavailable-title">
                Avaliação automática não está habilitada neste deploy.
              </p>
              <p className="metrics-unavailable-detail">
                O endpoint de avaliação depende do Ragas, que é um
                extra opcional. O agente continua funcionando
                normalmente sem ele.
              </p>
            </div>
          </div>
        )}

        {state.kind === 'error' && (
          <div className="metrics-error" role="alert">
            <span className="error-dot" aria-hidden="true" />
            <p>{state.message}</p>
          </div>
        )}
      </div>

      {showActionButton && (
        <div className="metrics-actions">
          <button
            type="button"
            className="evaluate-button"
            onClick={onEvaluate}
          >
            {state.kind === 'success'
              ? 'Reavaliar resposta'
              : 'Avaliar qualidade desta resposta'}
          </button>
        </div>
      )}
    </section>
  );
}

// ---------------------------------------------------------------------------
//                              Gauge component
// ---------------------------------------------------------------------------

interface GaugeProps {
  value: number;
  label: string;
  hint: string;
}

const RADIUS = 50;
const CIRCUMFERENCE = 2 * Math.PI * RADIUS;

function Gauge({ value, label, hint }: GaugeProps) {
  // Start the fill at zero on mount so the CSS transition on
  // stroke-dasharray plays from 0 → value. Using rAF instead of a
  // microtask ensures the initial render commits with dasharray=0
  // before we set the real value.
  const [animatedValue, setAnimatedValue] = useState<number>(0);

  useEffect(() => {
    const id = window.requestAnimationFrame(() => setAnimatedValue(value));
    return () => window.cancelAnimationFrame(id);
  }, [value]);

  const dash = animatedValue * CIRCUMFERENCE;
  const color = colorForScore(value);
  const arcStyle: CSSProperties = {
    ['--gauge-color' as string]: color,
  };

  return (
    <div className="gauge" style={arcStyle}>
      <div className="gauge-arc">
        <svg
          viewBox="0 0 120 120"
          width="148"
          height="148"
          aria-hidden="true"
        >
          <circle cx="60" cy="60" r={RADIUS} className="gauge-track" />
          <circle
            cx="60"
            cy="60"
            r={RADIUS}
            className="gauge-fill"
            strokeDasharray={`${dash} ${CIRCUMFERENCE}`}
            transform="rotate(-90 60 60)"
          />
        </svg>
        <div className="gauge-center">
          <span className="gauge-value">{Math.round(value * 100)}%</span>
          <span className="gauge-fraction">{value.toFixed(2)}</span>
        </div>
      </div>
      <div className="gauge-meta">
        <span className="gauge-label">{label}</span>
        <span className="gauge-tooltip-wrapper">
          <button
            type="button"
            className="gauge-tooltip-trigger"
            aria-label={`Mais informação sobre ${label}`}
          >
            ?
          </button>
          <span className="gauge-tooltip" role="tooltip">
            {hint}
          </span>
        </span>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
//                                  Helpers
// ---------------------------------------------------------------------------

function clamp01(n: number): number {
  if (Number.isNaN(n)) return 0;
  if (n < 0) return 0;
  if (n > 1) return 1;
  return n;
}

function colorForScore(score: number): string {
  if (score >= 0.8) return 'var(--good)';
  if (score >= 0.5) return 'var(--warn)';
  return 'var(--bad)';
}
