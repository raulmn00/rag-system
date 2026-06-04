import './Spinner.css';

interface SpinnerProps {
  message: string;
  /** Slightly different size for inline contexts (small) vs the main loading
   *  surfaces (medium). Defaults to medium. */
  size?: 'sm' | 'md';
}

export function Spinner({ message, size = 'md' }: SpinnerProps) {
  return (
    <div className={`spinner spinner-${size}`} role="status" aria-live="polite">
      <span className="spinner-ring" aria-hidden="true" />
      <span className="spinner-text">{message}</span>
    </div>
  );
}
