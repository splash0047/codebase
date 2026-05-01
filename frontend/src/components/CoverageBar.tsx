import './CoverageBar.css';

interface CoverageBarProps {
  coveragePct: number;
  totalFiles: number;
  indexedFiles: number;
  ingestionStatus: string;
}

export default function CoverageBar({ coveragePct, totalFiles, indexedFiles, ingestionStatus }: CoverageBarProps) {
  const isActive = ingestionStatus === 'partial' || ingestionStatus === 'pending';
  const statusText =
    ingestionStatus === 'complete' ? 'Fully indexed' :
    ingestionStatus === 'partial'  ? 'Indexing in progress…' :
    ingestionStatus === 'failed'   ? 'Indexing failed' :
    'Waiting to start…';

  return (
    <div className="coverage-bar glass-card">
      <div className="coverage-info">
        <div className="coverage-label">
          <span>Coverage</span>
          <span className={`coverage-status ${ingestionStatus}`}>
            {isActive && <span className="spinner" style={{ width: 12, height: 12 }} />}
            {statusText}
          </span>
        </div>
        <div className="coverage-numbers mono">
          <span className="coverage-pct">
            {coveragePct.toFixed(1)}%
            {coveragePct < 100 && ingestionStatus !== 'complete' && ' ⚠️'}
          </span>
          <span className="coverage-files">
            {indexedFiles} / {totalFiles} files
          </span>
        </div>
      </div>
      <div className="coverage-track">
        <div
          className={`coverage-fill ${isActive ? 'animating' : ''}`}
          style={{ width: `${Math.min(coveragePct, 100)}%` }}
        />
      </div>
    </div>
  );
}
