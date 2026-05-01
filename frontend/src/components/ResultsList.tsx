import { FileCode, Layers, ChevronDown, ChevronUp } from 'lucide-react';
import { useState } from 'react';
import type { ResultItem } from '../lib/api';
import './ResultsList.css';

interface ResultsListProps {
  results: ResultItem[];
  onResultClick?: (result: ResultItem) => void;
}

function ScoreBar({ score }: { score: number }) {
  const pct = Math.min(score * 100, 100);
  const color =
    score >= 0.7 ? 'var(--success)' :
    score >= 0.4 ? 'var(--warning)' :
    'var(--error)';
  return (
    <div className="score-bar">
      <div className="score-bar-fill" style={{ width: `${pct}%`, background: color }} />
    </div>
  );
}

export default function ResultsList({ results, onResultClick }: ResultsListProps) {
  const [expandedIdx, setExpandedIdx] = useState<number | null>(null);

  if (results.length === 0) return null;

  return (
    <div className="results-list glass-card">
      <div className="results-header">
        <Layers size={16} />
        <h3>Retrieved Code Contexts</h3>
        <span className="badge badge-primary">{results.length}</span>
      </div>

      <div className="results-items">
        {results.map((r, i) => {
          const isExpanded = expandedIdx === i;
          return (
            <div
              key={i}
              className={`result-item fade-in ${isExpanded ? 'expanded' : ''} ${i < 3 ? 'top-result' : ''}`}
              style={{ animationDelay: `${i * 60}ms` }}
            >
              <div
                className="result-row"
                onClick={() => { setExpandedIdx(isExpanded ? null : i); onResultClick?.(r); }}
              >
                <div className="result-rank">
                  <span className="rank-number">{i + 1}</span>
                </div>
                <div className="result-info">
                  <div className="result-symbol mono">
                    <FileCode size={14} />
                    {r.symbol || 'Unknown'}
                  </div>
                  <div className="result-file truncate">
                    {r.file || 'unknown file'}
                  </div>
                </div>
                <div className="result-meta">
                  <ScoreBar score={r.score} />
                  <span className="score-label mono">{(r.score * 100).toFixed(0)}%</span>
                </div>
                <div className="result-expand">
                  {isExpanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                </div>
              </div>

              {isExpanded && (
                <div className="result-details slide-up">
                  <div className="detail-row">
                    <span className="detail-label">Lines</span>
                    <span className="mono">{r.lines}</span>
                  </div>
                  <div className="detail-row">
                    <span className="detail-label">Sources</span>
                    <div className="source-tags">
                      {r.sources.map((s) => (
                        <span key={s} className={`badge badge-${s === 'vector' ? 'primary' : s === 'graph' ? 'warning' : 'info'}`}>
                          {s}
                        </span>
                      ))}
                    </div>
                  </div>
                  {r.gap !== undefined && r.gap > 0 && (
                    <div className="detail-row">
                      <span className="detail-label">Confidence gap</span>
                      <span className="mono">{(r.gap * 100).toFixed(1)}%</span>
                    </div>
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
