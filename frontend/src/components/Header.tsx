import { useState } from 'react';
import { GitBranch, Zap, Menu, X } from 'lucide-react';
import './Header.css';

interface HeaderProps {
  repoName: string;
  coveragePct: number;
  ingestionStatus: string;
  onSettingsClick?: () => void;
}

export default function Header({ repoName, coveragePct, ingestionStatus, onSettingsClick }: HeaderProps) {
  const [menuOpen, setMenuOpen] = useState(false);

  const statusColor =
    ingestionStatus === 'complete' ? 'var(--success)' :
    ingestionStatus === 'partial'  ? 'var(--warning)' :
    ingestionStatus === 'failed'   ? 'var(--error)' :
    'var(--text-muted)';

  return (
    <header className="app-header">
      <div className="header-left">
        <div className="logo-mark">
          <Zap size={20} />
        </div>
        <h1 className="app-title">
          <span className="title-accent">Codebase</span> Knowledge AI
        </h1>
      </div>

      <div className="header-center">
        <div className="repo-pill glass-card">
          <GitBranch size={14} />
          <span className="mono truncate">{repoName || 'No repo selected'}</span>
          <span className="status-dot" style={{ background: statusColor }} />
        </div>
      </div>

      <div className="header-right">
        <div className="coverage-mini">
          <div className="coverage-mini-bar">
            <div
              className="coverage-mini-fill"
              style={{ width: `${Math.min(coveragePct, 100)}%` }}
            />
          </div>
          <span className="coverage-mini-label mono">
            {coveragePct.toFixed(0)}%
            {coveragePct < 100 && ' ⚠️'}
          </span>
        </div>
        <button
          className="btn btn-icon btn-ghost"
          onClick={() => { setMenuOpen(!menuOpen); onSettingsClick?.(); }}
          aria-label="Menu"
        >
          {menuOpen ? <X size={18} /> : <Menu size={18} />}
        </button>
      </div>
    </header>
  );
}
