import { useState, useRef, useEffect } from 'react';
import { Send, Sparkles, Target, GitFork, BookOpen, Loader2 } from 'lucide-react';
import './QueryPanel.css';

interface QueryPanelProps {
  onSubmit: (query: string, intent?: string) => void;
  isLoading: boolean;
  streamStatus?: string;
  streamMessage?: string;
}

const INTENT_OPTIONS = [
  { value: '',            label: 'Auto-detect', icon: Sparkles, color: 'var(--accent-primary)' },
  { value: 'definition',  label: 'Definition',  icon: Target,   color: 'var(--info)' },
  { value: 'dependency',  label: 'Dependency',  icon: GitFork,  color: 'var(--warning)' },
  { value: 'explanation', label: 'Explanation', icon: BookOpen,  color: 'var(--success)' },
];

export default function QueryPanel({ onSubmit, isLoading, streamStatus, streamMessage }: QueryPanelProps) {
  const [query, setQuery] = useState('');
  const [selectedIntent, setSelectedIntent] = useState('');
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => { inputRef.current?.focus(); }, []);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim() || isLoading) return;
    onSubmit(query.trim(), selectedIntent || undefined);
  };

  return (
    <div className="query-panel glass-card fade-in">
      {/* Intent selector */}
      <div className="intent-row">
        {INTENT_OPTIONS.map((opt) => {
          const Icon = opt.icon;
          const active = selectedIntent === opt.value;
          return (
            <button
              key={opt.value}
              className={`intent-chip ${active ? 'active' : ''}`}
              style={active ? { borderColor: opt.color, color: opt.color } : {}}
              onClick={() => setSelectedIntent(opt.value)}
              type="button"
            >
              <Icon size={14} />
              {opt.label}
            </button>
          );
        })}
      </div>

      {/* Query input */}
      <form className="query-form" onSubmit={handleSubmit}>
        <div className="query-input-wrap">
          <input
            ref={inputRef}
            className="input query-input"
            type="text"
            placeholder="Ask about your codebase… e.g. 'Where is the auth handler defined?'"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            disabled={isLoading}
          />
          <button
            className="btn btn-primary query-submit"
            type="submit"
            disabled={!query.trim() || isLoading}
          >
            {isLoading ? <Loader2 size={18} className="spin-icon" /> : <Send size={18} />}
          </button>
        </div>
      </form>

      {/* Stream status indicator */}
      {streamStatus && (
        <div className="stream-status fade-in">
          <div className="stream-dot pulse-ring" />
          <span>{streamMessage || streamStatus}</span>
        </div>
      )}
    </div>
  );
}
