import { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { ThumbsUp, ThumbsDown, Copy, Check, Clock, Cpu, Zap, Shield } from 'lucide-react';
import type { QueryResponse } from '../lib/api';
import './AnswerPanel.css';

interface AnswerPanelProps {
  response: QueryResponse | null;
  onFeedback?: (helpful: boolean) => void;
}

export default function AnswerPanel({ response, onFeedback }: AnswerPanelProps) {
  const [copied, setCopied] = useState(false);
  const [feedbackGiven, setFeedbackGiven] = useState<boolean | null>(null);

  if (!response) return null;

  const handleCopy = () => {
    navigator.clipboard.writeText(response.answer);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleFeedback = (helpful: boolean) => {
    setFeedbackGiven(helpful);
    onFeedback?.(helpful);
  };

  const intentLabel: Record<string, string> = {
    semantic:    '🔍 Semantic',
    definition:  '🎯 Definition',
    dependency:  '🔗 Dependency',
    explanation: '📖 Explanation',
    unknown:     '🤖 Hybrid',
  };

  const totalMs = Object.values(response.pipeline_ms).reduce((a, b) => a + b, 0);

  return (
    <div className="answer-panel glass-card slide-up">
      {/* Metadata bar */}
      <div className="answer-meta">
        <span className="badge badge-primary">
          {intentLabel[response.intent] || response.intent}
        </span>
        {response.zero_llm_mode && (
          <span className="badge badge-success"><Shield size={12} /> Zero-LLM</span>
        )}
        {response.cached && (
          <span className="badge badge-info"><Zap size={12} /> Cached</span>
        )}

        <div className="meta-stats">
          <span className="meta-stat">
            <Clock size={12} /> {totalMs.toFixed(0)}ms
          </span>
          <span className="meta-stat">
            <Cpu size={12} /> {response.context_tokens} tokens
          </span>
          <span className="meta-stat mono" title="Confidence">
            {(response.confidence * 100).toFixed(0)}% conf
          </span>
        </div>
      </div>

      {/* UI warning */}
      {response.ui_message && (
        <div className={`answer-warning ${response.ui_action === 'suggest_refine' ? 'warn-refine' : 'warn-uncertain'}`}>
          {response.ui_message}
        </div>
      )}

      {/* Answer content */}
      <div className="answer-content">
        <ReactMarkdown
          components={{
            code({ className, children, ...props }) {
              const match = /language-(\w+)/.exec(className || '');
              const isInline = !match;
              return isInline ? (
                <code className="inline-code" {...props}>{children}</code>
              ) : (
                <SyntaxHighlighter
                  style={oneDark as any}
                  language={match[1]}
                  PreTag="div"
                  customStyle={{
                    borderRadius: '10px',
                    fontSize: '0.85rem',
                    margin: '12px 0',
                  }}
                >
                  {String(children).replace(/\n$/, '')}
                </SyntaxHighlighter>
              );
            },
          }}
        >
          {response.answer}
        </ReactMarkdown>
      </div>

      {/* Coverage warning */}
      {response.coverage_warning && (
        <div className="answer-warning warn-coverage">
          ⚠️ Context was truncated — results may be incomplete. Try a more specific query.
        </div>
      )}

      {/* Actions */}
      <div className="answer-actions">
        <button className="btn btn-ghost btn-sm" onClick={handleCopy}>
          {copied ? <Check size={14} /> : <Copy size={14} />}
          {copied ? 'Copied!' : 'Copy'}
        </button>

        <div className="feedback-group">
          <span className="feedback-label">Helpful?</span>
          <button
            className={`btn btn-icon btn-ghost ${feedbackGiven === true ? 'active-positive' : ''}`}
            onClick={() => handleFeedback(true)}
            disabled={feedbackGiven !== null}
          >
            <ThumbsUp size={14} />
          </button>
          <button
            className={`btn btn-icon btn-ghost ${feedbackGiven === false ? 'active-negative' : ''}`}
            onClick={() => handleFeedback(false)}
            disabled={feedbackGiven !== null}
          >
            <ThumbsDown size={14} />
          </button>
        </div>

        <span className="version-hash mono">{response.version_hash}</span>
      </div>
    </div>
  );
}
