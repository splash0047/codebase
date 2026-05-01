import { useState, useCallback } from 'react';
import Header from './components/Header';
import QueryPanel from './components/QueryPanel';
import GraphViewer from './components/GraphViewer';
import ResultsList from './components/ResultsList';
import AnswerPanel from './components/AnswerPanel';
import CoverageBar from './components/CoverageBar';
import { submitQuery, submitFeedback } from './lib/api';
import type { QueryResponse } from './lib/api';
import './App.css';

// Demo repo ID for development (replace with selector in production)
const DEMO_REPO_ID = 'demo-repo';

export default function App() {
  const [response, setResponse] = useState<QueryResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [streamStatus, setStreamStatus] = useState('');
  const [streamMessage, setStreamMessage] = useState('');
  const [lastQuery, setLastQuery] = useState('');

  // Mock repo state (will be fetched from API in production)
  const [repoState] = useState({
    name: 'codebase-knowledge-ai',
    coveragePct: 87.5,
    ingestionStatus: 'partial',
    totalFiles: 47,
    indexedFiles: 41,
  });

  const handleQuery = useCallback(async (query: string, intent?: string) => {
    setIsLoading(true);
    setStreamStatus('searching');
    setStreamMessage('🔍 Searching codebase…');
    setLastQuery(query);
    setResponse(null);

    try {
      // Use direct API call (SSE streaming can be toggled)
      const result = await submitQuery({
        query,
        repo_id: DEMO_REPO_ID,
        intent_override: intent || null,
      });
      setResponse(result);
    } catch (err: any) {
      // On API error, show a fallback message
      setResponse({
        answer: `⚠️ Could not reach the backend. Error: ${err.message}\n\nMake sure the FastAPI server is running on \`http://localhost:8000\`.`,
        intent: 'unknown',
        confidence: 0,
        confidence_gap: 0,
        results: [],
        context_tokens: 0,
        pipeline_ms: {},
        zero_llm_mode: true,
        cached: false,
        version_hash: 'error',
        coverage_warning: false,
        ui_action: 'suggest_refine',
        ui_message: 'Backend is not reachable. Start it with: uvicorn app.main:app --reload',
      });
    } finally {
      setIsLoading(false);
      setStreamStatus('');
      setStreamMessage('');
    }
  }, []);

  const handleFeedback = useCallback((helpful: boolean) => {
    if (response) {
      submitFeedback({
        repo_id: DEMO_REPO_ID,
        query: lastQuery,
        version_hash: response.version_hash,
        helpful,
      }).catch(() => { /* Feedback is fire-and-forget */ });
    }
  }, [response, lastQuery]);

  return (
    <div className="app-layout">
      <Header
        repoName={repoState.name}
        coveragePct={repoState.coveragePct}
        ingestionStatus={repoState.ingestionStatus}
      />

      <main className="app-main">
        {/* Left column: Query + Answer */}
        <div className="main-left">
          <QueryPanel
            onSubmit={handleQuery}
            isLoading={isLoading}
            streamStatus={streamStatus}
            streamMessage={streamMessage}
          />

          <AnswerPanel
            response={response}
            onFeedback={handleFeedback}
          />

          <ResultsList
            results={response?.results || []}
          />
        </div>

        {/* Right column: Graph + Coverage */}
        <div className="main-right">
          <CoverageBar
            coveragePct={repoState.coveragePct}
            totalFiles={repoState.totalFiles}
            indexedFiles={repoState.indexedFiles}
            ingestionStatus={repoState.ingestionStatus}
          />

          <GraphViewer
            results={response?.results || []}
          />
        </div>
      </main>
    </div>
  );
}
