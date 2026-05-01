import { useEffect, useRef, useState, useCallback, useMemo } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import { Maximize2, Minimize2, ZoomIn } from 'lucide-react';
import type { ResultItem } from '../lib/api';
import './GraphViewer.css';

interface GraphNode {
  id: string;
  name: string;
  type: 'function' | 'class' | 'file' | 'module';
  score: number;
  file?: string;
  highlighted?: boolean;
}

interface GraphLink {
  source: string;
  target: string;
  type: 'CALLS' | 'EXTENDS' | 'DEFINED_IN' | 'IMPORTS';
}

interface GraphViewerProps {
  results: ResultItem[];
  onNodeClick?: (node: GraphNode) => void;
  focusMode?: 'all' | 'high_importance' | 'path';
}

const NODE_COLORS: Record<string, string> = {
  function: '#6366f1',
  class:    '#8b5cf6',
  file:     '#3b82f6',
  module:   '#10b981',
};

const EDGE_COLORS: Record<string, string> = {
  CALLS:      '#f59e0b',
  EXTENDS:    '#ef4444',
  DEFINED_IN: '#6366f1',
  IMPORTS:    '#10b981',
};

function buildGraph(results: ResultItem[]): { nodes: GraphNode[]; links: GraphLink[] } {
  const nodeMap = new Map<string, GraphNode>();
  const links: GraphLink[] = [];

  results.forEach((r, i) => {
    const symbolId = r.symbol || `result-${i}`;
    const fileId   = r.file || `file-${i}`;

    if (!nodeMap.has(symbolId)) {
      nodeMap.set(symbolId, {
        id: symbolId,
        name: r.symbol || 'unknown',
        type: r.sources?.includes('graph') ? 'class' : 'function',
        score: r.score,
        file: r.file || undefined,
        highlighted: i < 3,
      });
    }

    if (r.file && !nodeMap.has(fileId)) {
      nodeMap.set(fileId, {
        id: fileId,
        name: r.file.split('/').pop() || r.file,
        type: 'file',
        score: r.score * 0.5,
        file: r.file,
      });
    }

    if (r.file) {
      links.push({ source: symbolId, target: fileId, type: 'DEFINED_IN' });
    }

    // Create inter-symbol CALLS links for adjacent results (heuristic)
    if (i > 0) {
      const prevSymbol = results[i - 1].symbol || `result-${i - 1}`;
      if (prevSymbol !== symbolId) {
        links.push({ source: prevSymbol, target: symbolId, type: 'CALLS' });
      }
    }
  });

  return { nodes: Array.from(nodeMap.values()), links };
}

export default function GraphViewer({ results, onNodeClick, focusMode = 'all' }: GraphViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const fgRef = useRef<any>(null);
  const [expanded, setExpanded] = useState(false);
  const [dimensions, setDimensions] = useState({ width: 600, height: 400 });

  const graphData = useMemo(() => buildGraph(results), [results]);

  // Filter nodes based on focus mode
  const filteredData = useMemo(() => {
    if (focusMode === 'high_importance') {
      const filtered = graphData.nodes.filter((n) => n.score > 0.3);
      const ids = new Set(filtered.map((n) => n.id));
      return {
        nodes: filtered,
        links: graphData.links.filter((l) => ids.has(l.source as string) && ids.has(l.target as string)),
      };
    }
    return graphData;
  }, [graphData, focusMode]);

  useEffect(() => {
    const resize = () => {
      if (containerRef.current) {
        setDimensions({
          width: containerRef.current.clientWidth,
          height: expanded ? window.innerHeight - 100 : 400,
        });
      }
    };
    resize();
    window.addEventListener('resize', resize);
    return () => window.removeEventListener('resize', resize);
  }, [expanded]);

  const nodeCanvasObject = useCallback((node: any, ctx: CanvasRenderingContext2D) => {
    const size = 4 + (node.score || 0) * 12;
    const color = NODE_COLORS[node.type] || '#6366f1';

    // Glow for highlighted nodes
    if (node.highlighted) {
      ctx.beginPath();
      ctx.arc(node.x, node.y, size + 6, 0, 2 * Math.PI);
      ctx.fillStyle = `${color}33`;
      ctx.fill();
    }

    // Node circle
    ctx.beginPath();
    ctx.arc(node.x, node.y, size, 0, 2 * Math.PI);
    ctx.fillStyle = color;
    ctx.fill();
    ctx.strokeStyle = node.highlighted ? '#fff' : `${color}88`;
    ctx.lineWidth = node.highlighted ? 2 : 1;
    ctx.stroke();

    // Label
    ctx.font = `${node.highlighted ? '600' : '400'} 10px Inter, sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.fillStyle = node.highlighted ? '#f1f5f9' : '#94a3b8';
    ctx.fillText(node.name, node.x, node.y + size + 4);
  }, []);

  const linkColor = useCallback((link: any) => {
    return EDGE_COLORS[link.type] || '#64748b44';
  }, []);

  if (results.length === 0) {
    return (
      <div className="graph-empty glass-card">
        <div className="graph-empty-icon">🕸️</div>
        <p>Knowledge graph will appear after your first query</p>
      </div>
    );
  }

  return (
    <div className={`graph-viewer glass-card ${expanded ? 'expanded' : ''}`} ref={containerRef}>
      <div className="graph-toolbar">
        <h3 className="graph-title">Knowledge Graph</h3>
        <div className="graph-legend">
          {Object.entries(EDGE_COLORS).map(([type, color]) => (
            <span key={type} className="legend-item">
              <span className="legend-line" style={{ background: color }} />
              <span className="legend-label">{type}</span>
            </span>
          ))}
        </div>
        <div className="graph-actions">
          <button className="btn btn-icon btn-ghost" onClick={() => fgRef.current?.zoomToFit(400, 40)} title="Fit view">
            <ZoomIn size={16} />
          </button>
          <button className="btn btn-icon btn-ghost" onClick={() => setExpanded(!expanded)} title={expanded ? 'Collapse' : 'Expand'}>
            {expanded ? <Minimize2 size={16} /> : <Maximize2 size={16} />}
          </button>
        </div>
      </div>

      <ForceGraph2D
        ref={fgRef}
        graphData={filteredData}
        width={dimensions.width - 2}
        height={expanded ? dimensions.height - 60 : 340}
        backgroundColor="transparent"
        nodeCanvasObject={nodeCanvasObject}
        linkColor={linkColor}
        linkWidth={1.5}
        linkDirectionalArrowLength={4}
        linkDirectionalArrowRelPos={0.85}
        onNodeClick={(node: any) => onNodeClick?.(node)}
        cooldownTime={2000}
        d3AlphaDecay={0.04}
        d3VelocityDecay={0.3}
      />
    </div>
  );
}
