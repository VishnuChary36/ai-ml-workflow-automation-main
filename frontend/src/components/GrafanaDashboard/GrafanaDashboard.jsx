import React, { useState, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  ArrowLeft, LayoutDashboard, RefreshCw, ExternalLink,
  AlertCircle, Sparkles, ChevronDown, ChevronUp,
  Database, Activity, Loader2, CheckCircle,
  TrendingUp, BarChart3, Brain, Bell, FileText
} from 'lucide-react';
import {
  getGrafanaStatus,
  createModelDashboard,
  getModelNarrative,
  setupGrafanaDatasource
} from '../../api/client';

const GrafanaDashboard = ({ modelId, onBack }) => {
  const navigate = useNavigate();

  // Dashboard state
  const [dashboardInfo, setDashboardInfo] = useState(null);
  const [embedUrl, setEmbedUrl] = useState(null);
  const [loading, setLoading] = useState(true);
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState(null);

  // Narrative state
  const [narrative, setNarrative] = useState(null);
  const [narrativeMode, setNarrativeMode] = useState(null);
  const [narrativeLoading, setNarrativeLoading] = useState(false);
  const [narrativeExpanded, setNarrativeExpanded] = useState(true);

  // Grafana health
  const [grafanaOnline, setGrafanaOnline] = useState(null);

  // ── Check Grafana health ─────────────────────────────────────────
  const checkHealth = useCallback(async () => {
    try {
      const status = await getGrafanaStatus();
      const online = status.status === 'ok';
      setGrafanaOnline(online);
      return online;
    } catch {
      setGrafanaOnline(false);
      return false;
    }
  }, []);

  // ── Generate dashboard ───────────────────────────────────────────
  const generateDashboard = useCallback(async () => {
    try {
      setGenerating(true);
      setError(null);

      // Ensure datasource exists first
      try {
        await setupGrafanaDatasource();
      } catch {
        // Non-fatal — datasource may already exist
      }

      const result = await createModelDashboard(modelId);
      setDashboardInfo(result);
      setEmbedUrl(result.embed_url);
      return result;
    } catch (err) {
      const msg =
        err.response?.data?.detail ||
        err.message ||
        'Failed to generate Grafana dashboard';
      setError(msg);
      return null;
    } finally {
      setGenerating(false);
    }
  }, [modelId]);

  // ── Fetch narrative ──────────────────────────────────────────────
  const fetchNarrative = useCallback(async () => {
    try {
      setNarrativeLoading(true);
      const data = await getModelNarrative(modelId);
      setNarrative(data.narrative);
      setNarrativeMode(data.mode);
    } catch {
      // Non-critical — we just won't show a narrative
      setNarrative(null);
    } finally {
      setNarrativeLoading(false);
    }
  }, [modelId]);

  // ── Initial load ─────────────────────────────────────────────────
  useEffect(() => {
    const init = async () => {
      setLoading(true);
      const online = await checkHealth();

      if (!online) {
        setError(
          'Grafana is not reachable. Please ensure Grafana is running and the GRAFANA_URL / GRAFANA_API_KEY environment variables are configured.'
        );
        setLoading(false);
        return;
      }

      await generateDashboard();
      fetchNarrative(); // fire-and-forget
      setLoading(false);
    };

    init();
  }, [modelId]); // eslint-disable-line react-hooks/exhaustive-deps

  // ── Handle refresh ───────────────────────────────────────────────
  const handleRefresh = async () => {
    await generateDashboard();
    fetchNarrative();
  };

  // ── Simple markdown-ish renderer ─────────────────────────────────
  // ── Inline markdown renderer ──────────────────────────────────────
  const renderInline = (text) => {
    // Split on **bold**, *italic*, and `code` patterns
    const tokens = text.split(/(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)/g);
    return tokens.map((tok, j) => {
      if (tok.startsWith('**') && tok.endsWith('**')) {
        return <strong key={j} className="font-semibold text-on-surface">{tok.slice(2, -2)}</strong>;
      }
      if (tok.startsWith('*') && tok.endsWith('*') && tok.length > 2) {
        return <em key={j} className="italic text-on-surface-variant">{tok.slice(1, -1)}</em>;
      }
      if (tok.startsWith('`') && tok.endsWith('`')) {
        return (
          <code key={j} className="px-1.5 py-0.5 bg-surface-container-high text-primary-500 rounded text-sm font-mono">
            {tok.slice(1, -1)}
          </code>
        );
      }
      return tok;
    });
  };

  const renderNarrative = (text) => {
    if (!text) return null;
    return text.split('\n').map((line, i) => {
      // Horizontal rule
      if (/^---+$/.test(line.trim())) {
        return <hr key={i} className="my-4 border-outline-variant" />;
      }
      // Headings
      if (line.startsWith('## ')) {
        return (
          <h2 key={i} className="text-lg font-bold text-on-surface mt-5 mb-2">
            {renderInline(line.replace('## ', ''))}
          </h2>
        );
      }
      if (line.startsWith('### ')) {
        return (
          <h3 key={i} className="text-base font-semibold text-on-surface-variant mt-4 mb-1">
            {renderInline(line.replace('### ', ''))}
          </h3>
        );
      }
      // Blockquote
      if (line.startsWith('> ')) {
        return (
          <blockquote key={i} className="border-l-4 border-primary-400 pl-4 py-1 my-2 rounded-r-lg text-on-surface-variant" style={{ background: 'rgba(77, 142, 255, 0.06)' }}>
            {renderInline(line.replace(/^>\s*/, ''))}
          </blockquote>
        );
      }
      // Pipe table row
      if (line.trim().startsWith('|') && line.trim().endsWith('|')) {
        const cells = line.trim().slice(1, -1).split('|').map(c => c.trim());
        // Skip separator rows like |---|---|
        if (cells.every(c => /^[-:]+$/.test(c))) return null;
        const isBold = cells.some(c => /[A-Z]/.test(c) && !/\d/.test(c));
        return (
          <div key={i} className="flex gap-0 text-sm font-mono">
            {cells.map((cell, ci) => (
              <span key={ci} className={`flex-1 px-3 py-1.5 border border-outline-variant ${isBold && i === 0 ? 'font-semibold bg-surface-container-low text-on-surface-variant' : 'text-on-surface-variant'}`}>
                {renderInline(cell)}
              </span>
            ))}
          </div>
        );
      }
      // List items
      if (line.startsWith('- ') || line.startsWith('* ')) {
        return (
          <li key={i} className="ml-5 text-on-surface-variant list-disc leading-relaxed">
            {renderInline(line.replace(/^[-*] /, ''))}
          </li>
        );
      }
      // Empty line
      if (line.trim() === '') return <div key={i} className="h-2" />;
      // Regular paragraph with inline formatting
      return (
        <p key={i} className="text-on-surface-variant leading-relaxed">
          {renderInline(line)}
        </p>
      );
    });
  };

  // ── Loading state ────────────────────────────────────────────────
  if (loading) {
    return (
      <div className="page-container">
        <div className="content-wrapper">
          <Header onBack={onBack} />
          <div className="flex flex-col items-center justify-center h-64 gap-4">
            <Loader2 className="w-10 h-10 text-primary-500 animate-spin" />
            <p className="text-on-surface-variant">Connecting to Grafana…</p>
          </div>
        </div>
      </div>
    );
  }

  // ── Error state (Grafana offline) ────────────────────────────────
  if (error && !embedUrl) {
    return (
      <div className="page-container">
        <div className="content-wrapper">
          <Header onBack={onBack} />
          <div className="rounded-xl p-6 text-center" style={{ background: 'rgba(239, 68, 68, 0.06)', border: '1px solid rgba(239, 68, 68, 0.2)' }}>
            <AlertCircle className="w-10 h-10 text-error-400 mx-auto mb-3" />
            <p className="text-error-400 mb-4 whitespace-pre-wrap">{error}</p>
            <button onClick={handleRefresh} className="btn-primary">
              <RefreshCw size={16} />
              Retry
            </button>
          </div>
        </div>
      </div>
    );
  }

  // ── Main render ──────────────────────────────────────────────────
  return (
    <div className="page-container">
      <div className="content-wrapper">
        <Header onBack={onBack}>
          <div className="flex items-center gap-3">
            {grafanaOnline !== null && (
              <span
                className={`inline-flex items-center gap-1.5 text-xs font-medium px-2.5 py-1 rounded-full ${
                  grafanaOnline
                    ? 'text-success-500'
                    : 'text-error-400'
                }`}
                style={{ background: grafanaOnline ? 'rgba(16,185,129,0.1)' : 'rgba(239,68,68,0.1)' }}
              >
                <span
                  className={`w-2 h-2 rounded-full ${
                    grafanaOnline ? 'bg-success-500' : 'bg-error-500'
                  }`}
                />
                {grafanaOnline ? 'Grafana Online' : 'Offline'}
              </span>
            )}
            <button
              onClick={handleRefresh}
              disabled={generating}
              className="btn-secondary text-sm"
            >
              <RefreshCw size={14} className={generating ? 'animate-spin' : ''} />
              {generating ? 'Refreshing…' : 'Refresh'}
            </button>
            {embedUrl && (
              <a
                href={embedUrl.replace('&kiosk', '')}
                target="_blank"
                rel="noopener noreferrer"
                className="btn-secondary text-sm"
              >
                <ExternalLink size={14} />
                Open in Grafana
              </a>
            )}
          </div>
        </Header>

        {/* ── Dashboard Sections Guide ─────────────────────── */}
        <div className="section-card mb-6">
          <div className="p-5">
            <h2 className="text-lg font-semibold text-on-surface mb-4 flex items-center gap-2">
              <LayoutDashboard size={18} className="text-primary-500" />
              Dashboard Sections
            </h2>
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
              {[
                { icon: '📌', label: 'Dataset Summary', desc: 'KPI cards & stats' },
                { icon: '📈', label: 'Key Trends', desc: 'Time-series & drift' },
                { icon: '📊', label: 'Feature Insights', desc: 'Distributions & balance' },
                { icon: '🤖', label: 'Model Performance', desc: 'Accuracy & predictions' },
                { icon: '🧠', label: 'AI Explanation', desc: 'Data-driven insights' },
                { icon: '🚨', label: 'Alerts / Drift', desc: 'Thresholds & anomalies' },
              ].map((section, i) => (
                <div key={i} className="bg-surface-container-low rounded-lg p-3 text-center border border-outline-variant hover:border-primary-200 transition-colors">
                  <span className="text-2xl">{section.icon}</span>
                  <p className="text-xs font-semibold text-on-surface-variant mt-1">{section.label}</p>
                  <p className="text-[10px] text-on-surface-variant mt-0.5">{section.desc}</p>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* ── AI Narrative ─────────────────────────────────────── */}
        {(narrative || narrativeLoading) && (
          <div className="section-card mb-6">
            <button
              onClick={() => setNarrativeExpanded(!narrativeExpanded)}
              className="w-full flex items-center justify-between p-5"
            >
              <div className="flex items-center gap-3">
                <div className="w-9 h-9 bg-gradient-to-br from-purple-100 to-purple-50 rounded-lg flex items-center justify-center">
                  <Sparkles size={18} className="text-purple-600" />
                </div>
                <div className="text-left">
                  <h2 className="text-lg font-semibold text-on-surface">
                    AI-Generated Insights
                  </h2>
                  {narrativeMode && (
                    <span className="text-xs text-on-surface-variant">
                      {narrativeMode === 'llm' ? 'Powered by LLM' : 'Template-based analysis'}
                    </span>
                  )}
                </div>
              </div>
              {narrativeExpanded ? (
                <ChevronUp size={20} className="text-on-surface-variant" />
              ) : (
                <ChevronDown size={20} className="text-on-surface-variant" />
              )}
            </button>

            {narrativeExpanded && (
              <div className="px-5 pb-5 border-t border-outline-variant pt-4">
                {narrativeLoading ? (
                  <div className="flex items-center gap-3 text-on-surface-variant py-4">
                    <Loader2 size={16} className="animate-spin" />
                    Generating insights…
                  </div>
                ) : (
                  <div className="prose prose-slate max-w-none">
                    {renderNarrative(narrative)}
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* ── Embedded Grafana Dashboard ───────────────────────── */}
        {embedUrl ? (
          <div className="section-card overflow-hidden">
            <div className="flex items-center justify-between px-5 py-3 border-b border-outline-variant bg-surface-container-low/50">
              <div className="flex items-center gap-2 text-sm text-on-surface-variant">
                <Activity size={14} className="text-primary-500" />
                <span className="font-medium">
                  {dashboardInfo?.title || 'Grafana Dashboard'}
                </span>
              </div>
              {dashboardInfo?.uid && (
                <span className="text-xs font-mono text-on-surface-variant">
                  UID: {dashboardInfo.uid}
                </span>
              )}
            </div>
            <iframe
              src={embedUrl}
              title="Grafana Dashboard"
              className="w-full border-0"
              style={{ height: '1200px' }}
              allow="fullscreen"
            />
          </div>
        ) : (
          <div className="section-card p-8 text-center">
            <Database className="w-12 h-12 text-outline-variant mx-auto mb-4" />
            <h3 className="text-lg font-semibold text-on-surface-variant mb-2">
              No Dashboard Generated Yet
            </h3>
            <p className="text-on-surface-variant mb-6 max-w-md mx-auto">
              Generate an interactive Grafana dashboard with 6 sections: Dataset Summary, Key Trends,
              Feature Insights, Model Performance, AI Explanation, and Alerts & Drift.
            </p>
            <button
              onClick={handleRefresh}
              disabled={generating}
              className="btn-primary"
            >
              {generating ? (
                <>
                  <Loader2 size={16} className="animate-spin" />
                  Generating…
                </>
              ) : (
                <>
                  <LayoutDashboard size={16} />
                  Generate Dashboard
                </>
              )}
            </button>
          </div>
        )}

        {/* Error banner (non-blocking) */}
        {error && embedUrl && (
          <div className="mt-4 rounded-xl p-4 flex items-start gap-3" style={{ background: 'rgba(245, 158, 11, 0.06)', border: '1px solid rgba(245, 158, 11, 0.15)' }}>
            <AlertCircle size={18} className="text-warning-400 mt-0.5 shrink-0" />
            <p className="text-sm text-warning-400">{error}</p>
          </div>
        )}
      </div>
    </div>
  );
};

// ── Reusable header ──────────────────────────────────────────────────
function Header({ onBack, children }) {
  return (
    <div className="flex items-center justify-between mb-8">
      <div>
        <h1 className="text-2xl font-bold text-on-surface flex items-center gap-3">
          <div className="w-12 h-12 bg-gradient-to-br from-orange-500 to-amber-600 rounded-xl flex items-center justify-center shadow-lg shadow-orange-500/20">
            <LayoutDashboard className="w-6 h-6 text-white" />
          </div>
          <div>
            <span>Grafana Dashboard</span>
            <p className="text-sm font-normal text-on-surface-variant mt-0.5">
              Dataset Summary · Key Trends · Feature Insights · Model Performance · AI Explanation · Alerts
            </p>
          </div>
        </h1>
      </div>
      <div className="flex items-center gap-3">
        {children}
        <button onClick={onBack} className="btn-secondary">
          <ArrowLeft size={18} />
          Back
        </button>
      </div>
    </div>
  );
}

export default GrafanaDashboard;
