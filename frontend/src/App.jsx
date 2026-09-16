import React, { useEffect, useRef, useState } from 'react';
import './App.css';

/* Backend base URL: local dev defaults to localhost; production (Vercel) sets
   VITE_API_BASE to the Render backend URL, e.g. https://escalation-rag-backend.onrender.com */
const API_BASE = (import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8000').replace(/\/$/, '');

const EXAMPLE_QUERIES = [
  'Coating area thickness not as per drawing size',
  'DC copy not received for dispatched material',
  'Material ready for dispatch but pending approval',
  'Wrong material received from supplier',
];

/* ---------- Inline SVG icons ---------- */
const BoltIcon = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
    <path d="M13 2L4.5 13.5H11L9.5 22L19.5 9.5H12.5L13 2Z" />
  </svg>
);

const SearchIcon = ({ className = 'search-icon' }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="11" cy="11" r="8" />
    <line x1="21" y1="21" x2="16.65" y2="16.65" />
  </svg>
);

const AlertIcon = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="12" cy="12" r="10" />
    <line x1="12" y1="8" x2="12" y2="12" />
    <line x1="12" y1="16" x2="12.01" y2="16" />
  </svg>
);

const CheckIcon = () => (
  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M9 12l2 2 4-4" />
    <path d="M12 2a10 10 0 1 0 10 10" />
  </svg>
);

const CaseIcon = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
    <polyline points="14 2 14 8 20 8" />
    <line x1="16" y1="13" x2="8" y2="13" />
    <line x1="16" y1="17" x2="8" y2="17" />
  </svg>
);

const SaveIcon = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z" />
    <polyline points="17 21 17 13 7 13 7 21" />
    <polyline points="7 3 7 8 15 8" />
  </svg>
);

const RefreshIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="23 4 23 10 17 10" />
    <polyline points="1 20 1 14 7 14" />
    <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15" />
  </svg>
);

const ChevronIcon = ({ open }) => (
  <svg className={`chevron ${open ? 'open' : ''}`} width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="6 9 12 15 18 9" />
  </svg>
);

const EmptyIcon = () => (
  <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="11" cy="11" r="8" />
    <path d="M21 21l-4.35-4.35" />
    <line x1="8" y1="11" x2="14" y2="11" />
  </svg>
);

/* ---------- Helpers ---------- */
function getSimilarityBadge(similarity) {
  if (similarity == null) return 'low';
  const s = typeof similarity === 'number' ? similarity : parseFloat(similarity);
  if (s >= 0.62) return 'high';
  if (s >= 0.4) return 'medium';
  return 'low';
}

function formatSimilarity(similarity) {
  if (similarity == null) return '—';
  const s = typeof similarity === 'number' ? similarity : parseFloat(similarity);
  return `${Math.round(s * 100)}% match`;
}

function displayValue(val) {
  return val !== undefined && val !== null && String(val).trim() !== '' ? val : '—';
}

/* Split the AI's combined solution text into a Solution section and an
   Effort/Action section. The model is prompted to output both sections. */
function splitSolution(text) {
  if (!text) return { solution: '', effort: '' };
  const lower = text.toLowerCase();
  const effortIndex = lower.indexOf('recommended effort');
  const actionIndex = lower.indexOf('recommended effort/action');
  const idx = actionIndex !== -1 ? actionIndex : effortIndex;
  if (idx === -1) {
    return { solution: text.trim(), effort: '' };
  }
  return {
    solution: text.slice(0, idx).replace(/recommended solution:?\s*/i, '').trim(),
    effort: text.slice(idx).trim(),
  };
}

/* Coalesce high-frequency stream updates into ONE state update per animation
   frame.

   The backend emits one SSE event per AI token (~150 events for a short
   answer) and each event used to trigger its own `setResult`, i.e. ~150 React
   re-renders per search. Buffering the tokens and flushing once per frame
   keeps the text visually smooth with only a handful of renders. */
function createFrameBatcher(onFlush) {
  let pending = '';
  let frame = null;

  const run = () => {
    frame = null;
    const text = pending;
    pending = '';
    if (text) onFlush(text);
  };

  return {
    push(text) {
      pending += text;
      if (frame === null) frame = requestAnimationFrame(run);
    },
    /* Emit anything still buffered right now. */
    flush() {
      if (frame !== null) {
        cancelAnimationFrame(frame);
        frame = null;
      }
      const text = pending;
      pending = '';
      if (text) onFlush(text);
    },
    /* Drop anything buffered (used when the answer restarts). */
    reset() {
      if (frame !== null) {
        cancelAnimationFrame(frame);
        frame = null;
      }
      pending = '';
    },
  };
}

const TEXT_CORRECTIONS = {
  recieved: 'received',
  recieve: 'receive',
  disaprch: 'dispatch',
  disapatch: 'dispatch',
  dispath: 'dispatch',
  worang: 'wrong',
  thki: 'thickness',
  reqd: 'required',
  obs: 'observed',
  maching: 'machining',
  cutomer: 'customer',
  cutomers: 'customers',
  thier: 'their',
  clernce: 'clearance',
  clearence: 'clearance',
  seperation: 'separation',
  expidite: 'expedite',
  resloved: 'resolved',
};

/* Format problem/solution text into proper English for display. */
function cleanCause(val) {
  if (val === undefined || val === null) return '—';
  const normalized = String(val).replace(/\s+/g, ' ').trim();
  if (!normalized) return '—';

  let text = normalized;
  Object.keys(TEXT_CORRECTIONS).forEach((wrong) => {
    const pattern = new RegExp(`\\b${wrong}\\b`, 'gi');
    text = text.replace(pattern, TEXT_CORRECTIONS[wrong]);
  });

  text = text.replace(/\s+([,.;:!?])/g, '$1');
  text = text.replace(/([,;:])(?=[A-Za-z0-9])/g, '$1 ');
  text = text.replace(/\s+/g, ' ').trim();
  text = text
    .split(/(?<=[.!?])\s+/)
    .map((s) => (s ? s.charAt(0).toUpperCase() + s.slice(1) : s))
    .join(' ');

  return text;
}

/* ---------- App Component ---------- */
export default function App() {
  const [cause, setCause] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [similarOpen, setSimilarOpen] = useState(true);
  const [saveOpen, setSaveOpen] = useState(false);
  const [saveForm, setSaveForm] = useState({ cause: '', prevention: '', effort: '', status: 'Open' });
  const [saving, setSaving] = useState(false);
  const [backendOnline, setBackendOnline] = useState(null);
  const [rebuilding, setRebuilding] = useState(false);
  const [toast, setToast] = useState(null);
  const similarRef = useRef(null);

  /* Live backend status indicator. */
  useEffect(() => {
    let active = true;
    const check = async () => {
      try {
        const response = await fetch(`${API_BASE}/health`);
        if (active) setBackendOnline(response.ok);
      } catch {
        if (active) setBackendOnline(false);
      }
    };
    check();
    const interval = setInterval(check, 30000);
    return () => {
      active = false;
      clearInterval(interval);
    };
  }, []);

  /* Auto-dismiss toasts. */
  useEffect(() => {
    if (!toast) return undefined;
    const timer = setTimeout(() => setToast(null), 4000);
    return () => clearTimeout(timer);
  }, [toast]);

  async function handleSearch() {
    if (!cause.trim()) return;
    setLoading(true);
    setError('');
    setResult(null);

    // Immediately show a streaming result card so the AI text appears progressively.
    setResult({
      recommended_solution: '',
      recommended_effort: '',
      escalation_id: '',
      timestamp: '',
      department: '',
      customer_name: '',
      material_name: '',
      kp_no: '',
      level: '',
      status: '',
      match_type: '',
      similar_cases: [],
      streaming: true,
    });

    try {
      const response = await fetch(`${API_BASE}/search/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ cause }),
      });
      if (!response.ok) {
        throw new Error(`Request failed: ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let solutionText = '';
      let aiTruncated = false;

      /* Render at most once per animation frame instead of once per token. */
      const paint = createFrameBatcher(() => {
        const parts = splitSolution(solutionText);
        setResult((prev) => ({
          ...prev,
          recommended_solution: parts.solution,
          recommended_effort: parts.effort,
          streaming: true,
        }));
      });

      // Read the SSE stream and accumulate the AI solution token by token.
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop();
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const payloadText = line.slice(6).trim();
            if (!payloadText) continue;
            try {
              const payload = JSON.parse(payloadText);
              if (payload.type === 'meta') {
                // First event: metadata + similar cases.
                setResult((prev) => ({
                  ...prev,
                  escalation_id: payload.escalation_id || '',
                  timestamp: payload.timestamp || '',
                  department: payload.department || '',
                  customer_name: payload.customer_name || '',
                  material_name: payload.material_name || '',
                  kp_no: payload.kp_no || '',
                  level: payload.level || '',
                  status: payload.status || '',
                  match_type: payload.match_type || '',
                  similar_cases: Array.isArray(payload.similar_cases) ? payload.similar_cases : [],
                  streaming: true,
                }));
              } else if (payload.type === 'ai_start') {
                // AI refinement begins — it replaces the instant data-driven text.
                solutionText = '';
                paint.reset();
                setResult((prev) => ({
                  ...prev,
                  recommended_solution: '',
                  recommended_effort: '',
                  streaming: true,
                }));
              } else if (payload.type === 'text') {
                solutionText += payload.content;
                paint.push(payload.content);
              } else if (payload.type === 'ai_end') {
                // Generation finished (or hit its deadline and was cut short).
                aiTruncated = Boolean(payload.truncated);
              }
              // "ai_skip" means the model produced nothing in time: keep the
              // instant data-driven answer that is already on screen.
            } catch (e) {
              // Ignore malformed payloads.
            }
          }
        }
      }

      // Stream finished — flush any pending tokens and finalize the result.
      paint.flush();
      const parts = splitSolution(solutionText);
      setResult((prev) => ({
        ...prev,
        recommended_solution: parts.solution,
        recommended_effort: parts.effort,
        streaming: false,
        ai_truncated: aiTruncated,
      }));
    } catch (err) {
      setError(err.message || 'Search failed');
      setResult(null);
    } finally {
      setLoading(false);
    }
  }

  async function handleSave(event) {
    event.preventDefault();
    if (!saveForm.cause.trim()) {
      setToast('Please describe the escalation cause.');
      return;
    }
    setSaving(true);
    try {
      const response = await fetch(`${API_BASE}/save`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          cause: saveForm.cause,
          prevention: saveForm.prevention,
          effort_taken: saveForm.effort,
          status: saveForm.status,
        }),
      });
      if (!response.ok) {
        const detail = await response.json().catch(() => ({}));
        throw new Error(detail.detail || `Save failed: ${response.status}`);
      }
      const payload = await response.json();
      setToast(`Saved as ${payload.escalation_id || 'new escalation'} — search index refreshed.`);
      setSaveForm({ cause: '', prevention: '', effort: '', status: 'Open' });
      setSaveOpen(false);
    } catch (err) {
      setToast(err.message || 'Save failed');
    } finally {
      setSaving(false);
    }
  }

  async function handleRebuild() {
    setRebuilding(true);
    try {
      const response = await fetch(`${API_BASE}/rebuild-index`, { method: 'POST' });
      if (!response.ok) throw new Error(`Rebuild failed: ${response.status}`);
      const payload = await response.json();
      setToast(`Index rebuilt — ${payload.records ?? '?'} records re-embedded.`);
    } catch (err) {
      setToast(err.message || 'Rebuild failed');
    } finally {
      setRebuilding(false);
    }
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter') handleSearch();
  }

  function openSavePanel() {
    setSaveForm((prev) => ({ ...prev, cause: prev.cause || cause }));
    setSaveOpen(true);
  }

  const similarCases = Array.isArray(result?.similar_cases) ? result.similar_cases : [];
  const hasResult = result !== null && !error;

  function scrollSimilar(direction) {
    const el = similarRef.current;
    if (!el) return;
    const card = el.querySelector('.similar-card');
    const step = card ? card.offsetWidth + 14 : 320;
    el.scrollBy({ left: direction * step, behavior: 'smooth' });
  }

  return (
    <div className="app">
      <nav className="topbar">
        <div className="brand">
          <span className="brand-mark"><BoltIcon /></span>
          <span className="brand-name">Escalation <b>Assistant</b></span>
        </div>
        <div className="topbar-actions">
          <span className={`status-pill ${backendOnline === null ? 'checking' : backendOnline ? 'online' : 'offline'}`}>
            <span className="status-dot" />
            {backendOnline === null ? 'Checking…' : backendOnline ? 'Backend online' : 'Backend offline'}
          </span>
          <button className="btn-ghost" onClick={handleRebuild} disabled={rebuilding} title="Re-embed all records from the data file">
            <RefreshIcon />
            {rebuilding ? 'Rebuilding…' : 'Rebuild Index'}
          </button>
        </div>
      </nav>

      <div className="app-inner">
        {/* ---------- Hero ---------- */}
        <header className="hero">
          <h1>Escalation Management<br />Knowledge Assistant</h1>
          <p>
            Search past escalation causes and instantly retrieve the real solution,
            the full action trail across every escalation level, and the closest
            historical cases — fetched accurately from the escalation log.
          </p>
          <div className="stat-row">
            <span className="stat-pill"><b>Hybrid</b> Semantic + Keyword Search</span>
            <span className="stat-pill"><b>RAG</b> Powered Answers</span>
            <span className="stat-pill"><b>FAISS</b> Vector Index</span>
          </div>
        </header>

        {/* ---------- Search ---------- */}
        <section className="search-card">
          <div className="search-label">
            <SearchIcon /> Describe the escalation cause
          </div>
          <div className="search-row">
            <div className="search-input-wrap">
              <SearchIcon />
              <input
                className="search-input"
                value={cause}
                onChange={(e) => setCause(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="e.g. Coating area thickness not as per drawing size…"
                disabled={loading}
              />
            </div>
            <button className="btn-search" onClick={handleSearch} disabled={loading || !cause.trim()}>
              {loading && <span className="spinner" />}
              {loading ? 'Searching…' : 'Search'}
            </button>
          </div>
          <div className="example-row">
            <span className="example-label">Try:</span>
            {EXAMPLE_QUERIES.map((query) => (
              <button key={query} className="example-chip" onClick={() => setCause(query)} disabled={loading}>
                {query}
              </button>
            ))}
          </div>
        </section>

        {/* ---------- Error ---------- */}
        {error && (
          <div className="error-banner">
            <AlertIcon />
            <span>{error}</span>
          </div>
        )}

        {/* ---------- Loading skeleton ---------- */}
        {loading && (
          <div className="results">
            <div className="reco-card skeleton-card">
              <div className="skeleton-line wide" />
              <div className="skeleton-panels">
                <div className="skeleton-line tall" />
                <div className="skeleton-line tall" />
              </div>
              <div className="skeleton-line" />
              <div className="skeleton-line short" />
            </div>
            <div className="similar-section">
              <div className="skeleton-line medium" />
              <div className="similar-grid-skeleton">
                <div className="skeleton-card small" />
                <div className="skeleton-card small" />
                <div className="skeleton-card small" />
              </div>
            </div>
          </div>
        )}


        {/* ---------- Results ---------- */}
        {hasResult && (
          <div className="results">
            {/* Recommended Suggestion */}
            <div className="reco-card">
              <div className="reco-head">
                <CheckIcon />
                <h2>Recommended Suggestion</h2>
                {result.match_type === 'exact' && (
                  <span className="match-badge exact">
                    {result.streaming ? 'Instant historical match' : 'Exact match'}
                  </span>
                )}
                {result.match_type === 'generalized' && result.streaming && (
                  <span className="streaming-badge">
                    <span className="spinner" /> AI enhancing…
                  </span>
                )}
                {result.match_type === 'generalized' && !result.streaming && (
                  <span className="match-badge generalized">AI-generalized</span>
                )}
                {result.streaming && !result.match_type && (
                  <span className="streaming-badge">
                    <span className="spinner" /> Searching…
                  </span>
                )}
              </div>
              <div className="reco-panels">
                <div className="reco-panel">
                  <span className="reco-panel-label">Solution</span>
                  <div className="reco-panel-value">
                    {result.recommended_solution || '…'}
                  </div>
                </div>
                <div className="reco-panel">
                  <span className="reco-panel-label">Effort / Action</span>
                  <div className="reco-panel-value">
                    {result.recommended_effort || '…'}
                  </div>
                </div>
              </div>
              <div className="reco-grid">
                <div className="reco-item">
                  <span className="reco-item-label">Escalation ID</span>
                  <span className="reco-item-value">
                    {displayValue(result.escalation_id) !== '—' ? (
                      <span className="tag tag-id">{result.escalation_id}</span>
                    ) : '—'}
                  </span>
                </div>
                <div className="reco-item">
                  <span className="reco-item-label">Department</span>
                  <span className="reco-item-value">
                    {displayValue(result.department) !== '—' ? (
                      <span className="tag tag-dept">{result.department}</span>
                    ) : '—'}
                  </span>
                </div>
                <div className="reco-item">
                  <span className="reco-item-label">Customer</span>
                  <span className="reco-item-value">{displayValue(result.customer_name)}</span>
                </div>
                <div className="reco-item">
                  <span className="reco-item-label">Material</span>
                  <span className="reco-item-value">{displayValue(result.material_name)}</span>
                </div>
                <div className="reco-item">
                  <span className="reco-item-label">KP No.</span>
                  <span className="reco-item-value">{displayValue(result.kp_no)}</span>
                </div>
                <div className="reco-item">
                  <span className="reco-item-label">Levels</span>
                  <span className="reco-item-value">{displayValue(result.level)}</span>
                </div>
                <div className="reco-item">
                  <span className="reco-item-label">Status</span>
                  <span className="reco-item-value">{displayValue(result.status)}</span>
                </div>
                <div className="reco-item">
                  <span className="reco-item-label">Last Update</span>
                  <span className="reco-item-value">{displayValue(result.timestamp)}</span>
                </div>
              </div>
            </div>


            {/* Similar Cases */}
            {similarCases.length > 0 && (
              <div className="similar-section">
                <div className="similar-header" onClick={() => setSimilarOpen((o) => !o)}>
                  <h3>
                    <CaseIcon /> Similar Cases
                    <span>{similarCases.length}</span>
                  </h3>
                  <ChevronIcon open={similarOpen} />
                </div>

                {similarOpen && (
                  <div className="similar-carousel">
                    <button className="similar-arrow similar-arrow-left" onClick={() => scrollSimilar(-1)} aria-label="Previous similar cases">
                      <span>‹</span>
                    </button>
                    <div className="similar-grid" ref={similarRef}>
                      {similarCases.map((item, index) => (
                        <div className="similar-card" key={index}>
                          <div className="similar-card-top">
                            <span className="similar-card-title">{cleanCause(item.cause)}</span>
                            <span className={`similarity-badge ${getSimilarityBadge(item.similarity)}`}>
                              {formatSimilarity(item.similarity)}
                            </span>
                          </div>
                          <div className="similar-card-body">
                            <div className="similar-card-detail">
                              <span className="similar-card-label">Solution</span>
                              <span className="similar-card-value">{cleanCause(item.solution)}</span>
                            </div>
                            <div className="similar-card-detail">
                              <span className="similar-card-label">Effort</span>
                              <span className="similar-card-value">{cleanCause(item.effort_taken)}</span>
                            </div>
                          </div>
                          <div className="similar-card-tags">
                            {item.escalation_id && <span className="tag tag-id">{item.escalation_id}</span>}
                            {item.department && <span className="tag tag-dept">{item.department}</span>}
                            {item.customer_name && item.customer_name !== 'Not Found' && (
                              <span className="tag">{item.customer_name}</span>
                            )}
                            {item.material_name && item.material_name !== 'Not Found' && (
                              <span className="tag">{item.material_name}</span>
                            )}
                            {item.level && <span className="tag tag-level">{item.level}</span>}
                          </div>
                        </div>
                      ))}
                    </div>
                    <button className="similar-arrow similar-arrow-right" onClick={() => scrollSimilar(1)} aria-label="Next similar cases">
                      <span>›</span>
                    </button>
                  </div>
                )}
              </div>
            )}
          </div>
        )}


        {/* ---------- Empty state ---------- */}
        {!hasResult && !loading && !error && (
          <div className="empty-state">
            <EmptyIcon />
            <h3>No results yet</h3>
            <p>Type an escalation cause above or pick an example, then press Search.</p>
          </div>
        )}

        {/* ---------- Save a new escalation ---------- */}
        <section className="save-section">
          <button className="save-toggle" onClick={() => (saveOpen ? setSaveOpen(false) : openSavePanel())}>
            <SaveIcon />
            {saveOpen ? 'Close new escalation form' : 'Log a new escalation'}
            <ChevronIcon open={saveOpen} />
          </button>

          {saveOpen && (
            <form className="save-form" onSubmit={handleSave}>
              <div className="save-field">
                <label htmlFor="save-cause">Cause / Problem description *</label>
                <textarea
                  id="save-cause"
                  rows={2}
                  value={saveForm.cause}
                  onChange={(e) => setSaveForm((prev) => ({ ...prev, cause: e.target.value }))}
                  placeholder="What went wrong?"
                />
              </div>
              <div className="save-field">
                <label htmlFor="save-prevention">Prevention / Final resolution</label>
                <textarea
                  id="save-prevention"
                  rows={2}
                  value={saveForm.prevention}
                  onChange={(e) => setSaveForm((prev) => ({ ...prev, prevention: e.target.value }))}
                  placeholder="The fix that prevents recurrence (stored in the Solution column)"
                />
              </div>
              <div className="save-row">
                <div className="save-field grow">
                  <label htmlFor="save-effort">Effort / Actions taken</label>
                  <input
                    id="save-effort"
                    value={saveForm.effort}
                    onChange={(e) => setSaveForm((prev) => ({ ...prev, effort: e.target.value }))}
                    placeholder="e.g. Informed customer via mail and confirmed"
                  />
                </div>
                <div className="save-field">
                  <label htmlFor="save-status">Status</label>
                  <select
                    id="save-status"
                    value={saveForm.status}
                    onChange={(e) => setSaveForm((prev) => ({ ...prev, status: e.target.value }))}
                  >
                    <option value="Open">Open</option>
                    <option value="Close">Close</option>
                  </select>
                </div>
                <button type="submit" className="btn-search btn-save" disabled={saving}>
                  {saving && <span className="spinner" />}
                  {saving ? 'Saving…' : 'Save'}
                </button>
              </div>
              <p className="save-hint">
                The record is appended to the escalation log with the next escalation ID,
                and the search index is refreshed automatically.
              </p>
            </form>
          )}
        </section>

        {/* ---------- Footer ---------- */}
        <footer className="footer">
          Escalation Management Knowledge Assistant &mdash; React · FastAPI · FAISS · Hybrid Retrieval
        </footer>
      </div>

      {/* ---------- Toast ---------- */}
      {toast && (
        <div className="toast">
          <CheckIcon />
          <span>{toast}</span>
        </div>
      )}
    </div>
  );
}
