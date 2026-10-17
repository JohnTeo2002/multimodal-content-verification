import React, { useState } from 'react';
import './App.css';

interface ClaimCheckResult {
  verdict: 'supported' | 'uncertain' | 'unverifiable';
  confidence: number;
  cited_evidence: string[];
  evidence: Record<string, string>;
}

interface HumanResponseResult {
  trust: number;
  trust_level: string;
  impact: number;
  impact_level: string;
}

interface VerificationResponse {
  claim_check: ClaimCheckResult;
  human_response?: HumanResponseResult;
}

export default function App() {
  const [claim, setClaim] = useState("The photo was taken at the summit");
  const [evidenceList, setEvidenceList] = useState<string[]>([
    "Summit photo taken Monday",
    "Photo taken at dawn",
  ]);
  const [includeScores, setIncludeScores] = useState(true);
  const [aigc, setAigc] = useState(0.2);
  const [belief, setBelief] = useState(0.8);
  const [dissemination, setDissemination] = useState(0.8);

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<VerificationResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleAddEvidence = () => {
    setEvidenceList([...evidenceList, ""]);
  };

  const handleEvidenceChange = (index: number, value: string) => {
    const updated = [...evidenceList];
    updated[index] = value;
    setEvidenceList(updated);
  };

  const handleRemoveEvidence = (index: number) => {
    setEvidenceList(evidenceList.filter((_, i) => i !== index));
  };

  const loadPreset = (type: 'summit' | 'unrelated' | 'viral') => {
    if (type === 'summit') {
      setClaim("The photo was taken at the summit");
      setEvidenceList(["Summit photo taken Monday", "Photo taken at dawn"]);
      setIncludeScores(true);
      setAigc(0.2);
      setBelief(0.8);
      setDissemination(0.8);
    } else if (type === 'unrelated') {
      setClaim("The artifact was discovered underwater in 1900");
      setEvidenceList(["Heavy rainfall reported downtown", "Traffic accident on highway"]);
      setIncludeScores(true);
      setAigc(0.8);
      setBelief(0.4);
      setDissemination(0.3);
    } else if (type === 'viral') {
      setClaim("Breaking news about quantum computing breakthrough");
      setEvidenceList(["Quantum processor announced", "Computing milestone reached in lab"]);
      setIncludeScores(true);
      setAigc(0.9);
      setBelief(0.9);
      setDissemination(0.9);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    const filteredEvidence = evidenceList.filter((ev) => ev.trim() !== "");

    const payload: any = {
      claim,
      evidence: filteredEvidence,
    };

    if (includeScores) {
      payload.scores = [Number(aigc), Number(belief), Number(dissemination)];
    }

    try {
      const response = await fetch('http://localhost:8000/api/verify', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || 'Failed to verify content');
      }

      const data = await response.json();
      setResult(data);
    } catch (err: any) {
      setError(err.message || 'Network error connecting to backend API');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <header className="app-header">
        <div className="header-badge">Content Verification Studio</div>
        <h1>Content Verification & Impact Dashboard</h1>
        <p>Evaluate claims against text evidence and analyze trust & viral impact metrics.</p>
      </header>

      <div className="presets-bar">
        <span>Quick Presets:</span>
        <button type="button" onClick={() => loadPreset('summit')} className="preset-btn">Summit Claim (Supported)</button>
        <button type="button" onClick={() => loadPreset('unrelated')} className="preset-btn">Unrelated Evidence</button>
        <button type="button" onClick={() => loadPreset('viral')} className="preset-btn">Viral AI Content (Suspicious)</button>
      </div>

      <div className="main-layout">
        <div className="form-card">
          <h2>Input Parameters</h2>
          <form onSubmit={handleSubmit}>
            <div className="form-group">
              <label htmlFor="claim-input">Claim Text (1 - 500 chars)</label>
              <textarea
                id="claim-input"
                value={claim}
                onChange={(e) => setClaim(e.target.value)}
                maxLength={500}
                rows={3}
                required
              />
              <span className="char-count">{claim.length}/500</span>
            </div>

            <div className="form-group">
              <div className="evidence-header">
                <label>Evidence Snippets</label>
                <button type="button" onClick={handleAddEvidence} className="secondary-btn">
                  + Add Evidence
                </button>
              </div>
              {evidenceList.map((ev, index) => (
                <div key={index} className="evidence-row">
                  <input
                    type="text"
                    value={ev}
                    onChange={(e) => handleEvidenceChange(index, e.target.value)}
                    placeholder={`Evidence snippet #${index + 1}`}
                  />
                  <button
                    type="button"
                    onClick={() => handleRemoveEvidence(index)}
                    className="danger-btn"
                    title="Remove evidence"
                  >
                    ×
                  </button>
                </div>
              ))}
            </div>

            <div className="form-group checkbox-group">
              <label>
                <input
                  type="checkbox"
                  checked={includeScores}
                  onChange={(e) => setIncludeScores(e.target.checked)}
                />
                Include Human Response Scores
              </label>
            </div>

            {includeScores && (
              <div className="scores-section">
                <h3>Human Response Metrics (0.0 to 1.0)</h3>
                
                <div className="slider-control">
                  <div className="slider-label">
                    <span>AIGC Score (AI-Generated Appearance):</span>
                    <strong>{Number(aigc).toFixed(2)}</strong>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.05"
                    value={aigc}
                    onChange={(e) => setAigc(parseFloat(e.target.value))}
                  />
                </div>

                <div className="slider-control">
                  <div className="slider-label">
                    <span>Belief Score (Perceived Credibility):</span>
                    <strong>{Number(belief).toFixed(2)}</strong>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.05"
                    value={belief}
                    onChange={(e) => setBelief(parseFloat(e.target.value))}
                  />
                </div>

                <div className="slider-control">
                  <div className="slider-label">
                    <span>Dissemination Score (Sharing Likelihood):</span>
                    <strong>{Number(dissemination).toFixed(2)}</strong>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.05"
                    value={dissemination}
                    onChange={(e) => setDissemination(parseFloat(e.target.value))}
                  />
                </div>
              </div>
            )}

            <button type="submit" className="primary-btn" disabled={loading}>
              {loading ? 'Analyzing Content...' : 'Run Analysis'}
            </button>
          </form>
        </div>

        <div className="results-card">
          <h2>Analysis Results</h2>
          {error && <div className="error-banner">{error}</div>}
          {!result && !error && !loading && (
            <div className="placeholder-result">
              <p>Configure your claim and evidence on the left and click <strong>"Run Analysis"</strong> to view verification metrics.</p>
            </div>
          )}
          {loading && (
            <div className="loading-state">
              <div className="spinner"></div>
              <p>Evaluating claim and calculating metrics...</p>
            </div>
          )}
          {result && (
            <div className="results-content">
              <div className="result-section">
                <h3>Claim Verification</h3>
                <div className="metric-row">
                  <span className="metric-label">Verdict:</span>
                  <span className={`badge verdict-${result.claim_check.verdict}`}>
                    {result.claim_check.verdict.toUpperCase()}
                  </span>
                </div>
                <div className="metric-row">
                  <span className="metric-label">Confidence:</span>
                  <div className="progress-bar-container">
                    <div
                      className="progress-bar-fill"
                      style={{ width: `${result.claim_check.confidence * 100}%` }}
                    ></div>
                    <span className="progress-text">
                      {(result.claim_check.confidence * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>

                <div className="evidence-summary">
                  <h4>Cited Evidence Sources:</h4>
                  {result.claim_check.cited_evidence.length === 0 ? (
                    <p className="no-cited">No evidence matched keywords in the claim.</p>
                  ) : (
                    <ul>
                      {result.claim_check.cited_evidence.map((id) => (
                        <li key={id}>
                          <code>{id}</code>: {result.claim_check.evidence[id]}
                        </li>
                      ))}
                    </ul>
                  )}
                </div>
              </div>

              {result.human_response && (
                <div className="result-section">
                  <h3>Human Response & Impact</h3>
                  <div className="metric-grid">
                    <div className="metric-box">
                      <span className="box-title">Trust Score</span>
                      <span className="box-value">{result.human_response.trust.toFixed(2)}</span>
                      <span className={`badge level-${result.human_response.trust_level}`}>
                        {result.human_response.trust_level}
                      </span>
                    </div>
                    <div className="metric-box">
                      <span className="box-title">Impact Score</span>
                      <span className="box-value">{result.human_response.impact.toFixed(2)}</span>
                      <span className={`badge level-${result.human_response.impact_level}`}>
                        {result.human_response.impact_level}
                      </span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
