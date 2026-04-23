import { useState } from "react";
import "./App.css";

const API = "http://localhost:8000";

function Badge({ label }) {
  const fake = label === "yes";
  return (
    <span className={`badge ${fake ? "badge-fake" : "badge-real"}`}>
      {fake ? "⚠ FAKE" : "✓ REAL"}
    </span>
  );
}

export default function App() {
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  const isUrl = /^https?:\/\//i.test(input.trim());

  async function handleSubmit(e) {
    e.preventDefault();
    if (!input.trim()) return;
    setLoading(true);
    setResult(null);
    setError("");
    try {
      const res = await fetch(`${API}/check`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ input: input.trim() }),
      });
      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.detail || "Server error");
      }
      setResult(await res.json());
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="page">
      <header className="hero-header">
        <h1>Fake News Detector</h1>
        <p className="subtitle">
          Paste a <strong>news statement</strong> or a <strong>URL</strong> — the AI searches multiple sources to verify it.
        </p>
      </header>

      <main className="main">
        <form onSubmit={handleSubmit} className="form">
          <div className="input-wrap">
            <textarea
              className="input"
              rows={4}
              placeholder="Enter a news headline / article text   OR   paste a URL starting with https://..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
            />
            {input.trim() && (
              <span className={`type-pill ${isUrl ? "pill-url" : "pill-text"}`}>
                {isUrl ? "🔗 URL" : "📝 Text"}
              </span>
            )}
          </div>

          <button className="btn" type="submit" disabled={loading || !input.trim()}>
            {loading ? "Analysing…" : "Check News"}
          </button>
        </form>

        {loading && (
          <div className="status-box">
            <div className="spinner" />
            <p>Searching &amp; verifying across sources — this can take 20–40 s…</p>
          </div>
        )}

        {error && <div className="error-box">{error}</div>}

        {result && (
          <div className={`result-card ${result.label === "yes" ? "card-fake" : "card-real"}`}>
            <div className="result-top">
              <Badge label={result.label} />
              <span className="input-type-tag">
                {result.input_type === "url" ? "🔗 Article verified" : "📝 Statement verified"}
              </span>
            </div>

            <section className="result-section">
              <h3>Reason</h3>
              <p>{result.reason}</p>
            </section>

            <section className="result-section">
              <h3>Evidence Links</h3>
              {result.evidence_links && result.evidence_links.length > 0 ? (
                <ul className="link-list">
                  {result.evidence_links.map((url, i) => (
                    <li key={i}>
                      <a href={url} target="_blank" rel="noreferrer">{url}</a>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="muted">No evidence links returned.</p>
              )}
            </section>
          </div>
        )}
      </main>
    </div>
  );
}
