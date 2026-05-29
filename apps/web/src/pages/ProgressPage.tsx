import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { useAuth, getUserId } from "../lib/auth";
import { fetchMastery, fetchProgress, Mastery, ProgressSummary } from "../lib/api";
import {
  clearRecognitionFeedback,
  downloadText,
  readRecognitionFeedback,
  recognitionFeedbackCsv,
  summarizeRecognitionFeedback,
  type RecognitionFeedbackEntry,
} from "../lib/recognitionFeedback";

type LoadState = "loading" | "loaded" | "error";

function outcomeLabel(outcome: string) {
  if (outcome === "pass") return "pass";
  if (outcome === "retry") return "needs practice";
  return outcome;
}

export default function ProgressPage() {
  const auth = useAuth();
  const userId = getUserId(auth);
  const [summary, setSummary] = useState<ProgressSummary | null>(null);
  const [mastery, setMastery] = useState<Mastery[]>([]);
  const [state, setState] = useState<LoadState>("loading");
  const [error, setError] = useState("");
  const [recognitionFeedback, setRecognitionFeedback] = useState<RecognitionFeedbackEntry[]>([]);

  useEffect(() => {
    let cancelled = false;
    setState("loading");
    Promise.all([
      fetchProgress(userId, auth.session?.access_token),
      fetchMastery(userId, auth.session?.access_token),
    ])
      .then(([s, m]) => {
        if (cancelled) return;
        setSummary(s);
        setMastery(m);
        setState("loaded");
      })
      .catch((e) => {
        if (cancelled) return;
        setError((e as Error).message);
        setState("error");
      });
    return () => {
      cancelled = true;
    };
  }, [userId, auth.session]);

  useEffect(() => {
    setRecognitionFeedback(readRecognitionFeedback());
  }, []);

  // Sort mastery: mastered first, then by total_attempts desc.
  const sortedMastery = [...mastery].sort((a, b) => {
    if (a.mastered !== b.mastered) return a.mastered ? -1 : 1;
    return b.total_attempts - a.total_attempts;
  });

  const noProgressYet =
    state === "loaded" &&
    summary !== null &&
    summary.total_attempts === 0 &&
    mastery.length === 0;
  const feedbackSummary = summarizeRecognitionFeedback(recognitionFeedback);
  const feedbackRows = Object.entries(feedbackSummary.bySign)
    .map(([signId, row]) => {
      const topPrediction =
        Object.entries(row.commonPredictions).sort((a, b) => b[1] - a[1])[0]?.[0] ?? "unknown";
      return { signId, ...row, topPrediction };
    })
    .sort((a, b) => b.rejected - a.rejected || b.total - a.total)
    .slice(0, 8);

  const exportFeedback = (format: "csv" | "json") => {
    if (format === "csv") {
      downloadText("recognition_feedback.csv", recognitionFeedbackCsv(recognitionFeedback), "text/csv");
      return;
    }
    downloadText(
      "recognition_feedback.json",
      JSON.stringify(recognitionFeedback, null, 2),
      "application/json"
    );
  };

  const resetFeedback = () => {
    clearRecognitionFeedback();
    setRecognitionFeedback([]);
  };

  return (
    <div className="container">
      <Link to="/lobby">← Lobby</Link>
      <h1>Your Progress</h1>

      {state === "loading" && (
        <div className="card">
          <p style={{ color: "var(--muted)" }}>Loading…</p>
        </div>
      )}

      {state === "error" && (
        <div className="card status-fail">
          <p>Failed to load progress: {error}</p>
          <p style={{ color: "var(--muted)", fontSize: "0.85rem" }}>
            Check that the API server is running and reachable.
          </p>
        </div>
      )}

      {noProgressYet && (
        <div className="card">
          <p>
            You haven't practiced yet. <Link to="/lobby">Start a Wave 1 session</Link> to log attempts.
          </p>
        </div>
      )}

      {state === "loaded" && summary && !noProgressYet && (
        <>
          <div className="card">
            <p style={{ fontSize: "1.1rem", marginBottom: "0.5rem" }}>
              <strong>{summary.mastered_count}</strong> signs mastered ·{" "}
              <strong>{summary.total_passes}</strong> passes /{" "}
              <strong>{summary.total_attempts}</strong> attempts
              {summary.total_attempts > 0 && (
                <span style={{ color: "var(--muted)" }}>
                  {" "}
                  ({((summary.total_passes / summary.total_attempts) * 100).toFixed(0)}% pass rate)
                </span>
              )}
            </p>
            <Link to="/lobby" className="btn" style={{ display: "inline-block", marginTop: "0.5rem" }}>
              Practice more
            </Link>
          </div>

          {sortedMastery.length > 0 && (
            <>
              <h2>Per-sign mastery</h2>
              <div className="card" style={{ padding: 0 }}>
                <table style={{ width: "100%", borderCollapse: "collapse" }}>
                  <thead>
                    <tr style={{ borderBottom: "1px solid var(--border)" }}>
                      <th style={{ textAlign: "left", padding: "0.6rem 1rem" }}>Sign</th>
                      <th style={{ textAlign: "left", padding: "0.6rem 1rem" }}>Mastered</th>
                      <th style={{ textAlign: "right", padding: "0.6rem 1rem" }}>Passes</th>
                      <th style={{ textAlign: "right", padding: "0.6rem 1rem" }}>Attempts</th>
                      <th style={{ textAlign: "right", padding: "0.6rem 1rem" }}>Streak</th>
                    </tr>
                  </thead>
                  <tbody>
                    {sortedMastery.map((m) => (
                      <tr key={m.sign_id} style={{ borderBottom: "1px solid var(--border)" }}>
                        <td style={{ padding: "0.5rem 1rem" }}>
                          <code>{m.sign_id}</code>
                        </td>
                        <td style={{ padding: "0.5rem 1rem" }}>
                          {m.mastered ? <span className="status-pass">✓</span> : <span style={{ color: "var(--muted)" }}>—</span>}
                        </td>
                        <td style={{ padding: "0.5rem 1rem", textAlign: "right" }}>
                          {m.total_passes}
                        </td>
                        <td style={{ padding: "0.5rem 1rem", textAlign: "right" }}>
                          {m.total_attempts}
                        </td>
                        <td style={{ padding: "0.5rem 1rem", textAlign: "right" }}>
                          {m.consecutive_passes}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          )}

          <h2>Recent attempts</h2>
          <ul style={{ listStyle: "none", padding: 0 }}>
            {summary.recent_attempts.length === 0 && (
              <li className="card" style={{ color: "var(--muted)" }}>
                No attempts yet. Start practicing to see your history here.
              </li>
            )}
            {summary.recent_attempts.map((a) => (
              <li key={a.id} className="card" style={{ marginBottom: "0.5rem" }}>
                <code>{a.sign_id}</code> —{" "}
                <span className={`status-${a.outcome}`}>{outcomeLabel(a.outcome)}</span>
                {a.confidence != null && (
                  <span style={{ color: "var(--muted)" }}> ({(a.confidence * 100).toFixed(0)}% confidence)</span>
                )}
                <span style={{ color: "var(--muted)", marginLeft: "0.5rem", fontSize: "0.85rem" }}>
                  {new Date(a.created_at).toLocaleString()}
                </span>
              </li>
            ))}
          </ul>

          <h2>Recognition feedback</h2>
          <div className="card">
            {feedbackSummary.total === 0 ? (
              <p style={{ color: "var(--muted)", margin: 0 }}>
                No model feedback yet. Use recognition mode and answer the correctness prompt after each result.
              </p>
            ) : (
              <>
                <p style={{ marginTop: 0 }}>
                  <strong>{feedbackSummary.total}</strong> labeled model results -{" "}
                  <span className="status-pass">{feedbackSummary.accepted} right</span> /{" "}
                  <span className="status-fail">{feedbackSummary.rejected} wrong</span>
                </p>
                {feedbackRows.length > 0 && (
                  <table className="compact-table">
                    <thead>
                      <tr>
                        <th>Sign</th>
                        <th>Votes</th>
                        <th>Wrong</th>
                        <th>Top prediction</th>
                      </tr>
                    </thead>
                    <tbody>
                      {feedbackRows.map((row) => (
                        <tr key={row.signId}>
                          <td>
                            <code>{row.signId}</code>
                          </td>
                          <td>{row.total}</td>
                          <td>{row.rejected}</td>
                          <td>
                            <code>{row.topPrediction}</code>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}
                <div className="button-row">
                  <button className="btn btn-secondary" onClick={() => exportFeedback("csv")}>
                    Export CSV
                  </button>
                  <button className="btn btn-secondary" onClick={() => exportFeedback("json")}>
                    Export JSON
                  </button>
                  <button className="btn btn-secondary" onClick={resetFeedback}>
                    Clear feedback
                  </button>
                </div>
              </>
            )}
          </div>
        </>
      )}
    </div>
  );
}
