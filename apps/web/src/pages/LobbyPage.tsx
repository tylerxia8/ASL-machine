import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { useAuth, getUserId } from "../lib/auth";
import { checkApiHealth, createSession, fetchProgress } from "../lib/api";

export default function LobbyPage() {
  const auth = useAuth();
  const userId = getUserId(auth);
  const [progress, setProgress] = useState<{ mastered_count: number; total_attempts: number } | null>(null);
  const [apiOk, setApiOk] = useState<boolean | null>(null);
  const [apiUrl, setApiUrl] = useState("");
  const [modelInfo, setModelInfo] = useState("");

  useEffect(() => {
    checkApiHealth().then((h) => {
      setApiOk(h.ok);
      setApiUrl(h.url);
    });
    fetch("/models/model_meta.json")
      .then((r) => r.json())
      .then((m) => setModelInfo(`${m.model_version} · ${m.num_classes} signs`))
      .catch(() => setModelInfo("model unknown"));
    fetchProgress(userId, auth.session?.access_token).then(setProgress).catch(() => setProgress(null));
  }, [userId, auth.session]);

  const startSession = async (wave: number) => {
    const sess = await createSession(userId, auth.session?.access_token);
    sessionStorage.setItem("practice_wave", String(wave));
    sessionStorage.setItem("practice_session_id", sess.id);
    sessionStorage.removeItem("session_log");
    window.location.href = "/practice";
  };

  return (
    <div className="container">
      <h1>Practice Lobby</h1>
      <div className="card" style={{ marginBottom: "1rem", fontSize: "0.9rem" }}>
        <span style={{ color: apiOk ? "var(--pass)" : apiOk === false ? "var(--fail)" : "var(--muted)" }}>
          API {apiOk ? "connected" : apiOk === false ? "offline" : "checking…"} ({apiUrl})
        </span>
        <span style={{ marginLeft: "1rem", color: "var(--muted)" }}>Model: {modelInfo}</span>
      </div>
      <div className="card" style={{ marginBottom: "1rem", borderColor: "var(--accent)" }}>
        <strong>Wave 1 track</strong>
        <ol style={{ margin: "0.5rem 0", paddingLeft: "1.25rem", color: "var(--muted)" }}>
          <li>
            <Link to="/capture">Record clips</Link> → <code>import-from-downloads.ps1</code>
          </li>
          <li>
            Double-click <code>scripts/continue-wave1.cmd</code> to retrain
          </li>
          <li>
            <Link to="/dry-run">Run dry run checklist</Link>
          </li>
        </ol>
      </div>
      {progress && (
        <div className="card" style={{ marginBottom: "1rem" }}>
          <strong>Your progress</strong>
          <p>
            Mastered: {progress.mastered_count} · Attempts: {progress.total_attempts}
          </p>
          <Link to="/progress">View full history →</Link>
        </div>
      )}
      <div style={{ display: "grid", gap: "1rem", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))" }}>
        <div className="card" style={{ outline: "2px solid var(--accent)" }}>
          <h3>Wave 1 — Recommended</h3>
          <p>25 trained signs: greetings, questions, numbers 1–5, common verbs</p>
          <button className="btn" onClick={() => startSession(1)} disabled={apiOk === false}>
            Start Wave 1 session
          </button>
        </div>
        <div className="card">
          <h3>Full catalog</h3>
          <p>All 100+ signs (untrained signs show as reference only)</p>
          <button className="btn btn-secondary" onClick={() => startSession(99)} disabled={apiOk === false}>
            Start session
          </button>
        </div>
      </div>
      <p className="footer-meta" style={{ marginTop: "2rem" }}>
        Web app URL may vary (e.g. :5176). API should match <code>apps/web/.env.local</code>.
      </p>
    </div>
  );
}
