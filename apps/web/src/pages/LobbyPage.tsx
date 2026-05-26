import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { useAuth, getUserId } from "../lib/auth";
import { checkApiHealth, createSession, fetchProgress } from "../lib/api";
import {
  BUNDLED_SOURCE,
  ModelSource,
  getSelectedSourceId,
  listReleaseSources,
  setSelectedSourceId,
} from "../lib/modelSource";

type ModelMeta = {
  model_version?: string;
  num_classes?: number;
  val_accuracy?: number;
};

export default function LobbyPage() {
  const auth = useAuth();
  const userId = getUserId(auth);
  const [progress, setProgress] = useState<{ mastered_count: number; total_attempts: number } | null>(null);
  const [apiOk, setApiOk] = useState<boolean | null>(null);
  const [apiUrl, setApiUrl] = useState("");
  const [modelInfo, setModelInfo] = useState("");
  const [modelWarning, setModelWarning] = useState("");
  const [sources, setSources] = useState<ModelSource[]>([BUNDLED_SOURCE]);
  const [selectedSourceId, setSelectedSourceIdState] = useState(getSelectedSourceId());

  useEffect(() => {
    checkApiHealth().then((h) => {
      setApiOk(h.ok);
      setApiUrl(h.url);
    });
    fetchProgress(userId, auth.session?.access_token).then(setProgress).catch(() => setProgress(null));
    listReleaseSources().then((rs) => setSources([BUNDLED_SOURCE, ...rs]));
  }, [userId, auth.session]);

  useEffect(() => {
    const src = sources.find((s) => s.id === selectedSourceId) || BUNDLED_SOURCE;
    fetch(src.metaUrl)
      .then((r) => (r.ok ? r.json() : null))
      .then((m: ModelMeta | null) => {
        if (!m) {
          setModelInfo("model unknown");
          setModelWarning("");
          return;
        }
        const accuracy =
          typeof m.val_accuracy === "number" ? ` - val ${(m.val_accuracy * 100).toFixed(1)}%` : "";
        setModelInfo(`${m.model_version || "unknown"} - ${m.num_classes || "?"} signs${accuracy}`);
        setModelWarning(
          typeof m.val_accuracy === "number" && m.val_accuracy < 0.5
            ? "Demo/integration model only. Validation accuracy is below the pilot-quality bar."
            : ""
        );
      })
      .catch(() => {
        setModelInfo("model unknown");
        setModelWarning("");
      });
  }, [selectedSourceId, sources]);

  const onModelChange = (id: string) => {
    setSelectedSourceId(id);
    setSelectedSourceIdState(id);
    // Force a reload so the inference module re-fetches labels + model from the
    // new source on next attempt. sessionStorage clears practice context too.
    sessionStorage.clear();
    window.location.reload();
  };

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
        <div>
          <span style={{ color: apiOk ? "var(--pass)" : apiOk === false ? "var(--fail)" : "var(--muted)" }}>
            API {apiOk ? "connected" : apiOk === false ? "offline" : "checking..."} ({apiUrl})
          </span>
          <span style={{ marginLeft: "1rem", color: "var(--muted)" }}>Model: {modelInfo}</span>
        </div>
        {modelWarning && (
          <p className="status-fail" style={{ margin: "0.5rem 0 0", fontSize: "0.85rem" }}>
            {modelWarning}
          </p>
        )}
        <div style={{ marginTop: "0.5rem", display: "flex", gap: "0.5rem", alignItems: "center" }}>
          <label htmlFor="model-source" style={{ color: "var(--muted)" }}>
            Source:
          </label>
          <select
            id="model-source"
            className="input"
            style={{ maxWidth: "20rem" }}
            value={selectedSourceId}
            onChange={(e) => onModelChange(e.target.value)}
          >
            {sources.map((s) => (
              <option key={s.id} value={s.id}>
                {s.label}
              </option>
            ))}
          </select>
          <span style={{ color: "var(--muted)", fontSize: "0.8rem" }}>(page will reload on change)</span>
        </div>
      </div>
      <div className="card" style={{ marginBottom: "1rem", borderColor: "var(--accent)" }}>
        <strong>Wave 1 track</strong>
        <ol style={{ margin: "0.5rem 0", paddingLeft: "1.25rem", color: "var(--muted)" }}>
          <li>
            <Link to="/capture">Record clips</Link> {"->"} <code>import-from-downloads.ps1</code>
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
            Mastered: {progress.mastered_count} - Attempts: {progress.total_attempts}
          </p>
          <Link to="/progress">View full history {"->"}</Link>
        </div>
      )}
      <div style={{ display: "grid", gap: "1rem", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))" }}>
        <div className="card" style={{ outline: "2px solid var(--accent)" }}>
          <h3>Wave 1 - Recommended</h3>
          <p>25 trained signs: greetings, questions, numbers 1-5, common verbs</p>
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
