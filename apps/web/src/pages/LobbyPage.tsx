import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { useAuth, getUserId } from "../lib/auth";
import { checkApiHealth, createSession, fetchMastery, fetchProgress, type Mastery, type ProgressSummary } from "../lib/api";
import { COURSE_UNITS, dueReviewSigns, unitProgress } from "../lib/coursePlan";
import { practiceStreak, recommendedUnit, todayMissions } from "../lib/learnerDashboard";
import {
  BUNDLED_SOURCE,
  ModelSource,
  getSelectedSourceId,
  listReleaseSources,
  setSelectedSourceId,
} from "../lib/modelSource";
import { loadRecognitionCalibration, reliabilityFor, type RecognitionCalibration } from "../lib/recognitionCalibration";
import { readRecognitionFeedback, summarizeRecognitionFeedback } from "../lib/recognitionFeedback";

type ModelMeta = {
  model_version?: string;
  num_classes?: number;
  val_accuracy?: number;
};

export default function LobbyPage() {
  const auth = useAuth();
  const userId = getUserId(auth);
  const [progress, setProgress] = useState<ProgressSummary | null>(null);
  const [mastery, setMastery] = useState<Mastery[]>([]);
  const [apiOk, setApiOk] = useState<boolean | null>(null);
  const [apiUrl, setApiUrl] = useState("");
  const [modelInfo, setModelInfo] = useState("");
  const [modelWarning, setModelWarning] = useState("");
  const [calibration, setCalibration] = useState<RecognitionCalibration | null>(null);
  const [sources, setSources] = useState<ModelSource[]>([BUNDLED_SOURCE]);
  const [selectedSourceId, setSelectedSourceIdState] = useState(getSelectedSourceId());

  useEffect(() => {
    checkApiHealth().then((h) => {
      setApiOk(h.ok);
      setApiUrl(h.url);
    });
    fetchProgress(userId, auth.session?.access_token).then(setProgress).catch(() => setProgress(null));
    fetchMastery(userId, auth.session?.access_token).then(setMastery).catch(() => setMastery([]));
    listReleaseSources().then((rs) => setSources([BUNDLED_SOURCE, ...rs]));
    loadRecognitionCalibration().then(setCalibration).catch(() => setCalibration(null));
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

  const startSession = async (wave: number, order?: "weak_first" | "confusions" | "shuffle" | "default") => {
    const sess = await createSession(userId, auth.session?.access_token);
    sessionStorage.setItem("practice_wave", String(wave));
    sessionStorage.setItem("practice_session_id", sess.id);
    sessionStorage.removeItem("session_log");
    localStorage.removeItem("practice_unit_filter");
    if (order) localStorage.setItem("practice_order", order);
    window.location.href = "/practice";
  };

  const thresholdRows = Object.entries(calibration?.thresholds ?? {}).map(([signId, row]) => ({
    signId,
    row,
    reliability: reliabilityFor(row),
  }));
  const modelHealth = {
    strong: thresholdRows.filter((r) => r.reliability === "strong").length,
    watch: thresholdRows.filter((r) => r.reliability === "watch").length,
    weak: thresholdRows.filter((r) => r.reliability === "weak").length,
  };
  const weakestSigns = thresholdRows
    .filter((r) => r.reliability !== "strong")
    .sort(
      (a, b) =>
        (a.row.f1 ?? 1) - (b.row.f1 ?? 1) ||
        (a.row.support ?? 999) - (b.row.support ?? 999) ||
        a.signId.localeCompare(b.signId)
    )
    .slice(0, 4);
  const feedbackSummary = summarizeRecognitionFeedback(readRecognitionFeedback());
  const dueSigns = dueReviewSigns(COURSE_UNITS, mastery, feedbackSummary);
  const streak = practiceStreak(mastery);
  const unit = recommendedUnit(COURSE_UNITS, mastery);
  const unitStats = unitProgress(unit, mastery);
  const missions = todayMissions({ summary: progress, feedback: feedbackSummary, dueCount: dueSigns.length });

  return (
    <div className="container wide-container">
      <section className="dashboard-hero">
        <div>
          <p className="eyebrow">Intro ASL study dashboard</p>
          <h1>Ready for today’s practice?</h1>
          <p>
            Start with a short review, learn your next unit, or jump into phrases for class.
          </p>
          <div className="button-row">
            <Link to="/learn" className="btn btn-large">
              Continue learning
            </Link>
            <button className="btn btn-secondary btn-large" onClick={() => startSession(1, "weak_first")} disabled={apiOk === false}>
              Quick review
            </button>
          </div>
        </div>
        <div className="hero-stats">
          <div>
            <span className="metric-label">Streak</span>
            <strong>{streak} day{streak === 1 ? "" : "s"}</strong>
          </div>
          <div>
            <span className="metric-label">Mastered</span>
            <strong>{progress?.mastered_count ?? 0}</strong>
          </div>
          <div>
            <span className="metric-label">Next unit</span>
            <strong>{unit.week}</strong>
          </div>
        </div>
      </section>

      <section className="dashboard-grid">
        <div className="card mission-panel">
          <div className="section-heading">
            <div>
              <p className="eyebrow">Today</p>
              <h2>Daily missions</h2>
            </div>
            <Link to="/progress">Progress</Link>
          </div>
          <div className="mission-list">
            {missions.map((mission) => (
              <Link key={mission.id} to={mission.href} className={`mission-card mission-${mission.tone}`}>
                <span>
                  <strong>{mission.title}</strong>
                  <small>{mission.detail}</small>
                </span>
                <span>{mission.cta}</span>
              </Link>
            ))}
          </div>
        </div>

        <div className="card next-unit-panel">
          <p className="eyebrow">Recommended path</p>
          <h2>{unit.title}</h2>
          <p>{unit.goal}</p>
          <div className="progress-track" aria-label={`${unitStats.mastered} of ${unitStats.total} mastered`}>
            <span style={{ width: `${Math.round(unitStats.pct * 100)}%` }} />
          </div>
          <p className="muted-small">
            {unitStats.mastered}/{unitStats.total} mastered - {unitStats.attempted} attempted
          </p>
          <div className="button-row">
            <Link to="/learn" className="btn">
              Open unit
            </Link>
            <button className="btn btn-secondary" onClick={() => startSession(1, "shuffle")} disabled={apiOk === false}>
              Quiz me
            </button>
          </div>
        </div>
      </section>

      <section>
        <div className="section-heading">
          <div>
            <p className="eyebrow">Study modes</p>
            <h2>Choose how you want to practice</h2>
          </div>
        </div>
        <div className="mode-grid">
          <div className="mode-card">
            <span className="mode-icon">1</span>
            <h3>Guided lessons</h3>
            <p>Weekly units, flashcards, phrases, and culture notes.</p>
            <Link to="/learn" className="btn btn-secondary">Learn</Link>
          </div>
          <div className="mode-card">
            <span className="mode-icon">2</span>
            <h3>Recognition practice</h3>
            <p>Use the camera for coached sign attempts and feedback.</p>
            <button className="btn btn-secondary" onClick={() => startSession(1)} disabled={apiOk === false}>
              Practice
            </button>
          </div>
          <div className="mode-card">
            <span className="mode-icon">3</span>
            <h3>Phrases</h3>
            <p>Build short sequences for intro ASL conversations.</p>
            <Link to="/phrases" className="btn btn-secondary">Phrases</Link>
          </div>
          <div className="mode-card">
            <span className="mode-icon">4</span>
            <h3>Confusion drill</h3>
            <p>Focus on signs the model or learner commonly mixes up.</p>
            <button className="btn btn-secondary" onClick={() => startSession(1, "confusions")} disabled={apiOk === false}>
              Drill
            </button>
          </div>
        </div>
      </section>

      <details className="technical-panel">
        <summary>Model and project tools</summary>
        <div className="card">
          <p className="muted-small">
            API {apiOk ? "connected" : apiOk === false ? "offline" : "checking"} ({apiUrl}) - Model: {modelInfo}
          </p>
          {modelWarning && <p className="status-fail">{modelWarning}</p>}
          <label htmlFor="model-source" className="metric-label">
            Model source
          </label>
          <select
            id="model-source"
            className="input"
            value={selectedSourceId}
            onChange={(e) => onModelChange(e.target.value)}
          >
            {sources.map((s) => (
              <option key={s.id} value={s.id}>
                {s.label}
              </option>
            ))}
          </select>
          <div className="metric-grid">
            <div>
              <span className="metric-label">Strong</span>
              <strong>{modelHealth.strong}</strong>
            </div>
            <div>
              <span className="metric-label">Watch</span>
              <strong>{modelHealth.watch}</strong>
            </div>
            <div>
              <span className="metric-label">Weak</span>
              <strong>{modelHealth.weak}</strong>
            </div>
            <div>
              <span className="metric-label">Lowest F1</span>
              <strong>{weakestSigns[0]?.signId ?? "none"}</strong>
            </div>
          </div>
          <div className="button-row">
            <Link to="/capture" className="btn btn-secondary">Capture clips</Link>
            <Link to="/review-captures" className="btn btn-secondary">Review captures</Link>
            <Link to="/dry-run" className="btn btn-secondary">Dry run</Link>
            <Link to="/model-health" className="btn btn-secondary">Model health</Link>
          </div>
        </div>
      </details>
      <p className="footer-meta">
        Inspired by common learning-product patterns: missions, mastery, study modes, and course progress.
      </p>
    </div>
  );
}
