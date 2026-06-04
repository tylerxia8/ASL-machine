import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { fetchSigns, type SignMeta } from "../lib/api";
import { buildLearningPriorities } from "../lib/learningPlan";
import { BUNDLED_SOURCE } from "../lib/modelSource";
import { loadRecognitionCalibration, reliabilityFor, type RecognitionCalibration } from "../lib/recognitionCalibration";
import { readRecognitionFeedback, summarizeRecognitionFeedback } from "../lib/recognitionFeedback";
import type { EnsembleConfig } from "../lib/ensembleRouting";

type ModelMeta = {
  model_version?: string;
  model_size?: string;
  input_type?: string;
  num_classes?: number;
  val_accuracy?: number;
};

const PROTECTED_SIGNS = ["deaf", "where", "sleep", "who", "five"];

export default function ModelHealthPage() {
  const [meta, setMeta] = useState<ModelMeta | null>(null);
  const [calibration, setCalibration] = useState<RecognitionCalibration | null>(null);
  const [ensemble, setEnsemble] = useState<EnsembleConfig | null>(null);
  const [signs, setSigns] = useState<SignMeta[]>([]);
  const [feedback, setFeedback] = useState(readRecognitionFeedback);

  useEffect(() => {
    fetch(BUNDLED_SOURCE.metaUrl).then((r) => (r.ok ? r.json() : null)).then(setMeta).catch(() => setMeta(null));
    fetch("/models/ensemble_config.json", { cache: "no-store" })
      .then((r) => (r.ok ? r.json() : null))
      .then(setEnsemble)
      .catch(() => setEnsemble(null));
    loadRecognitionCalibration().then(setCalibration).catch(() => setCalibration(null));
    fetchSigns(1).then(setSigns).catch(() => setSigns([]));
    setFeedback(readRecognitionFeedback());
  }, []);

  const feedbackSummary = useMemo(() => summarizeRecognitionFeedback(feedback), [feedback]);
  const thresholdRows = Object.entries(calibration?.thresholds ?? {}).map(([signId, row]) => ({
    signId,
    row,
    reliability: reliabilityFor(row),
  }));
  const health = {
    strong: thresholdRows.filter((r) => r.reliability === "strong").length,
    watch: thresholdRows.filter((r) => r.reliability === "watch").length,
    weak: thresholdRows.filter((r) => r.reliability === "weak").length,
  };
  const protectedRows = PROTECTED_SIGNS.map((signId) => {
    const threshold = calibration?.thresholds?.[signId];
    return { signId, threshold, reliability: reliabilityFor(threshold) };
  });
  const learningPriorities = buildLearningPriorities(signs, calibration, feedbackSummary).slice(0, 10);
  const routeRows = Object.entries(feedbackSummary.routed.byRoute).map(([routeId, row]) => ({
    routeId,
    ...row,
    accuracy: row.total ? row.accepted / row.total : 0,
  }));
  const qualityRows = Object.entries(feedbackSummary.bySign)
    .map(([signId, row]) => ({
      signId,
      ...row,
      wrongRate: row.total ? row.rejected / row.total : 0,
      qualityIssues: row.lowAgreement + row.lowTracking,
    }))
    .filter((row) => row.total >= 2)
    .sort((a, b) => b.wrongRate - a.wrongRate || b.qualityIssues - a.qualityIssues)
    .slice(0, 10);

  return (
    <div className="container">
      <Link to="/lobby">{"<-"} Lobby</Link>
      <h1>Model Health</h1>

      <div className="card">
        <div className="metric-grid">
          <div>
            <span className="metric-label">Production model</span>
            <strong><code>{meta?.model_version ?? "unknown"}</code></strong>
          </div>
          <div>
            <span className="metric-label">Input</span>
            <strong>{meta?.input_type ?? "unknown"}</strong>
          </div>
          <div>
            <span className="metric-label">Validation</span>
            <strong>{typeof meta?.val_accuracy === "number" ? `${(meta.val_accuracy * 100).toFixed(1)}%` : "n/a"}</strong>
          </div>
          <div>
            <span className="metric-label">Eval</span>
            <strong>{typeof calibration?.accuracy === "number" ? `${(calibration.accuracy * 100).toFixed(1)}%` : "n/a"}</strong>
          </div>
        </div>
      </div>

      <div className="card">
        <strong>Ensemble routing</strong>
        {!ensemble?.enabled ? (
          <p style={{ color: "var(--muted)" }}>Specialist routing is disabled.</p>
        ) : (
          <>
            <p style={{ color: "var(--muted)" }}>
              Primary source: <code>{ensemble.primarySourceId}</code>
            </p>
            <table className="compact-table">
              <thead>
                <tr>
                  <th>Specialist</th>
                  <th>Release</th>
                  <th>Allowed signs</th>
                  <th>Gate</th>
                </tr>
              </thead>
              <tbody>
                {(ensemble.specialists ?? []).map((route) => (
                  <tr key={route.id}>
                    <td><code>{route.id}</code></td>
                    <td><code>{route.releaseTag}</code></td>
                    <td>{route.allowedSigns.join(", ")}</td>
                    <td>
                      conf &gt;= {route.minConfidence ?? 0.5}, margin &gt;= {route.minMargin ?? 0}, primary &lt;={" "}
                      {route.maxPrimaryConfidence ?? 1}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </>
        )}
      </div>

      <div className="card">
        <strong>Routed feedback</strong>
        {feedbackSummary.routed.total === 0 ? (
          <p style={{ color: "var(--muted)" }}>
            No specialist overrides have been labeled yet. Practice routed signs and answer “Was recognition right?”
          </p>
        ) : (
          <table className="compact-table">
            <thead>
              <tr>
                <th>Route</th>
                <th>Votes</th>
                <th>Right</th>
                <th>Wrong</th>
                <th>Accuracy</th>
              </tr>
            </thead>
            <tbody>
              {routeRows.map((row) => (
                <tr key={row.routeId}>
                  <td><code>{row.routeId}</code></td>
                  <td>{row.total}</td>
                  <td>{row.accepted}</td>
                  <td>{row.rejected}</td>
                  <td>{Math.round(row.accuracy * 100)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      <div className="card">
        <strong>Reliability</strong>
        <div className="metric-grid" style={{ marginTop: "0.75rem" }}>
          <div>
            <span className="metric-label">Strong</span>
            <strong>{health.strong}</strong>
          </div>
          <div>
            <span className="metric-label">Watch</span>
            <strong className={health.watch ? "status-retry" : undefined}>{health.watch}</strong>
          </div>
          <div>
            <span className="metric-label">Weak</span>
            <strong className={health.weak ? "status-fail" : undefined}>{health.weak}</strong>
          </div>
        </div>
        <strong style={{ display: "block", marginTop: "1rem" }}>Protected signs</strong>
        <table className="compact-table">
          <thead>
            <tr>
              <th>Sign</th>
              <th>Status</th>
              <th>F1</th>
              <th>Support</th>
            </tr>
          </thead>
          <tbody>
            {protectedRows.map((row) => (
              <tr key={row.signId}>
                <td><code>{row.signId}</code></td>
                <td>
                  <span className={`status-${row.reliability === "weak" ? "fail" : row.reliability === "watch" ? "retry" : "pass"}`}>
                    {row.reliability}
                  </span>
                </td>
                <td>{typeof row.threshold?.f1 === "number" ? `${Math.round(row.threshold.f1 * 100)}%` : "n/a"}</td>
                <td>{row.threshold?.support ?? "n/a"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="card">
        <strong>Next data priorities</strong>
        <table className="compact-table">
          <thead>
            <tr>
              <th>Sign</th>
              <th>Reason</th>
              <th>F1</th>
              <th>Local misses</th>
            </tr>
          </thead>
          <tbody>
            {learningPriorities.map((row) => (
              <tr key={row.sign.sign_id}>
                <td><code>{row.sign.sign_id}</code></td>
                <td>{row.reasons.slice(0, 2).join(", ")}</td>
                <td>{typeof row.f1 === "number" ? `${Math.round(row.f1 * 100)}%` : "n/a"}</td>
                <td>{row.localTotal ? `${row.localWrong}/${row.localTotal}` : "none"}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <div className="button-row">
          <Link to="/practice" className="btn">Practice routed signs</Link>
          <Link to="/capture" className="btn btn-secondary">Capture more clips</Link>
        </div>
      </div>

      <div className="card">
        <strong>Local quality signals</strong>
        {qualityRows.length === 0 ? (
          <p style={{ color: "var(--muted)" }}>No repeated local recognition feedback yet.</p>
        ) : (
          <table className="compact-table">
            <thead>
              <tr>
                <th>Sign</th>
                <th>Wrong</th>
                <th>Agreement</th>
                <th>Tracking</th>
                <th>Quality flags</th>
              </tr>
            </thead>
            <tbody>
              {qualityRows.map((row) => (
                <tr key={row.signId}>
                  <td><code>{row.signId}</code></td>
                  <td>{row.rejected}/{row.total}</td>
                  <td>{row.avgAgreement === null ? "n/a" : `${Math.round(row.avgAgreement * 100)}%`}</td>
                  <td>{row.avgTrackingRatio === null ? "n/a" : `${Math.round(row.avgTrackingRatio * 100)}%`}</td>
                  <td>
                    {row.lowAgreement} shaky, {row.lowTracking} tracking
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
