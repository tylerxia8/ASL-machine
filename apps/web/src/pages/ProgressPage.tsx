import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { useAuth, getUserId } from "../lib/auth";
import { fetchProgress, ProgressSummary } from "../lib/api";

export default function ProgressPage() {
  const auth = useAuth();
  const userId = getUserId(auth);
  const [data, setData] = useState<ProgressSummary | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    fetchProgress(userId, auth.session?.access_token)
      .then(setData)
      .catch((e) => setError((e as Error).message));
  }, [userId, auth.session]);

  return (
    <div className="container">
      <Link to="/lobby">← Lobby</Link>
      <h1>Your Progress</h1>
      {error && <p className="status-fail">{error}</p>}
      {data && (
        <>
          <div className="card">
            <p>
              <strong>{data.mastered_count}</strong> signs mastered · <strong>{data.total_passes}</strong> passes ·{" "}
              <strong>{data.total_attempts}</strong> total attempts
            </p>
          </div>
          <h2>Recent attempts</h2>
          <ul style={{ listStyle: "none", padding: 0 }}>
            {data.recent_attempts.length === 0 && <li>No attempts yet.</li>}
            {data.recent_attempts.map((a) => (
              <li key={a.id} className="card" style={{ marginBottom: "0.5rem" }}>
                <span>{a.sign_id}</span> —{" "}
                <span className={`status-${a.outcome}`}>{a.outcome}</span>
                {a.confidence != null && ` (${(a.confidence * 100).toFixed(0)}%)`}
                <span style={{ color: "var(--muted)", marginLeft: "0.5rem", fontSize: "0.85rem" }}>
                  {new Date(a.created_at).toLocaleString()}
                </span>
              </li>
            ))}
          </ul>
        </>
      )}
    </div>
  );
}
