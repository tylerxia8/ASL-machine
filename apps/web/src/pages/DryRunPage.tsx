import { useState } from "react";
import { Link } from "react-router-dom";

const STEPS = [
  "Open app, continue dev mode or register",
  "Lobby → Wave 1 (recommended)",
  "Allow camera; confirm preview + guide box",
  "Complete 5 signs with intentional correct signing",
  "Complete 2 signs with intentional wrong sign (fail + hint)",
  "Trigger retry if possible (framing hint)",
  "Retry until pass; use Next sign",
  "Open Progress; confirm attempts listed",
  "Refresh page; progress still visible",
  "Deny camera; confirm error copy + retry",
];

export default function DryRunPage() {
  const [checked, setChecked] = useState<boolean[]>(() => STEPS.map(() => false));
  const [notes, setNotes] = useState("");

  const toggle = (i: number) => {
    const next = [...checked];
    next[i] = !next[i];
    setChecked(next);
  };

  const done = checked.filter(Boolean).length;

  const exportNotes = () => {
    const lines = [
      "# Wave 1 Dry Run Results",
      `Date: ${new Date().toISOString()}`,
      `Completed: ${done}/${STEPS.length}`,
      "",
      "## Checklist",
      ...STEPS.map((s, i) => `- [${checked[i] ? "x" : " "}] ${s}`),
      "",
      "## Issues",
      notes || "(none)",
    ];
    const blob = new Blob([lines.join("\n")], { type: "text/markdown" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `wave1_dry_run_${Date.now()}.md`;
    a.click();
  };

  return (
    <div className="container">
      <Link to="/lobby">← Lobby</Link>
      <h1>Wave 1 Dry Run</h1>
      <p style={{ color: "var(--muted)" }}>
        ~30 min with 2–3 people. 25 trained signs. See <code>docs/WAVE1_DRY_RUN.md</code> for the checklist.
      </p>
      <p>
        Progress: <strong>{done}</strong> / {STEPS.length}
      </p>
      <ul style={{ listStyle: "none", padding: 0 }}>
        {STEPS.map((step, i) => (
          <li key={i} className="card" style={{ marginBottom: "0.5rem", cursor: "pointer" }} onClick={() => toggle(i)}>
            <label style={{ display: "flex", gap: "0.75rem", alignItems: "center", cursor: "pointer" }}>
              <input type="checkbox" checked={checked[i]} onChange={() => toggle(i)} />
              <span>
                {i + 1}. {step}
              </span>
            </label>
          </li>
        ))}
      </ul>
      <div className="card">
        <strong>Issue log</strong>
        <textarea
          className="input"
          style={{ minHeight: 100, marginTop: "0.5rem" }}
          placeholder="False passes, confusing hints, latency, UX issues…"
          value={notes}
          onChange={(e) => setNotes(e.target.value)}
        />
      </div>
      <div style={{ display: "flex", gap: "0.75rem", marginTop: "1rem" }}>
        <button className="btn" onClick={exportNotes}>
          Download results (.md)
        </button>
        <Link className="btn btn-secondary" to="/lobby" style={{ textDecoration: "none", textAlign: "center" }}>
          Start practice
        </Link>
      </div>
    </div>
  );
}
