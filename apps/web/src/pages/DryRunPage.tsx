import { useEffect, useState } from "react";
import { Link } from "react-router-dom";

// Mirrors the 14-step checklist in docs/WAVE1_DRY_RUN.md. Each entry has a
// short label (shown inline) plus the expected pass criterion (shown as a
// muted hint underneath so facilitators don't have to flip docs).
type Step = { label: string; pass: string };
const STEPS: Step[] = [
  { label: "Register or continue dev mode → reach Lobby", pass: "Lobby page loads with API status" },
  { label: "Lobby → Start Wave 1 session", pass: "/practice loads with first sign visible" },
  { label: "Allow camera; confirm preview + guide box", pass: "Camera feed visible, dashed box overlay" },
  { label: 'Show me the sign → read Reference panel', pass: "Handshape / movement / location render" },
  { label: "Click Record 2s clip", pass: "3-2-1 countdown → red REC border 2s → Evaluating…" },
  { label: "Complete 5 trained signs with correct signing", pass: "Pass on ≥3/5; latency <3s" },
  { label: "Sign #6: perform the wrong sign deliberately", pass: "Fail outcome + specific hint (not 'incorrect')" },
  { label: "Sign #7: partial / mis-framed sign", pass: "Retry (uncertain) OR Fail; hint may mention framing" },
  { label: "Retry + Skip to next both functional", pass: "Both load the next sign cleanly" },
  { label: 'Full catalog session: hit a "reference" sign', pass: "Reference chip + no Record button" },
  { label: "Progress page: recent attempts listed", pass: "Sign IDs + outcomes + timestamps visible" },
  { label: "Hard-refresh (Ctrl+F5); progress persists", pass: "DB state survives reload" },
  { label: "Deny camera in browser settings; reload", pass: 'Shows "Camera access was denied" + Retry button' },
  { label: "DevTools Network tab: no media in POST bodies", pass: "Only /attempts + /hint JSON, no blobs/images" },
];

const STORAGE_KEY = "dryrun_state_v1";
type Persisted = { checked: boolean[]; notes: string };

export default function DryRunPage() {
  const [checked, setChecked] = useState<boolean[]>(() => STEPS.map(() => false));
  const [notes, setNotes] = useState("");
  const [hydrated, setHydrated] = useState(false);

  useEffect(() => {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (raw) {
        const p = JSON.parse(raw) as Persisted;
        // Length-check so we don't load stale state if STEPS changed shape.
        if (Array.isArray(p.checked) && p.checked.length === STEPS.length) {
          setChecked(p.checked);
        }
        if (typeof p.notes === "string") setNotes(p.notes);
      }
    } catch {
      /* ignore corrupt persisted state */
    }
    setHydrated(true);
  }, []);

  useEffect(() => {
    if (!hydrated) return;
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ checked, notes }));
  }, [checked, notes, hydrated]);

  const toggle = (i: number) => {
    setChecked((prev) => {
      const next = [...prev];
      next[i] = !next[i];
      return next;
    });
  };

  const reset = () => {
    if (!confirm("Reset all checkboxes and clear notes?")) return;
    setChecked(STEPS.map(() => false));
    setNotes("");
  };

  const done = checked.filter(Boolean).length;

  const exportNotes = () => {
    const lines = [
      "# Wave 1 Dry Run Results",
      `Date: ${new Date().toISOString()}`,
      `Completed: ${done}/${STEPS.length}`,
      "",
      "## Checklist",
      ...STEPS.map((s, i) => `- [${checked[i] ? "x" : " "}] ${i + 1}. ${s.label} — _expected: ${s.pass}_`),
      "",
      "## Issue log",
      notes.trim() || "(none)",
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
        ~30 min with 2–3 testers. 25 trained signs. Full protocol:{" "}
        <code>docs/WAVE1_DRY_RUN.md</code>. State persists in this browser (clear it with Reset).
      </p>
      <p style={{ fontSize: "1.1rem" }}>
        Progress: <strong>{done}</strong> / {STEPS.length}
      </p>
      <ul style={{ listStyle: "none", padding: 0 }}>
        {STEPS.map((step, i) => (
          <li
            key={i}
            className="card"
            style={{ marginBottom: "0.5rem", cursor: "pointer" }}
            onClick={() => toggle(i)}
          >
            {/* `<label>` would forward clicks to the inner `<input>`, which
                would also bubble back up to this `<li>` and double-toggle.
                Use plain `<div>` + readOnly checkbox; the outer click handler
                is the single source of truth. */}
            <div style={{ display: "flex", gap: "0.75rem", alignItems: "flex-start" }}>
              <input
                type="checkbox"
                checked={checked[i]}
                readOnly
                tabIndex={-1}
                style={{ marginTop: "0.2rem" }}
              />
              <div style={{ flex: 1 }}>
                <div>
                  {i + 1}. {step.label}
                </div>
                <div style={{ fontSize: "0.8rem", color: "var(--muted)", marginTop: "0.15rem" }}>
                  ✓ {step.pass}
                </div>
              </div>
            </div>
          </li>
        ))}
      </ul>
      <div className="card">
        <strong>Issue log</strong>
        <textarea
          className="input"
          style={{ minHeight: 100, marginTop: "0.5rem" }}
          placeholder="False passes (most important), false fails, confusing hints, latency, UX blockers…"
          value={notes}
          onChange={(e) => setNotes(e.target.value)}
        />
      </div>
      <div style={{ display: "flex", gap: "0.75rem", marginTop: "1rem", flexWrap: "wrap" }}>
        <button className="btn" onClick={exportNotes}>
          Download results (.md)
        </button>
        <Link
          className="btn btn-secondary"
          to="/lobby"
          style={{ textDecoration: "none", textAlign: "center" }}
        >
          Start practice
        </Link>
        <button className="btn btn-secondary" type="button" onClick={reset}>
          Reset checklist
        </button>
      </div>
    </div>
  );
}
