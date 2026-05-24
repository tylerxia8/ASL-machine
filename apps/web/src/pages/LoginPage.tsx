import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../lib/auth";

export default function LoginPage() {
  const { signIn, signUp, devUserId } = useAuth();
  const navigate = useNavigate();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [mode, setMode] = useState<"login" | "register">("login");
  const [submitting, setSubmitting] = useState(false);

  const continueDev = () => navigate("/lobby");

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setSubmitting(true);
    try {
      if (mode === "login") await signIn(email, password);
      else await signUp(email, password);
      navigate("/lobby");
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="container" style={{ maxWidth: 420 }}>
      <h1>ASL Practice Pilot</h1>
      <p style={{ color: "var(--muted)" }}>
        Sign in to save progress. Practice runs locally in your browser — video is not uploaded.
      </p>
      <div className="card">
        <form onSubmit={submit}>
          <input
            className="input"
            type="email"
            placeholder="Email"
            autoComplete="email"
            required
            disabled={submitting}
            value={email}
            onChange={(e) => setEmail(e.target.value)}
          />
          <input
            className="input"
            type="password"
            placeholder="Password (8+ characters)"
            autoComplete={mode === "login" ? "current-password" : "new-password"}
            required
            minLength={8}
            disabled={submitting}
            value={password}
            onChange={(e) => setPassword(e.target.value)}
          />
          {error && <p className="status-fail" role="alert">{error}</p>}
          <button className="btn" type="submit" disabled={submitting} style={{ width: "100%" }}>
            {submitting ? (mode === "login" ? "Logging in…" : "Registering…") : mode === "login" ? "Log in" : "Register"}
          </button>
        </form>
        <button
          className="btn btn-secondary"
          type="button"
          disabled={submitting}
          style={{ width: "100%", marginTop: "0.75rem" }}
          onClick={() => { setError(""); setMode(mode === "login" ? "register" : "login"); }}
        >
          {mode === "login" ? "Need an account? Register" : "Have an account? Log in"}
        </button>
        <hr style={{ borderColor: "var(--border)", margin: "1.25rem 0" }} />
        <button
          className="btn btn-secondary"
          type="button"
          disabled={submitting}
          style={{ width: "100%" }}
          onClick={continueDev}
        >
          Continue in dev mode ({devUserId.slice(0, 8)}…)
        </button>
        <p style={{ fontSize: "0.8rem", color: "var(--muted)", marginTop: "0.75rem" }}>
          Dev mode uses a local user id when Supabase is not configured. Your attempts persist in the local SQLite database.
        </p>
      </div>
    </div>
  );
}
