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

  const continueDev = () => navigate("/lobby");

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    try {
      if (mode === "login") await signIn(email, password);
      else await signUp(email, password);
      navigate("/lobby");
    } catch (err) {
      setError((err as Error).message);
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
            value={email}
            onChange={(e) => setEmail(e.target.value)}
          />
          <input
            className="input"
            type="password"
            placeholder="Password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
          />
          {error && <p className="status-fail">{error}</p>}
          <button className="btn" type="submit" style={{ width: "100%" }}>
            {mode === "login" ? "Log in" : "Register"}
          </button>
        </form>
        <button
          className="btn btn-secondary"
          style={{ width: "100%", marginTop: "0.75rem" }}
          onClick={() => setMode(mode === "login" ? "register" : "login")}
        >
          {mode === "login" ? "Need an account? Register" : "Have an account? Log in"}
        </button>
        <hr style={{ borderColor: "var(--border)", margin: "1.25rem 0" }} />
        <button className="btn btn-secondary" style={{ width: "100%" }} onClick={continueDev}>
          Continue in dev mode ({devUserId.slice(0, 8)}…)
        </button>
        <p style={{ fontSize: "0.8rem", color: "var(--muted)", marginTop: "0.75rem" }}>
          Dev mode uses local user id when Supabase is not configured.
        </p>
      </div>
    </div>
  );
}
